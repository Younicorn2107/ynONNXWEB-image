/** SD-Turbo: a prompt to a picture in one to four diffusion steps. */

import { huggingFaceResolver, loadModel, type ModelBundle } from "@youni/onnxweb-core";
import { ConfigurationError, Tokenizer, type TensorFactory } from "@youni/onnxweb-core/core";

import type { SdTurboEntry } from "./catalogue.js";
import { checkAborted, gaussian, toRgba, type GeneratedImage, type GenerateImageOptions, type ImageGeneratorOptions, type Pipeline } from "./pipeline.js";

/** CLIP's ids for `prompt`: start, the words, end, then padding — `length` in all. */
export function promptIds(tokenizer: Pick<Tokenizer, "encode">, prompt: string, entry: Pick<SdTurboEntry, "maxTokens" | "padId">): Int32Array {
  const START = 49406;
  const END = 49407;
  const words = tokenizer.encode(prompt.toLowerCase(), { addSpecialTokens: false }).filter((id) => id !== START && id !== END);
  const ids = new Int32Array(entry.maxTokens).fill(entry.padId);
  ids[0] = START;
  const kept = words.slice(0, entry.maxTokens - 2);
  ids.set(kept, 1);
  ids[kept.length + 1] = END;
  return ids;
}

/** The noise level sigma of each training timestep (`scaled_linear` betas). */
export function sdSigmas(entry: Pick<SdTurboEntry, "betaStart" | "betaEnd" | "trainSteps">): Float64Array {
  const sigmas = new Float64Array(entry.trainSteps);
  const a = Math.sqrt(entry.betaStart);
  const b = Math.sqrt(entry.betaEnd);
  let alphaBar = 1;
  for (let t = 0; t < entry.trainSteps; t += 1) {
    const beta = (a + ((b - a) * t) / (entry.trainSteps - 1)) ** 2;
    alphaBar *= 1 - beta;
    sigmas[t] = Math.sqrt((1 - alphaBar) / alphaBar);
  }
  return sigmas;
}

/** The timesteps of `steps` steps, "trailing": from the last training step down, evenly. */
export function sdTimesteps(steps: number, trainSteps: number): number[] {
  return Array.from({ length: steps }, (_, i) => Math.round(trainSteps - (i * trainSteps) / steps) - 1);
}

export class SdTurboPipeline implements Pipeline {
  private constructor(
    private readonly entry: SdTurboEntry,
    private readonly tokenizer: Tokenizer,
    private readonly text: ModelBundle,
    private readonly unet: ModelBundle,
    private readonly vae: ModelBundle,
    private readonly Tensor: TensorFactory,
  ) {}

  static async load(name: string, entry: SdTurboEntry, options: ImageGeneratorOptions): Promise<SdTurboPipeline> {
    const { Tensor, endpoint, model: _m, textEncoderBackends: _t, lowMemory: _l, ...rest } = options;
    const at = (subfolder: string) => huggingFaceResolver({ repo: entry.repo, subfolder, manifest: false, ...(endpoint ? { endpoint } : {}) });

    const part = (subfolder: string) => loadModel({ ...rest, resolver: at(subfolder), model: { name: `${entry.repo}#${subfolder}`, computeType: "float32" } });
    const tokenizerSource = huggingFaceResolver({ repo: entry.tokenizerRepo, manifest: false, ...(endpoint ? { endpoint } : {}) });
    const tokenizerUrl = tokenizerSource.assetUrl({ name: entry.tokenizerRepo, computeType: "float32" }, "tokenizer.json", "root").url;
    const tokenizer = Tokenizer.fromJson(await (await fetch(tokenizerUrl)).text(), null);
    const loaded: ModelBundle[] = [];
    try {
      for (const subfolder of ["text_encoder", "unet", "vae_decoder"]) loaded.push(await part(subfolder));
    } catch (error) {
      await Promise.all(loaded.map((b) => b.close()));
      throw error;
    }
    options.logger?.info?.(`${name} loaded on ${loaded[1]!.backend}`);
    return new SdTurboPipeline(entry, tokenizer, loaded[0]!, loaded[1]!, loaded[2]!, Tensor);
  }

  get backend(): string {
    return this.unet.backend;
  }

  async #encode(text: string): Promise<unknown> {
    const session = this.text.only;
    const ids = promptIds(this.tokenizer, text, this.entry);
    const encoded = await session.run({ [session.inputs[0]!.name]: new this.Tensor("int32", ids, [1, this.entry.maxTokens]) });
    return encoded["last_hidden_state"] ?? encoded[session.outputs[0]!.name];
  }

  async generate(prompt: string, options: GenerateImageOptions): Promise<GeneratedImage> {
    const { entry, Tensor } = this;
    if ((options.width ?? entry.size) !== entry.size || (options.height ?? entry.size) !== entry.size) {
      throw new ConfigurationError(`SD-Turbo draws ${entry.size} x ${entry.size} only`);
    }
    const steps = options.steps ?? entry.steps;
    if (!Number.isInteger(steps) || steps < 1 || steps > 50) throw new ConfigurationError(`steps must be a whole number from 1 to 50 (asked ${steps}); SD-Turbo is made for 1 to 4`);
    const guidance = options.guidanceScale ?? (options.negativePrompt ? 2 : 0);
    const seed = options.seed ?? Math.floor(Math.random() * 2 ** 31);
    const started = performance.now();

    const positive = await this.#encode(prompt);

    const negative = guidance > 0 ? await this.#encode(options.negativePrompt ?? "") : null;
    const textMs = performance.now() - started;
    checkAborted(options.signal);

    const side = entry.size / 8;
    const count = entry.latentChannels * side * side;
    const shape = [1, entry.latentChannels, side, side];
    const table = sdSigmas(entry);
    const timesteps = sdTimesteps(steps, entry.trainSteps);
    const sigmas = [...timesteps.map((t) => table[t]!), 0];
    const latent = gaussian(count, seed);
    for (let i = 0; i < count; i += 1) latent[i] = latent[i]! * sigmas[0]!;

    const unet = this.unet.only;
    const unetStarted = performance.now();
    const input = new Float32Array(count);
    for (let step = 0; step < steps; step += 1) {
      const sigma = sigmas[step]!;
      const next = sigmas[step + 1]!;
      const scale = 1 / Math.sqrt(sigma * sigma + 1);
      for (let i = 0; i < count; i += 1) input[i] = latent[i]! * scale;
      const predict = async (hidden: unknown) => {
        const out = await unet.run({
          sample: new Tensor("float32", input, shape),
          timestep: new Tensor("int64", BigInt64Array.of(BigInt(timesteps[step]!)), [1]),
          encoder_hidden_states: hidden,
        });
        return ((out["out_sample"] ?? out[unet.outputs[0]!.name]) as { data: Float32Array }).data;
      };
      let eps = await predict(positive);
      if (negative) {
        const uncond = await predict(negative);
        const guided = new Float32Array(count);
        for (let i = 0; i < count; i += 1) guided[i] = uncond[i]! + guidance * (eps[i]! - uncond[i]!);
        eps = guided;
      }

      const up = next > 0 ? Math.sqrt((next * next * (sigma * sigma - next * next)) / (sigma * sigma)) : 0;
      const down = Math.sqrt(next * next - up * up);
      const fresh = up > 0 ? gaussian(count, (seed + 7919 * (step + 1)) >>> 0) : null;
      for (let i = 0; i < count; i += 1) latent[i] = latent[i]! + eps[i]! * (down - sigma) + (fresh ? fresh[i]! * up : 0);
      options.onStep?.(step, steps);
      checkAborted(options.signal);
    }
    const unetMs = performance.now() - unetStarted;

    for (let i = 0; i < count; i += 1) latent[i] = latent[i]! / entry.vaeScale;
    const vaeSession = this.vae.only;
    const vaeStarted = performance.now();
    const decoded = await vaeSession.run({ [vaeSession.inputs[0]!.name]: new Tensor("float32", latent, shape) });
    const pixels = (decoded["sample"] ?? decoded[vaeSession.outputs[0]!.name]) as { data: Float32Array };
    const vaeMs = performance.now() - vaeStarted;

    return {
      data: toRgba(pixels.data, entry.size, entry.size),
      width: entry.size,
      height: entry.size,
      seed,
      steps,
      timings: { text: Math.round(textMs), denoise: Math.round(unetMs), vae: Math.round(vaeMs), total: Math.round(performance.now() - started) },
    };
  }

  async close(): Promise<void> {
    await Promise.all([this.text.close(), this.unet.close(), this.vae.close()]);
  }
}
