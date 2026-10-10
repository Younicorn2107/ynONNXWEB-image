/** SD-Turbo: a prompt to a picture in one diffusion step. */

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

  async generate(prompt: string, options: GenerateImageOptions): Promise<GeneratedImage> {
    const { entry, Tensor } = this;
    if ((options.width ?? entry.size) !== entry.size || (options.height ?? entry.size) !== entry.size) {
      throw new ConfigurationError(`SD-Turbo draws ${entry.size} × ${entry.size} only`);
    }
    const seed = options.seed ?? Math.floor(Math.random() * 2 ** 31);
    const started = performance.now();

    const textSession = this.text.only;
    const ids = promptIds(this.tokenizer, prompt, entry);
    const encoded = await textSession.run({ [textSession.inputs[0]!.name]: new Tensor("int32", ids, [1, entry.maxTokens]) });
    const hidden = encoded["last_hidden_state"] ?? encoded[textSession.outputs[0]!.name];
    const textMs = performance.now() - started;
    checkAborted(options.signal);

    const side = entry.size / 8;
    const count = entry.latentChannels * side * side;
    const noise = gaussian(count, seed);
    const scale = 1 / Math.sqrt(entry.sigma ** 2 + 1);
    const input = new Float32Array(count);
    for (let i = 0; i < count; i += 1) input[i] = noise[i]! * entry.sigma * scale;
    const unetSession = this.unet.only;
    const unetStarted = performance.now();
    const predicted = await unetSession.run({
      sample: new Tensor("float32", input, [1, entry.latentChannels, side, side]),
      timestep: new Tensor("int64", BigInt64Array.of(BigInt(entry.timestep)), [1]),
      encoder_hidden_states: hidden,
    });
    const eps = (predicted["out_sample"] ?? predicted[unetSession.outputs[0]!.name]) as { data: Float32Array };
    const unetMs = performance.now() - unetStarted;
    options.onStep?.(0, 1);
    checkAborted(options.signal);

    const latent = new Float32Array(count);
    for (let i = 0; i < count; i += 1) latent[i] = (noise[i]! * entry.sigma - entry.sigma * eps.data[i]!) / entry.vaeScale;
    const vaeSession = this.vae.only;
    const vaeStarted = performance.now();
    const decoded = await vaeSession.run({ [vaeSession.inputs[0]!.name]: new Tensor("float32", latent, [1, entry.latentChannels, side, side]) });
    const pixels = (decoded["sample"] ?? decoded[vaeSession.outputs[0]!.name]) as { data: Float32Array };
    const vaeMs = performance.now() - vaeStarted;

    return {
      data: toRgba(pixels.data, entry.size, entry.size),
      width: entry.size,
      height: entry.size,
      seed,
      steps: 1,
      timings: { text: Math.round(textMs), denoise: Math.round(unetMs), vae: Math.round(vaeMs), total: Math.round(performance.now() - started) },
    };
  }

  async close(): Promise<void> {
    await Promise.all([this.text.close(), this.unet.close(), this.vae.close()]);
  }
}
