/** Z-Image-Turbo: a diffusion transformer guided by a language model. */

import { graphAlias, huggingFaceResolver, loadModel, type ModelBundle, type ModelResolver } from "@youni/onnxweb-core";
import { ConfigurationError, Tokenizer, type Logger, type TensorFactory } from "@youni/onnxweb-core/core";

import type { ZImageEntry } from "./catalogue.js";
import { checkAborted, gaussian, toRgba, type GeneratedImage, type GenerateImageOptions, type ImageGeneratorOptions, type Pipeline } from "./pipeline.js";

interface OrtTensor {
  data: ArrayLike<number>;
  dims: readonly number[];
  location?: string;
  dispose?(): void;
}

/**
 * Qwen3's chat template for one user turn, the assistant's turn opened (thinking on: no empty think
 * block).
 */
export function zImagePrompt(prompt: string): string {
  return `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
}

/**
 * The transformer's time input for each step: the flow-matching schedule (timesteps spread evenly
 * between the shifted extremes, shifted again, as the scheduler does without dynamic shifting),
 * given as `1 − σ`; the last step at 1.
 */
export function zImageTimesteps(steps: number, shift: number, trainSteps = 1000): Float32Array {
  const shifted = (s: number) => (shift * s) / (1 + (shift - 1) * s);
  const sigmaMax = shifted(1);
  const sigmaMin = shifted(1 / trainSteps);
  const out = new Float32Array(steps);
  for (let i = 0; i < steps; i += 1) {
    const t = steps === 1 ? sigmaMax * trainSteps : sigmaMax * trainSteps + ((sigmaMin - sigmaMax) * trainSteps * i) / (steps - 1);
    out[i] = Math.fround((trainSteps - shifted(t / trainSteps) * trainSteps) / trainSteps);
  }
  out[steps - 1] = 1;
  return out;
}

export class ZImagePipeline implements Pipeline {
  #encoder: ModelBundle | null;
  #cached: { prompt: string; hidden: OrtTensor } | null = null;

  private constructor(
    private readonly entry: ZImageEntry,
    private readonly tokenizer: Tokenizer,
    private readonly loadEncoder: () => Promise<ModelBundle>,
    encoder: ModelBundle | null,
    private readonly shards: ModelBundle[],
    private readonly scheduler: ModelBundle,
    private readonly vaePre: ModelBundle,
    private readonly vae: ModelBundle,
    private readonly Tensor: TensorFactory,
    private readonly lowMemory: boolean,
    private readonly logger: Logger,
  ) {
    this.#encoder = encoder;
  }

  static async load(name: string, entry: ZImageEntry, options: ImageGeneratorOptions): Promise<ZImagePipeline> {
    const { Tensor, endpoint, model: _m, logger, textEncoderBackends, lowMemory = false, backends, ...rest } = options;
    const base: ModelResolver = huggingFaceResolver({ repo: entry.repo, subfolder: "onnx", manifest: false, ...(endpoint ? { endpoint } : {}) });
    const part = (stem: string, extra: Record<string, unknown> = {}) =>
      loadModel({ ...rest, ...(backends ? { backends } : {}), ...(logger ? { logger } : {}), resolver: graphAlias(base, stem), model: { name: `${entry.repo}#${stem}`, computeType: "float32" }, ...extra } as never);

    const tokenizerUrl = huggingFaceResolver({ repo: entry.repo, manifest: false, ...(endpoint ? { endpoint } : {}) }).assetUrl({ name: entry.repo, computeType: "float32" }, `${entry.tokenizerFolder}/tokenizer.json`, "root").url;
    const tokenizer = Tokenizer.fromJson(await (await fetch(tokenizerUrl)).text(), null);

    const loadEncoder = () => part(entry.graphs.textEncoder, textEncoderBackends ? { backends: textEncoderBackends } : {});
    const loaded: ModelBundle[] = [];
    const keep = <T extends ModelBundle>(bundle: T) => (loaded.push(bundle), bundle);
    try {
      const encoder = lowMemory ? null : keep(await loadEncoder());

      const shards: ModelBundle[] = [];
      for (const [i, stem] of entry.graphs.transformer.entries()) {
        const last = i === entry.graphs.transformer.length - 1;
        shards.push(keep(await part(stem, last ? {} : { session: { ...rest.session, preferredOutputLocation: "gpu-buffer" } })));
      }

      const scheduler = keep(await part(entry.graphs.scheduler, { backends: "wasm" }));
      const vaePre = keep(await part(entry.graphs.vaePreProcess, { backends: "wasm" }));
      const vae = keep(await part(entry.graphs.vaeDecoder));
      logger?.info?.(`${name} loaded: transformer on ${shards[0]!.backend}, text encoder on ${encoder?.backend ?? "demand"}`);
      return new ZImagePipeline(entry, tokenizer, loadEncoder, encoder, shards, scheduler, vaePre, vae, Tensor, lowMemory, logger ?? { info: () => undefined, warn: () => undefined } as unknown as Logger);
    } catch (error) {
      await Promise.all(loaded.map((b) => b.close()));
      throw error;
    }
  }

  get backend(): string {
    return this.shards[0]!.backend;
  }

  /** The prompt's hidden states, kept for the next picture of the same prompt. */
  async #encode(prompt: string): Promise<OrtTensor> {
    if (this.#cached?.prompt === prompt) return this.#cached.hidden;
    this.#cached?.hidden.dispose?.();
    this.#cached = null;
    const ids = this.tokenizer.encode(zImagePrompt(prompt), { addSpecialTokens: false }).slice(0, this.entry.maxTokens);
    const encoder = this.#encoder ?? (await this.loadEncoder());
    try {
      const session = encoder.only;
      const out = (await session.run({
        input_ids: new this.Tensor("int64", BigInt64Array.from(ids, (id) => BigInt(id)), [1, ids.length]),
        attention_mask: new this.Tensor("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
      })) as Record<string, OrtTensor>;
      const hidden = out[session.outputs[0]!.name]!;
      this.#cached = { prompt, hidden };
      return hidden;
    } finally {
      if (this.lowMemory) await encoder.close();
    }
  }

  async generate(prompt: string, options: GenerateImageOptions): Promise<GeneratedImage> {
    const { entry, Tensor } = this;
    const width = options.width ?? entry.size;
    const height = options.height ?? entry.size;
    if (width % entry.sideMultiple || height % entry.sideMultiple) throw new ConfigurationError(`Z-Image draws sides that are multiples of ${entry.sideMultiple} (asked ${width} × ${height})`);
    const steps = options.steps ?? entry.steps;
    const seed = options.seed ?? Math.floor(Math.random() * 2 ** 31);
    const started = performance.now();

    const hidden = await this.#encode(prompt);
    const textMs = performance.now() - started;
    checkAborted(options.signal);

    const h = height / 8;
    const w = width / 8;
    const shape = [1, entry.latentChannels, 1, h, w];
    let latents = gaussian(entry.latentChannels * h * w, seed);
    const timesteps = zImageTimesteps(steps, entry.shift);
    const denoiseStarted = performance.now();
    for (let step = 0; step < steps; step += 1) {
      const velocity = await this.#transformer(hidden, new Tensor("float32", latents, shape) as unknown as OrtTensor, timesteps[step]!);
      const scheduler = this.scheduler.only;
      const next = (await scheduler.run({
        noise_pred: velocity,
        latents: new Tensor("float32", latents, shape),
        step_info: new Tensor("float32", Float32Array.of(step, steps), [2]),
      })) as Record<string, OrtTensor>;
      latents = Float32Array.from(next[scheduler.outputs[0]!.name]!.data);
      options.onStep?.(step, steps);
      checkAborted(options.signal);
    }
    const denoiseMs = performance.now() - denoiseStarted;

    const vaeStarted = performance.now();
    const pre = this.vaePre.only;
    const scaled = (await pre.run({ [pre.inputs[0]!.name]: new Tensor("float32", latents, shape) })) as Record<string, OrtTensor>;
    const vae = this.vae.only;
    const decoded = (await vae.run({ [vae.inputs[0]!.name]: scaled[pre.outputs[0]!.name] })) as Record<string, OrtTensor>;
    const pixels = decoded[vae.outputs[0]!.name]!;
    const vaeMs = performance.now() - vaeStarted;

    return {
      data: toRgba(pixels.data, width, height),
      width,
      height,
      seed,
      steps,
      timings: { text: Math.round(textMs), denoise: Math.round(denoiseMs), vae: Math.round(vaeMs), total: Math.round(performance.now() - started) },
    };
  }

  /** One pass through the shards; the velocity, on the CPU. */
  async #transformer(hidden: OrtTensor, latents: OrtTensor, timestep: number): Promise<OrtTensor> {
    const known = new Map<string, OrtTensor>([
      ["encoder_hidden_states", hidden],
      ["hidden_states", latents],
      ["timestep", new this.Tensor("float32", Float32Array.of(timestep), [1]) as OrtTensor],
    ]);
    const made: OrtTensor[] = [];
    try {
      let out: Record<string, OrtTensor> = {};
      for (const shard of this.shards) {
        const session = shard.only;
        const feeds: Record<string, OrtTensor> = {};
        for (const input of session.inputs) {
          const tensor = known.get(input.name);
          if (!tensor) throw new ConfigurationError(`transformer shard ${session.filename} reads '${input.name}', which no earlier shard produced`);
          feeds[input.name] = tensor;
        }
        out = (await session.run(feeds)) as Record<string, OrtTensor>;
        for (const [name, tensor] of Object.entries(out)) {
          known.set(name, tensor);
          made.push(tensor);
        }
      }
      const last = this.shards.at(-1)!.only;
      return out[last.outputs[0]!.name]!;
    } finally {
      for (const tensor of made) if (tensor.location === "gpu-buffer") tensor.dispose?.();
    }
  }

  async close(): Promise<void> {
    this.#cached?.hidden.dispose?.();
    this.#cached = null;
    await Promise.all([this.#encoder, ...this.shards, this.scheduler, this.vaePre, this.vae].map((b) => b?.close()));
  }
}
