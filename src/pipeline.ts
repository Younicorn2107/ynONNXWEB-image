/**
 * What every pipeline shares: the options, the result, seeded noise and the conversion of decoded
 * planes to pixels.
 */

import type { LoadOptions } from "@youni/onnxweb-core";
import type { Logger, TensorFactory } from "@youni/onnxweb-core/core";

export interface ImageGeneratorOptions extends Omit<LoadOptions, "model" | "computeType" | "resolver"> {
  Tensor: TensorFactory;
  /** A catalogue name (`sd-turbo`, the default; `z-image-turbo`). */
  model?: string;
  /** The Hub, or a mirror of it. */
  endpoint?: string;
  logger?: Logger;
  /**
   * Z-Image: where the text encoder runs (`"webgpu"` by default with the rest; `"wasm"` puts its
   * 2.2 GB in main memory and leaves the GPU's to the transformer — slower to encode, once per
   * prompt).
   */
  textEncoderBackends?: string;
  /**
   * Z-Image: load the text encoder for each new prompt and release it after, so it never shares the
   * GPU's memory with the transformer. For a GPU with less than about 8 GB; each new prompt then
   * waits for the load.
   */
  lowMemory?: boolean;
}

export interface GenerateImageOptions {
  /** The noise's seed: the same seed, the same picture. Random when absent. */
  seed?: number;
  /** Denoising steps (Z-Image: 9 by default; SD-Turbo always takes one). */
  steps?: number;
  /** Output size in pixels (Z-Image: multiples of 16; SD-Turbo: 512 only). */
  width?: number;
  height?: number;
  signal?: AbortSignal;
  /** After each denoising step: its index and the step count. */
  onStep?: (step: number, steps: number) => void;
}

export interface GeneratedImage {
  /** RGBA, row by row (an `ImageData`'s layout). */
  data: Uint8ClampedArray;
  width: number;
  height: number;
  seed: number;
  steps: number;
  /** Time per stage, ms. */
  timings: { text: number; denoise: number; vae: number; total: number };
}

export interface Pipeline {
  readonly backend: string;
  generate(prompt: string, options: GenerateImageOptions): Promise<GeneratedImage>;
  close(): Promise<void>;
}

/** A small, seedable generator (mulberry32). */
function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal values (Box–Muller). */
export function gaussian(count: number, seed: number): Float32Array {
  const next = random(seed);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 2) {
    const u = Math.max(next(), 1e-12);
    const v = next();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < count) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

/** RGB planes in [−1, 1] to RGBA bytes. */
export function toRgba(planes: ArrayLike<number>, width: number, height: number): Uint8ClampedArray {
  const plane = width * height;
  const out = new Uint8ClampedArray(plane * 4);
  for (let i = 0; i < plane; i += 1) {
    for (let c = 0; c < 3; c += 1) out[i * 4 + c] = Math.round((planes[c * plane + i]! / 2 + 0.5) * 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}

export function checkAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw new DOMException("image generation was cancelled", "AbortError");
}
