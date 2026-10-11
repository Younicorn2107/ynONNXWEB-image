/** A prompt to a picture: the catalogue's pipelines behind one class. */

import { ConfigurationError, defaultLogger } from "@youni/onnxweb-core/core";

import { DEFAULT_IMAGE_MODEL, IMAGE_CATALOGUE, type ImageModelEntry } from "./catalogue.js";
import type { GeneratedImage, GenerateImageOptions, ImageGeneratorOptions, Pipeline } from "./pipeline.js";
import { SdTurboPipeline } from "./sdTurbo.js";
import { ZImagePipeline } from "./zImage.js";

/** The most pictures `generateBatch` draws in one call. */
export const MAX_BATCH = 9;

export class ImageGenerator {
  private constructor(
    readonly name: string,
    readonly entry: ImageModelEntry,
    private readonly pipeline: Pipeline,
  ) {}

  static async load(options: ImageGeneratorOptions): Promise<ImageGenerator> {
    const name = options.model ?? DEFAULT_IMAGE_MODEL;
    const entry = IMAGE_CATALOGUE[name];
    if (!entry) throw new ConfigurationError(`'${name}' is not an image model this library knows (${Object.keys(IMAGE_CATALOGUE).join(", ")})`);
    const withLogger = { ...options, logger: options.logger ?? defaultLogger };
    const pipeline = entry.pipeline === "z-image" ? await ZImagePipeline.load(name, entry, withLogger) : await SdTurboPipeline.load(name, entry, withLogger);
    return new ImageGenerator(name, entry, pipeline);
  }

  get backend(): string {
    return this.pipeline.backend;
  }

  /**
   * `count` pictures for `prompt` (1 to 9), one after the other, with seeds `seed`, `seed + 1`,
   * ...: any one of them is drawn again alone with its own seed.
   */
  async generateBatch(prompt: string, count: number, options: GenerateImageOptions & { onImage?: (image: GeneratedImage, index: number) => void } = {}): Promise<GeneratedImage[]> {
    if (!Number.isInteger(count) || count < 1 || count > MAX_BATCH) throw new ConfigurationError(`a batch is 1 to ${MAX_BATCH} pictures (asked ${count})`);
    const seed = options.seed ?? Math.floor(Math.random() * 2 ** 31);
    const images: GeneratedImage[] = [];
    for (let i = 0; i < count; i += 1) {
      const image = await this.pipeline.generate(prompt, { ...options, seed: (seed + i) >>> 0 });
      images.push(image);
      options.onImage?.(image, i);
    }
    return images;
  }

  /** One picture for `prompt`. */
  generate(prompt: string, options: GenerateImageOptions = {}): Promise<GeneratedImage> {
    return this.pipeline.generate(prompt, options);
  }

  close(): Promise<void> {
    return this.pipeline.close();
  }
}
