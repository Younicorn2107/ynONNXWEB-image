/** A prompt to a picture: the catalogue's pipelines behind one class. */

import { ConfigurationError, defaultLogger } from "@youni/onnxweb-core/core";

import { DEFAULT_IMAGE_MODEL, IMAGE_CATALOGUE, type ImageModelEntry } from "./catalogue.js";
import type { GeneratedImage, GenerateImageOptions, ImageGeneratorOptions, Pipeline } from "./pipeline.js";
import { SdTurboPipeline } from "./sdTurbo.js";
import { ZImagePipeline } from "./zImage.js";

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

  /** One picture for `prompt`. */
  generate(prompt: string, options: GenerateImageOptions = {}): Promise<GeneratedImage> {
    return this.pipeline.generate(prompt, options);
  }

  close(): Promise<void> {
    return this.pipeline.close();
  }
}
