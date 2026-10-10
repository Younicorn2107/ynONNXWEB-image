/** The image generators this library knows by name, one entry per pipeline: */

interface BaseEntry {
  /** Hugging Face repository of the ONNX graphs. */
  repo: string;
  /** Default output side, in pixels. */
  size: number;
  /** Download size, MB. */
  sizeMb: number;
  licence: string;
  notes: string;
}

export interface SdTurboEntry extends BaseEntry {
  pipeline: "sd-turbo";
  /**
   * Where a `tokenizer.json` for its CLIP vocabulary is published (the repository ships vocab and
   * merges only).
   */
  tokenizerRepo: string;
  latentChannels: number;
  /** The VAE's latent scaling (`scaling_factor` of its config). */
  vaeScale: number;
  /** The noise level of the single step (Euler, timestep 999). */
  sigma: number;
  timestep: number;
  /** Prompt tokens, start and end included. */
  maxTokens: number;
  /** The token prompts are padded with (SD 2.x: "!", id 0). */
  padId: number;
}

export interface ZImageEntry extends BaseEntry {
  pipeline: "z-image";
  /** The graphs' file stems in `onnx/`. */
  graphs: {
    textEncoder: string;
    /**
     * The transformer, cut into consecutive shards (each one's outputs feed the next ones by name).
     */
    transformer: string[];
    scheduler: string;
    vaePreProcess: string;
    vaeDecoder: string;
  };
  /** The tokenizer's folder in the repository. */
  tokenizerFolder: string;
  /** Prompt tokens kept, chat template included. */
  maxTokens: number;
  latentChannels: number;
  /** Denoising steps by default. */
  steps: number;
  /** The flow-matching schedule's shift. */
  shift: number;
  /** Sides must be a multiple of this (VAE 8 × patch 2). */
  sideMultiple: number;
}

export type ImageModelEntry = SdTurboEntry | ZImageEntry;

export const IMAGE_CATALOGUE: Readonly<Record<string, ImageModelEntry>> = {
  "sd-turbo": {
    pipeline: "sd-turbo",
    repo: "schmuell/sd-turbo-ort-web",
    tokenizerRepo: "openai/clip-vit-large-patch14",
    size: 512,
    latentChannels: 4,
    vaeScale: 0.18215,
    sigma: 14.6146,
    timestep: 999,
    maxTokens: 77,
    padId: 0,
    sizeMb: 2514,
    licence: "Stability AI Non-Commercial Research Community License",
    notes: "Stability AI's SD-Turbo (SD 2.1 distilled to one step), float16 for WebGPU. Non-commercial use only.",
  },
  "z-image-turbo": {
    pipeline: "z-image",
    repo: "cretz/Z-Image-Turbo-ONNX-sharded",
    graphs: {
      textEncoder: "text_encoder_model_q4f16",
      transformer: [0, 1, 2, 3, 4, 5].map((i) => `transformer_model_q4f16_shard${i}`),
      scheduler: "scheduler_step_model_f16",
      vaePreProcess: "vae_pre_process_model_f16",
      vaeDecoder: "vae_decoder_model_f16",
    },
    tokenizerFolder: "tokenizer",
    maxTokens: 512,
    size: 512,
    latentChannels: 16,
    steps: 9,
    shift: 3,
    sideMultiple: 16,
    sizeMb: 6640,
    licence: "Apache-2.0",
    notes: "Tongyi-MAI's Z-Image-Turbo (6B S3-DiT, Qwen3 4B text encoder), 4-bit weights, the transformer in six shards for WebGPU. Needs about 7 GB of GPU memory.",
  },
};

export const DEFAULT_IMAGE_MODEL = "sd-turbo";
