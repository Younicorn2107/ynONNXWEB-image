import { describe, expect, it } from "vitest";

import { IMAGE_CATALOGUE, gaussian, promptIds, sdSigmas, sdTimesteps, toRgba, zImagePrompt, zImageTimesteps, type SdTurboEntry } from "../../src/index.js";

describe("the pieces of the pipeline", () => {
  const entry = IMAGE_CATALOGUE["sd-turbo"] as SdTurboEntry;

  it("pads a prompt to 77 ids: start, the words, end, then '!'", () => {
    const tokenizer = { encode: () => [49406, 320, 1929, 49407] };
    const ids = promptIds(tokenizer as never, "A cat", entry);
    expect(ids.length).toBe(77);
    expect(Array.from(ids.slice(0, 5))).toEqual([49406, 320, 1929, 49407, 0]);
    expect(ids[76]).toBe(0);
  });

  it("cuts a long prompt and keeps its end token", () => {
    const tokenizer = { encode: () => Array.from({ length: 200 }, (_, i) => 1000 + i) };
    const ids = promptIds(tokenizer as never, "long", entry);
    expect(ids[0]).toBe(49406);
    expect(ids[75]).toBe(1074);
    expect(ids[76]).toBe(49407);
  });

  it("draws the same noise for the same seed, standard normal", () => {
    const a = gaussian(4 * 64 * 64, 7);
    expect(Array.from(gaussian(8, 7))).toEqual(Array.from(a.slice(0, 8)));
    expect(Array.from(gaussian(8, 8))).not.toEqual(Array.from(a.slice(0, 8)));
    const mean = a.reduce((s, v) => s + v, 0) / a.length;
    const variance = a.reduce((s, v) => s + (v - mean) ** 2, 0) / a.length;
    expect(Math.abs(mean)).toBeLessThan(0.05);
    expect(Math.abs(variance - 1)).toBeLessThan(0.05);
  });

  it("turns RGB planes in [-1, 1] into RGBA bytes", () => {
    const planes = Float32Array.of(-1, 1, 0, 0, 1, -1);
    expect(Array.from(toRgba(planes, 2, 1))).toEqual([0, 128, 255, 255, 255, 128, 0, 255]);
  });
});

describe("Z-Image", () => {
  it("wraps the prompt in Qwen3's chat template, thinking on", () => {
    expect(zImagePrompt("a red fox")).toBe("<|im_start|>user\na red fox<|im_end|>\n<|im_start|>assistant\n");
  });

  it("gives the flow-matching schedule of the reference pipeline", () => {
    const nine = [0, 0.04530615359544754, 0.09964098781347275, 0.16600202023983002, 0.24887891113758087, 0.3553137183189392, 0.49701493978500366, 0.6949910521507263, 1];
    const four = [0, 0.14230768382549286, 0.397849440574646, 1];
    Array.from(zImageTimesteps(9, 3)).forEach((v, i) => expect(v).toBeCloseTo(nine[i]!, 6));
    Array.from(zImageTimesteps(4, 3)).forEach((v, i) => expect(v).toBeCloseTo(four[i]!, 6));
  });

  it("is in the catalogue with its six shards", () => {
    const entry = IMAGE_CATALOGUE["z-image-turbo"]!;
    expect(entry.pipeline).toBe("z-image");
    expect(entry.pipeline === "z-image" && entry.graphs.transformer.length).toBe(6);
  });
});

describe("SD-Turbo schedule", () => {
  const entry = IMAGE_CATALOGUE["sd-turbo"] as SdTurboEntry;

  it("spreads the steps trailing from 999", () => {
    expect(sdTimesteps(1, 1000)).toEqual([999]);
    expect(sdTimesteps(4, 1000)).toEqual([999, 749, 499, 249]);
  });

  it("gives the training noise levels (sigma 14.6146 at 999)", () => {
    expect(sdSigmas(entry)[999]).toBeCloseTo(14.6146, 3);
  });
});
