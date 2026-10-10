# onnxweb-image

Image generation in the browser: a prompt to a picture, on WebGPU.

| Name | Model | Steps | Size | Licence |
| --- | --- | --- | --- | --- |
| `sd-turbo` (default) | Stability AI SD-Turbo, float16 | 1 | 2.5 GB | Non-commercial |
| `z-image-turbo` | Tongyi-MAI Z-Image-Turbo (6B diffusion transformer, Qwen3 4B text encoder), 4-bit | 9 | 6.6 GB | Apache-2.0 |

Z-Image's transformer is published in six shards, chained on the GPU without coming back to the CPU between them. Its text encoder can run on the GPU, on the CPU (to leave the GPU's memory to the transformer), or be loaded per prompt.

```ts
import * as ort from "onnxruntime-web";
import { ImageGenerator } from "@youni/onnxweb-image";

const generator = await ImageGenerator.load({ ort, Tensor: ort.Tensor, model: "z-image-turbo" });
const image = await generator.generate("a red fox sitting in the snow, photograph", { seed: 42, width: 768, height: 512 });
ctx.putImageData(new ImageData(image.data, image.width, image.height), 0, 0);
```

## Install

Builds against `@youni/onnxweb-core`:

```sh
npm install
npm run build   # dist/
npm test        # unit tests (Node, onnxruntime-web on WebAssembly)
```

## Licence

Apache-2.0 for the code. Model weights keep their own licence (listed above where it matters).
