# Third-party attribution

bit-jev is a separate implementation inspired by [Kev](https://github.com/jaredpalmer/kev)'s structured decision interface and pointer-head architecture. Kev is Copyright 2026 Jared Palmer and licensed under Apache-2.0. The project source license is in [`LICENSE`](LICENSE); this attribution does not claim that Jared Palmer authored bit-jev.

The backbone and native inference components are based on [Microsoft BitNet](https://github.com/microsoft/BitNet) and the [BitNet b1.58 2B model](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16). Microsoft distributes the BitNet source and base model under the MIT License. The GitHub source repository does not contain trained bit-jev weights. A separate [Hugging Face model repository](https://huggingface.co/jinghao1632/bit-jev-2b-distilled) publishes an I2_S derivative checkpoint and sanitized AutoDL measurements; its card identifies the Yelp data provenance and current rights status.

The native build uses a pinned BitNet source revision with the local patches recorded in this repository. The upstream copyright notices and license terms continue to apply to those upstream components.

## Microsoft BitNet base-model license notice

The base-model family used for local bit-jev research is `microsoft/bitnet-b1.58-2B-4T-bf16`. Its publisher supplies the following MIT license notice in the [model repository](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16/blob/main/LICENSE):

> MIT License
>
> Copyright (c) Microsoft Corporation.
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

Training and evaluation datasets have their own licenses. The GitHub source tree contains no trained checkpoint, but the separate [Hugging Face repository](https://huggingface.co/jinghao1632/bit-jev-2b-distilled) publishes an I2_S derivative and its model card. The GitHub tree also contains a sanitized aggregate AutoDL timing record; it does not contain predictions or raw review text.

## Dataset provenance and publication boundary

The local research evaluated some material originating from [Banking77](https://huggingface.co/datasets/PolyAI/banking77), published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). No raw Banking77 records or checkpoint predictions are included in the public release. The sanitized AutoDL report records aggregate timing and memory only.

The published checkpoint was trained on a multi-source decision set that included Yelp review records. Yelp's [dataset card](https://huggingface.co/datasets/Yelp/yelp_review_full) links to separate [dataset terms](https://s3-media3.fl.yelpcdn.com/assets/srv0/engineering_pages/bea5c1e92bf3/assets/vendor/yelp-dataset-agreement.pdf). A request asking about derivative weight and result publication was sent; as of 2026-09-28, no written response had been received. The checkpoint card discloses this status and does not assign an open-weights license. No raw Yelp review record is included in the model repository. Source code availability does not grant rights to any third-party dataset.
