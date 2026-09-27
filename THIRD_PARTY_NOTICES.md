# Third-party attribution

bit-jev is a separate implementation inspired by [Kev](https://github.com/jaredpalmer/kev)'s structured decision interface and pointer-head architecture. Kev is Copyright 2026 Jared Palmer and licensed under Apache-2.0. The project source license is in [`LICENSE`](LICENSE); this attribution does not claim that Jared Palmer authored bit-jev.

The backbone and native inference components are based on [Microsoft BitNet](https://github.com/microsoft/BitNet) and the [BitNet b1.58 2B model](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16). Microsoft distributes the BitNet source and base model under the MIT License. No trained bit-jev derivative weights are distributed in this source-only release.

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

Training and evaluation datasets have their own licenses. This source-only repository provides no trained checkpoint or checkpoint-derived benchmark report.

## Dataset provenance and publication boundary

The local research evaluated some material originating from [Banking77](https://huggingface.co/datasets/PolyAI/banking77), published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Dataset authors and terms should be credited if permitted inputs or outputs are shared in a future release. Current checkpoint-derived examples and reports are excluded from this source-only release.

The withheld local checkpoint was trained on a multi-source decision set that included Yelp review records. Yelp's [dataset card](https://huggingface.co/datasets/Yelp/yelp_review_full) links to separate [dataset terms](https://s3-media3.fl.yelpcdn.com/assets/srv0/engineering_pages/bea5c1e92bf3/assets/vendor/yelp-dataset-agreement.pdf). Their applicability to derivative weights and result publication has not been resolved, so neither the checkpoint nor its derived measurements are part of this release. Source code availability does not grant rights to any third-party dataset.
