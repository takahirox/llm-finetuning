# Third-party notices

The files in `scripts/freight_reference/` are copied from or adapted from
[openatlaspro-AI/mlx-lora-finetune](https://github.com/openatlaspro-AI/mlx-lora-finetune/tree/24f663d1a276203356d957d6e340ee2182ff83d8),
commit `24f663d1a276203356d957d6e340ee2182ff83d8`. The generator, prompt, and
schema are unchanged. `scoring.py` contains the unchanged pure scorer with
unused imports, the original runner, and trailing blank lines removed.

MIT License

Copyright (c) 2026 Mark Teji

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## SQL experiment provenance

The SQL workflow is independently implemented after reviewing
[W-Kaski/qwen-qlora-sql-benchmark at b90759ab610ae29a0e8744098eda6ea0d559df36](https://github.com/W-Kaski/qwen-qlora-sql-benchmark/tree/b90759ab610ae29a0e8744098eda6ea0d559df36).
That revision has no explicit source-code license; no reference code or fixtures
are vendored. Its scores are attributed third-party reports.

Downloaded and derived SQL records credit **b-mc2 (2023), sql-create-context**,
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), based on WikiSQL
(Zhong, Xiong and Socher, *Seq2SQL*, 2017) and Spider (Yu et al., *Spider*, 2018).
Our changes are deterministic subset selection and chat-format conversion.
[Dataset source, attribution, immutable revision and checksum](docs/sql-control.md#dataset-attribution-and-split-integrity)
are recorded in the experiment documentation and local data manifests. Source
records and derived splits are not committed. The Qwen model is Apache-2.0;
model download retains its LICENSE. These licenses are separate from the
freight reference's MIT notice above.
