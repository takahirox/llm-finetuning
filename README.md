# llm-finetuning

This ProjectWeave-managed repository prepares local LLM experiments on Apple Silicon.
The planned experiment compares a 4B-class Qwen model before and after LoRA
fine-tuning on Python code-generation problems, using held-out evaluation such as
MBPP. Model selection, datasets, evaluation, and training will be added in separate
Issues; this bootstrap only installs packages and checks the environment.

## Supported environment

- Apple Silicon Mac with native **arm64** Python and a terminal running without Rosetta.
- **macOS 14 or newer**, following the [MLX installation requirements](https://ml-explore.github.io/mlx/build/html/install.html).
- **Python 3.12**; `.python-version` selects 3.12.13 for reproducible setup.
- **uv 0.12.23**, the version used to generate and validate `uv.lock`.
- **MLX 0.32.3** and **MLX-LM 0.32.0**, plus their locked runtime dependencies.
  MLX-LM's training/evaluation extras are not installed.

The lock is restricted to Apple Silicon macOS. Intel Macs, Linux, Windows, and
other Python minor versions are unsupported by this bootstrap. The exact chip
and memory needed for future experiments remain to be determined; the check does
not establish that a particular model or training run will fit in memory.

## Setup

Install the pinned [uv tool](https://docs.astral.sh/uv/getting-started/installation/)
if it is not already available, then add it to the current shell:

```sh
curl -LsSf https://astral.sh/uv/0.12.23/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

Clone the repository and install the locked dependencies into `.venv`:

```sh
git clone https://github.com/takahirox/llm-finetuning.git
cd llm-finetuning
uv python install
uv sync --locked
.venv/bin/python scripts/check_environment.py
```

`uv python install` reads `.python-version`. `uv sync --locked` reads
`pyproject.toml` and `uv.lock`, creates the isolated environment, and refuses to
silently update an out-of-date lock. These steps download Python (if needed) and
Python packages, but no model weights or datasets. No environment activation is
required; using `.venv/bin/python` selects the project interpreter explicitly.

## Environment check

From the repository root, run:

```sh
.venv/bin/python scripts/check_environment.py
```

The check reports Python, OS, architecture, virtual-environment status, package
versions/imports, and Metal availability. It exits with status `0` on success or
`1` with recovery instructions on an unsupported or incomplete setup. It only
imports packages and queries Metal availability; it does not load models,
evaluate tensors, or run training. Once packages are installed, it can run offline.

If `.venv` does not exist, run `uv sync --locked` first. For platform diagnostics
before installation, `python3 scripts/check_environment.py` can run using only
the standard library on Python 3.9+; it reports unsupported versions/platforms
and skips package imports until the prerequisites are met. If it reports
`x86_64` on Apple Silicon, disable Rosetta for the terminal and recreate `.venv`
using native arm64 Python. To recreate it, move the old `.venv` outside the
repository, then run `uv sync --locked` again.

Keep model weights, dataset caches, checkpoints, and generated results in the
ignored `models/`, `data/`, `checkpoints/`, and `outputs/` directories (create them
only when needed). Local caches and secrets such as `.cache/` and `.env` are also
ignored. Keep useful source and configuration files outside artifact directories.

The hardware-independent diagnostic tests use the standard library:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Collaboration guidance

- [Development flow and English collaboration convention](docs/development-flow.md)
- [AI review and merge guidelines](docs/review-guidelines.md)
- [Issue template](.github/ISSUE_TEMPLATE/issue.md)
- [Pull request template](.github/pull_request_template.md)
- [AI contributor entry point](AGENTS.md)
