# AI Usage

## Tools used

**Claude Code (Anthropic), running Claude Sonnet and Claude Haiku family models.**
This was the only AI tool used for this assignment. It is an agentic coding
assistant that can read files, run shell commands, and edit code in this
repository, driven by natural-language instructions from me.

## How I used it

- **Implementation (Parts 1–3).** I wrote the assignment's specifications into
  the conversation and had Claude Code implement `scottyrl/advantages.py`,
  `scottyrl/datagen.py`, `scottyrl/tinker_adapter.py`, `scottyrl/train/common.py`,
  `scottyrl/train/sync.py`, and `scottyrl/train/async_grpo.py`. For the
  asynchronous trainer the assistant proposed the producer/consumer structure
  (bounded queue, staleness accounting, background evaluation) and I directed
  the design decisions.
- **Infrastructure.** Claude Code wrote the Modal deployment wrappers under
  `skyrl_tinker_modal/` and the helper scripts under `scripts/` (probes,
  `evaluate_checkpoint.py` wrappers, the plotting script, the PDF build script).
- **Debugging.** When runs failed — image build errors, the Tinker wire-format
  mismatch, and a base-model evaluation that 500s on the training backend — the
  assistant traced the failure from logs and source rather than guessing. In
  each case I ran the diagnostic commands and made the call on the fix before it
  was applied. The base-model failure was resolved by pulling the upstream
  evaluation patch, which replaces the workaround we had been using with a
  standalone vLLM server that serves the base weights and the adapter together.
- **Experiments.** Claude Code launched and monitored the training runs, polled
  `runs/*.jsonl` for health metrics, and stopped or redirected work when a run
  looked wrong. I authorized every GPU-spending run first.
- **Plots.** The diagnostic figures in `report_plots/` are produced by
  `scripts/plot_report.py`, which Claude Code wrote. The visual design follows a
  data-visualisation skill available to the assistant (fixed categorical colour
  order, single y-axis per panel, markers for eval points).
- **Report.** `REPORT.md` (and the `report.pdf` built from it) was drafted by
  Claude Code from the run logs and then reviewed and edited by me. Every
  number in the report was checked against the corresponding record in
  `runs/*.jsonl`.

## What I did not delegate

I decided the experimental design (which comparisons to run, what to hold fixed,
how to read the results), authorised and monitored every Modal deployment, and
reviewed all generated code.
