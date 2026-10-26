"""Run the four Part 5 evaluations and save their results as one JSON file.

Part 5 asks for the fine-tuned adapter and the base model on both the medium
and the hard validation split, four rollouts per task. That is four invocations
of scripts/evaluate_checkpoint.py against the standalone vLLM server
(skyrl_tinker_modal/serve_evaluation.py), which serves both models on one
endpoint: the adapter as `a3` and the base weights as `Qwen/Qwen3.5-4B`. This
script runs them and stores each result under a stable key so the report table
can be built from one file.

    uv run python scripts/eval_p5_all.py --server <url>

Each row is written to the results file as soon as it finishes, so a killed run
keeps the rows it already paid for.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = "configs/p5_textcraft_grpo.yaml"
OUTPUT = REPO_ROOT / "runs" / "p5_eval_results.json"
ADAPTER_MODEL = "a3"
BASE_MODEL = "Qwen/Qwen3.5-4B"

SPLITS = {
    "medium": None,  # evaluate_checkpoint.py defaults to the env's val split
    "hard": "data/textcraft_val_hard.jsonl",
}


def run_eval(server: str, model: str, tasks: str | None, k: int) -> dict:
    cmd = [sys.executable, "scripts/evaluate_checkpoint.py", "--config", CONFIG,
           "--server", server, "--model", model, "--k", str(k)]
    if tasks:
        cmd += ["--tasks", tasks]
    label = f"{'finetuned' if model == ADAPTER_MODEL else 'base'}/"
    print(f"[eval] {label}{tasks or 'medium'} ...", flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], file=sys.stderr)
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"evaluation failed: {label}{tasks or 'medium'}")
    # The script prints its JSON last, after any logging above it.
    start = proc.stdout.index("{")
    return json.loads(proc.stdout[start:])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--server", required=True, help="URL printed by modal deploy")
    p.add_argument("--k", type=int, default=4, help="rollouts per task")
    args = p.parse_args()

    results: dict[str, dict] = {}
    if OUTPUT.exists():
        # Re-running the script resumes: rows already on disk are kept, so a
        # failure in one row never costs the ones that finished before it.
        results = json.loads(OUTPUT.read_text())
        print(f"[eval] resuming, have: {', '.join(results)}", flush=True)

    failures: list[str] = []
    for model_name, model in (("finetuned", ADAPTER_MODEL), ("base", BASE_MODEL)):
        for split, tasks in SPLITS.items():
            key = f"{model_name}/{split}"
            if key in results:
                print(f"[eval] {key} already done, skipping", flush=True)
                continue
            try:
                results[key] = run_eval(args.server, model, tasks, args.k)
            except SystemExit as exc:
                # One row failing must not abort the rest of the evaluation.
                failures.append(f"{key}: {exc}")
                print(f"[eval] {key} FAILED: {exc}", file=sys.stderr, flush=True)
                continue
            # Saved per row: a later failure must not discard finished work.
            OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
            print(f"[eval] {key} done", flush=True)

    if failures:
        print("\n[eval] failed rows:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        raise SystemExit(1)

    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"{'row':<18}{'success_rate':>14}{'pass@4':>10}{'avg_steps':>11}")
    for key, r in results.items():
        print(f"{key:<18}{r['success_rate']:>14.4f}{r['pass@4']:>10.4f}"
              f"{r['avg_steps']:>11.2f}")


if __name__ == "__main__":
    main()
