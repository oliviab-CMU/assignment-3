"""Watch a live training run and shout if it goes wrong.

The Part 5 run is a hero run: seven hours of four H100s. The assignment says to
watch the metrics so a broken run is stopped before it burns budget. Sitting in
a sleep loop to do that wastes attention; this script polls the run log and
prints an ALERT line when a health condition trips, so the check can run in the
background and only speak up when there is something to say.

    uv run python scripts/watch_run.py p5_textcraft_grpo --hours 9

Exit status: 0 when the run reached its final step or the watch window ended,
1 if an alert fired (the caller decides whether to stop the run).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_steps(path: Path) -> dict[int, dict]:
    steps: dict[int, dict] = {}
    if not path.exists():
        return steps
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and isinstance(record.get("step"), int):
                steps.setdefault(record["step"], {}).update(record)
    return steps


def check(step: int, record: dict, history: list[dict]) -> list[str]:
    """Health conditions worth interrupting a human for.

    Thresholds mirror scottyrl.metrics.sanity_warnings, so an alert here means
    the trainer would have printed (or nearly printed) a [sanity] line. The
    only additions are step duration and a sustained collapse to zero reward,
    which the framework does not check but which would burn a hero-run budget.
    """
    alerts: list[str] = []
    degenerate = record.get("train/frac_degenerate_groups")
    if degenerate is not None and degenerate > 0.8:
        alerts.append(f"step {step}: {degenerate:.0%} of groups degenerate "
                      "(no learning signal; check the transcripts)")

    grad = record.get("train/grad_norm")
    if grad is not None and grad > 50.0:
        alerts.append(f"step {step}: grad_norm {grad:.3f} is a gradient spike")

    iw_mean = record.get("train/importance_weight_mean")
    if iw_mean is not None and not 0.5 <= iw_mean <= 2.0:
        alerts.append(f"step {step}: importance_weight_mean {iw_mean:.3f} "
                      "outside [0.5, 2.0] (policies have diverged)")

    iw_max = record.get("train/importance_weight_max")
    iw_std = record.get("train/importance_weight_std")
    if iw_max is not None and (iw_max > 10.0 or not math.isfinite(iw_max)):
        alerts.append(f"step {step}: importance_weight_max {iw_max:.3g}")
    if iw_std is not None and (iw_std > 1.0 or not math.isfinite(iw_std)):
        alerts.append(f"step {step}: importance_weight_std {iw_std:.3g}")
    # The per-token minimum is deliberately not checked: one token out of
    # several hundred thousand can sit far below 1 while the mean and std stay
    # put, and such a token is clipped by the PPO surrogate anyway.

    entropy = record.get("train/entropy_proxy")
    if entropy is not None and entropy < 0.005:
        alerts.append(f"step {step}: entropy_proxy {entropy:.4f} is very low")

    # A step that takes more than an hour means the deployment is unhealthy;
    # at that pace the remaining steps blow the budget.
    step_time = record.get("train/step_time_s")
    if step_time is not None and step_time > 3600:
        alerts.append(f"step {step}: step took {step_time / 60:.0f} min")

    # Reward falling to ~0 for several consecutive steps is a collapse, not noise.
    recent = [h.get("train/success_rate") for h in history[-4:]]
    recent = [r for r in recent if r is not None]
    if len(recent) >= 4 and max(recent) < 0.02:
        alerts.append(f"step {step}: train success rate has been ~0 for "
                      f"{len(recent)} steps")

    return alerts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_name")
    p.add_argument("--hours", type=float, default=9.0, help="watch window")
    p.add_argument("--final-step", type=int, default=50)
    p.add_argument("--poll", type=float, default=60.0, help="seconds between checks")
    args = p.parse_args()

    log = REPO_ROOT / "runs" / f"{args.run_name}.jsonl"
    deadline = time.time() + args.hours * 3600
    seen: set[int] = set()
    history: list[dict] = []

    while time.time() < deadline:
        steps = read_steps(log)
        for step in sorted(steps):
            if step in seen:
                continue
            seen.add(step)
            record = steps[step]
            history.append(record)
            if "train/step_time_s" not in record:
                print(f"[watch] step {step}: eval record", flush=True)
                continue
            print(
                f"[watch] step {step}: reward={record.get('train/success_rate')} "
                f"degen={record.get('train/frac_degenerate_groups')} "
                f"step_s={record.get('train/step_time_s', 0):.0f} "
                f"wait_frac={record.get('train/learner_wait_frac')}",
                flush=True,
            )
            alerts = check(step, record, history)
            for alert in alerts:
                print(f"ALERT {alert}", flush=True)
            if alerts:
                return 1

        if seen and max(seen) >= args.final_step:
            print(f"[watch] reached step {max(seen)}; done", flush=True)
            return 0

        time.sleep(args.poll)

    print(f"[watch] window ended at step {max(seen) if seen else 'none'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
