"""Standard diagnostic plots for the assignment report.

Reads runs/<run_name>.jsonl and writes figures to report_plots/. Each run gets
one multi-panel figure covering the four required diagnostic families:

  reward     train/success_rate, eval/success_rate, and every logged
             train/pass@* and eval/pass@* curve (labelled by exact key)
  optim      train/grad_norm
  drift      train/importance_weight_{mean,min,max,std}
  health     train/frac_degenerate_groups and train/entropy_proxy

Health is split into two panels rather than sharing a second y-axis: two
scales on one plot invent a correlation the data does not contain.

Usage:
    uv run python scripts/plot_report.py p2a_number_search_reinforce \
        p2b_number_search_grpo
    uv run python scripts/plot_report.py --compare p2a_number_search_reinforce \
        p2b_number_search_grpo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots, in fixed order (light surface). Identity follows the
# entity, never its rank in a sorted list.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

SERIES_STYLE = [
    (BLUE, "-", "o"),
    (ORANGE, "-", "s"),
    (AQUA, "--", "^"),
    (YELLOW, "--", "D"),
]


def load_run(path: str) -> dict[int, dict]:
    """One merged record per step: train and eval keys land together."""
    steps: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            steps.setdefault(r["step"], {}).update(r)
    return steps


def series(steps: dict[int, dict], key: str) -> tuple[list, list]:
    xs = [s for s in sorted(steps) if key in steps[s]]
    return xs, [steps[s][key] for s in xs]


def series_with_gaps(steps: dict[int, dict], key: str) -> tuple[list, list]:
    """Series for a per-step metric, with NaN where a step has no record.

    A train metric is expected on every training step, so a missing step is
    real information (the batch was skipped, or the update did not run) and
    the line must break there rather than bridge across it. Eval metrics are
    sampled only every `eval_every` steps and are plotted as sparse points.
    """
    if not key.startswith("train/"):
        return series(steps, key)
    xs = sorted(steps)
    return xs, [steps[s].get(key, float("nan")) for s in xs]


def all_keys(steps: dict[int, dict]) -> set[str]:
    keys: set[str] = set()
    for record in steps.values():
        keys.update(record)
    return keys


def _style(
    ax,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    # One constant pad for every panel: titles in the same row then share a
    # baseline, and the band is tall enough for a two-row legend. Panels with a
    # single series get no legend, so their band stays empty; that is the price
    # of aligning every title on one baseline.
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=50)
    ax.set_xlabel("training step", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _legend(ax, ncol: int = 2) -> None:
    """Legend in its own band above the axes, never over data or the title.

    Two columns by default so the band is at most two rows tall; the title
    is padded above the whole band. Placing the legend outside the axes also
    keeps it from hiding points at the start of a curve, which is where the
    interesting baseline values sit. Pass ncol=1 for a longer, single-row
    legend; `_style` reserves a matching title pad.
    """
    ax.legend(
        fontsize=7, frameon=False, labelcolor=INK_SECONDARY,
        loc="lower left", bbox_to_anchor=(0, 1.005), ncol=ncol,
        borderaxespad=0, handlelength=2.2, columnspacing=1.4,
    )


def _draw(
    ax,
    steps: dict[int, dict],
    keys: list[str],
    style_offset: int = 0,
    labels: dict[str, str] | None = None,
) -> None:
    for i, key in enumerate(keys):
        xs, ys = series_with_gaps(steps, key)
        if not xs:
            continue
        color, ls, marker = SERIES_STYLE[(i + style_offset) % len(SERIES_STYLE)]
        # Several of these curves sit flat at 1.0, where they coincide exactly.
        # Eval series are drawn last, so a filled eval marker would hide the
        # train marker under it. Instead train markers are small and filled and
        # eval markers are larger and hollow: at a coincidence the train dot
        # stays visible inside the eval ring. Train also gets the higher
        # zorder so it wins any tie rather than being painted over.
        filled = not key.startswith("eval/")
        # Markers show where records actually exist; NaN gaps break the line
        # instead of bridging across skipped steps.
        ax.plot(
            xs, ys, color=color, linestyle=ls, linewidth=1.4,
            marker=marker, markersize=4.5 if filled else 6.0,
            markerfacecolor=color if filled else "none",
            markeredgewidth=1.0,
            zorder=3 if filled else 2,
            label=(labels or {}).get(key, key),
        )


def reward_keys(steps: dict[int, dict]) -> list[str]:
    """Exact pass@ keys, so training and evaluation k values stay distinct.

    A series identical to an earlier one is dropped: with G=1, train success
    rate *is* train/pass@1, so plotting both would hide one line under the
    other and add a redundant legend entry.
    """
    keys = [
        "train/success_rate",
        "eval/success_rate",
        *sorted(k for k in all_keys(steps) if k.startswith("train/pass@")),
        *sorted(k for k in all_keys(steps) if k.startswith("eval/pass@")),
    ]
    kept: list[str] = []
    for key in keys:
        _, ys = series(steps, key)
        if not ys:
            continue
        if any(ys == series(steps, other)[1] for other in kept):
            continue
        kept.append(key)
    return kept


def plot_run(run_name: str, out_dir: Path = Path("report_plots")) -> None:
    steps = load_run(f"runs/{run_name}.jsonl")
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(
        f"{run_name} — {len(steps)} logged steps ({min(steps)}–{max(steps)})",
        color=INK, fontsize=12, x=0.01, ha="left",
    )

    # ---- reward quality ---------------------------------------------------
    ax = axes[0, 0]
    _draw(ax, steps, reward_keys(steps))
    _style(ax, "reward quality", "success / pass@k", (-0.02, 1.05))
    _legend(ax)

    # ---- optimization -----------------------------------------------------
    ax = axes[0, 1]
    _draw(ax, steps, ["train/grad_norm"])
    _style(ax, "optimization", "train/grad_norm")

    # ---- policy drift -----------------------------------------------------
    # All four moments the report compares. The mean alone hides off-policy
    # drift in the tails, which min/max/std make visible. The log scale keeps
    # both the values near 1 and a heavy tail legible on one axis.
    ax = axes[1, 0]
    # Reference first and below the data, so the mean line stays visible on
    # top of it instead of being hidden by it.
    ax.axhline(1.0, color=INK_SECONDARY, linewidth=0.8, linestyle=":",
               label="1.0 (on-policy)", zorder=0)
    _draw(ax, steps, [
        "train/importance_weight_mean",
        "train/importance_weight_min",
        "train/importance_weight_max",
        "train/importance_weight_std",
    ], labels={
        "train/importance_weight_mean": "mean",
        "train/importance_weight_min": "min",
        "train/importance_weight_max": "max",
        "train/importance_weight_std": "std",
    })
    _style(ax, "policy drift", "importance weight (log scale)")
    ax.set_yscale("log")
    # Five entries (four series plus the reference) fit one row in a wider
    # figure, which keeps the band clear of the title and of the panel below.
    _legend(ax, ncol=5)

    # ---- data health: degenerate groups -----------------------------------
    ax = axes[1, 1]
    _draw(ax, steps, ["train/frac_degenerate_groups"])
    # A run that never sees a degenerate group sits at 0.0, exactly on the
    # lower spine; pad below so the line and its markers stay visible.
    _style(ax, "data health: degenerate groups", "fraction of groups", (-0.05, 1.05))

    # ---- data health: entropy (own panel, never a second y-axis) ----------
    ax = axes[2, 0]
    _draw(ax, steps, ["train/entropy_proxy"])
    _style(ax, "data health: entropy proxy", "nats / token")

    # ---- timing -----------------------------------------------------------
    # Short labels: the full keys are too wide for a two-column legend here and
    # the last entry used to run off the canvas.
    ax = axes[2, 1]
    _draw(ax, steps, ["train/step_time_s", "train/update_time_s", "train/rollout_time_s"],
          labels={
              "train/step_time_s": "step (total)",
              "train/update_time_s": "update",
              "train/rollout_time_s": "rollout",
          })
    _style(ax, "timing", "seconds")
    _legend(ax, ncol=1)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out_dir / f"{run_name}_diagnostics.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"plotted {path}")


def plot_compare(run_names: list[str], out_dir: Path = Path("report_plots")) -> None:
    """Same quantity, same axis, one curve per run — no dual scale."""
    runs = {name: load_run(f"runs/{name}.jsonl") for name in run_names}
    panels = [
        ("train/success_rate", "train success rate"),
        ("eval/success_rate", "eval success rate"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.5))
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, title) in zip(axes, panels):
        for i, (name, steps) in enumerate(runs.items()):
            xs, ys = series_with_gaps(steps, key)
            color, _, marker = SERIES_STYLE[i % len(SERIES_STYLE)]
            ax.plot(xs, ys, color=color, linewidth=1.4, marker=marker,
                    markersize=3.2, label=name)
        _style(ax, title, key, (-0.02, 1.05))
        _legend(ax)
    fig.suptitle("REINFORCE vs GRPO on Number Search", color=INK, fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    path = out_dir / "compare_reinforce_vs_grpo.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"plotted {path}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--compare":
        plot_compare(argv[1:])
    else:
        for name in argv:
            plot_run(name)
