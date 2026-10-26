"""Summarise a TextCraft run's transcripts: parse rate, tool mix, episode length.

The Part 5 diagnostic question is *why* success rate sits where it does. The
logged metrics say how often episodes succeed; the transcripts say what the
policy actually emits. This script answers the second question, e.g. the share
of turns whose completion fails to parse into a tool call (those turns do
nothing, so they are pure wasted context) and which tools the model reaches
for, including names that do not exist in this environment.

    uv run python scripts/analyze_textcraft_transcripts.py \
        runs/p5_textcraft_grpo_trajectories/step_000000
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

KNOWN_TOOLS = {"craft", "get_info", "view_inventory", "finish"}


def summarize(directory: Path) -> dict:
    episodes = sorted(directory.glob("*.jsonl"))
    rewards: list[float] = []
    steps_per_episode: list[int] = []
    tool_calls: collections.Counter[str] = collections.Counter()
    parsed = total_turns = 0

    for path in episodes:
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        header = records[0]
        rewards.append(float(header["reward"]))
        steps_per_episode.append(int(header["num_steps"]))
        for record in records:
            if record.get("type") != "step":
                continue
            total_turns += 1
            call = record.get("tool_call")
            if not call:
                tool_calls["(none)"] += 1
                continue
            parsed += 1
            tool_calls[call["name"]] += 1

    n = len(episodes)
    invented = {
        name: count
        for name, count in tool_calls.items()
        if name not in KNOWN_TOOLS and name != "(none)"
    }
    return {
        "episodes": n,
        "success_rate": sum(1 for r in rewards if r > 0) / n if n else 0.0,
        "mean_reward": sum(rewards) / n if n else 0.0,
        "mean_steps": sum(steps_per_episode) / n if n else 0.0,
        "max_steps_hit": sum(1 for s in steps_per_episode if s >= 75) / n if n else 0.0,
        "turns": total_turns,
        "parsed_turns": parsed,
        "parse_rate": parsed / total_turns if total_turns else 0.0,
        "tool_calls": dict(tool_calls.most_common()),
        "invented_tool_calls": invented,
    }


def main() -> None:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else
                     "runs/p5_textcraft_grpo_trajectories/step_000000")
    result = summarize(directory)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
