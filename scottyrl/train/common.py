"""Group rollouts and batch assembly. *** STUDENT IMPLEMENTATION FILE ***

These two functions are shared by the sync (Parts 1-2) and async (Part 3)
trainers.
"""

from __future__ import annotations

import asyncio  # noqa: F401  (student dependency)
from typing import Callable

from scottyrl.advantages import compute_group_advantages  # noqa: F401  (student dependency)
from scottyrl.agents.base import Agent
from scottyrl.agents.xml_tool_call import XMLToolCallAgent
from scottyrl.datagen import trajectory_to_training_sequences  # noqa: F401  (student dependency)
from scottyrl.envs.base import Env
from scottyrl.policy import Policy
from scottyrl.rollout import run_episode  # noqa: F401  (student dependency)
from scottyrl.types import Task, TokenSequence, Trajectory


async def collect_group(
    policy: Policy,
    task: Task,
    env_factory: Callable[[Task], Env],
    group_size: int,
    agent_cls: type[Agent] = XMLToolCallAgent,
) -> list[Trajectory]:
    """Run `group_size` independent rollouts of `task` concurrently.

    Contract: every rollout gets its own fresh env instance and fresh agent
    instance (no shared state between rollouts), and the rollouts run
    concurrently so their LLM calls overlap in time. Each rollout is
    `run_episode(agent_cls(policy), env_factory(task))` (see
    scottyrl.rollout); launch all `group_size` of them with `asyncio.gather`.

    Returns: `group_size` trajectories, in rollout order.
    """
    return await asyncio.gather(
        *(
            run_episode(agent_cls(policy), env_factory(task))
            for _ in range(group_size)
        )
    )


def groups_to_batch(
    groups: list[list[Trajectory]],
    method: str = "grpo",
) -> tuple[list[TokenSequence], dict]:
    """Turn a list of trajectory groups into a flat training batch.

    Contract:
      - Each group's advantages come from your `compute_group_advantages`,
        applied to that group's episode rewards (in rollout order).
      - A degenerate group (advantages None) contributes no sequences and is
        counted, but its rewards still count toward mean_reward.
      - Every remaining trajectory contributes its training sequences (your
        `trajectory_to_training_sequences`) with its own advantage, even
        when that advantage is 0.0.

    Returns: (sequences, stats), where stats has at least these keys:
        "num_groups"              total groups passed in
        "num_degenerate_groups"   groups skipped as degenerate
        "frac_degenerate_groups"  ratio of the two above (0.0 if no groups)
        "mean_reward"             mean episode reward over ALL trajectories,
                                  degenerate groups included (0.0 if there
                                  are none)
        "num_sequences"           len(sequences)
        "num_action_tokens"       total mask==1 positions in the batch
    """
    sequences: list[TokenSequence] = []
    num_groups = len(groups)
    num_degenerate = 0
    all_rewards: list[float] = []

    for group in groups:
        rewards = [traj.reward for traj in group]
        all_rewards.extend(rewards)
        advantages = compute_group_advantages(rewards, method=method)
        if advantages is None:
            # Degenerate group: no learning signal, but its rewards still
            # count toward mean_reward.
            num_degenerate += 1
            continue
        for traj, advantage in zip(group, advantages):
            # Every trajectory contributes its sequences, even when its
            # advantage is 0.0 (all-zero advantages within a group).
            sequences.extend(
                trajectory_to_training_sequences(traj, advantage)
            )

    return sequences, {
        "num_groups": num_groups,
        "num_degenerate_groups": num_degenerate,
        "frac_degenerate_groups": (num_degenerate / num_groups) if num_groups else 0.0,
        "mean_reward": (sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0,
        "num_sequences": len(sequences),
        "num_action_tokens": sum(seq.num_action_tokens for seq in sequences),
    }
