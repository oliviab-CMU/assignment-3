"""Probe: run one TextCraft episode and print every turn.

Checks that the base model emits parseable tool calls at all -- if every
completion fails to parse, every group is degenerate and GRPO has no signal.
"""
import asyncio, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scottyrl.agents import AGENT_REGISTRY
from scottyrl.config import load_config
from scottyrl.envs import make_env_factory
from scottyrl.policy import TinkerPolicy, load_chat_template
from scottyrl.rollout import run_episode
from scottyrl.tinker_adapter import make_service_client
from scottyrl.types import load_tasks


async def main():
    config = load_config("configs/p5_textcraft_grpo.yaml")
    model_cfg, train_cfg = config.model, config.train
    svc = make_service_client()
    tc = await svc.create_lora_training_client_async(
        base_model=model_cfg.base_model, rank=model_cfg.lora_rank
    )
    res = await (await tc.save_weights_for_sampler_async(name="probe_turns")).result_async()
    policy = TinkerPolicy(
        svc.create_sampling_client(model_path=res.path), tc.get_tokenizer(),
        max_tokens=model_cfg.max_tokens, temperature=model_cfg.temperature,
        chat_template=load_chat_template(model_cfg.chat_template),
        enable_thinking=model_cfg.enable_thinking,
        preserve_thinking_history=model_cfg.preserve_thinking_history,
        stop_sequences=model_cfg.stop_sequences,
    )
    tasks = load_tasks("data/textcraft_val.jsonl")[:1]
    env = make_env_factory(train_cfg.env)(tasks[0])
    from scottyrl.rollout import PENDING_TRAJECTORIES
    traj = await run_episode(AGENT_REGISTRY[train_cfg.agent](policy), env)
    print("reward", traj.reward, "steps", traj.num_steps)
    turns = PENDING_TRAJECTORIES[-1][-1]
    for i, t in enumerate(turns[:12]):
        print(f"--- turn {i} tokens={t.get('completion_tokens')} "
              f"call={t.get('tool_call') and t['tool_call']['name']}")
        print("completion:", repr((t.get("agent") or "")[:600]))
        print("env:", repr((t.get("env") or "")[:200]))
    parsed = sum(1 for t in turns if t.get("tool_call"))
    print(f"PARSED {parsed}/{len(turns)} turns")


asyncio.run(main())
