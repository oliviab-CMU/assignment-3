"""Probe: estimate the base model's TextCraft success rate on N tasks.

Standalone base-model evaluation never initializes the backend (see
probe_base_via_training_client.py), so create a fresh LoRA training client
first; its untrained adapter samples the base weights.
"""
import asyncio, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scottyrl.agents import AGENT_REGISTRY
from scottyrl.config import load_config
from scottyrl.envs import make_env_factory
from scottyrl.evaluate import evaluate_policy
from scottyrl.policy import TinkerPolicy, load_chat_template
from scottyrl.tinker_adapter import make_service_client
from scottyrl.types import load_tasks


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    split = sys.argv[2] if len(sys.argv) > 2 else "data/textcraft_val.jsonl"
    config = load_config("configs/p5_textcraft_grpo.yaml")
    model_cfg, train_cfg = config.model, config.train
    svc = make_service_client()
    tc = await svc.create_lora_training_client_async(
        base_model=model_cfg.base_model, rank=model_cfg.lora_rank
    )
    res = await (await tc.save_weights_for_sampler_async(name="probe_scale")).result_async()
    policy = TinkerPolicy(
        svc.create_sampling_client(model_path=res.path), tc.get_tokenizer(),
        max_tokens=model_cfg.max_tokens, temperature=model_cfg.temperature,
        chat_template=load_chat_template(model_cfg.chat_template),
        enable_thinking=model_cfg.enable_thinking,
        preserve_thinking_history=model_cfg.preserve_thinking_history,
        stop_sequences=model_cfg.stop_sequences,
    )
    tasks = load_tasks(split)[:n]
    r = await evaluate_policy(
        policy, tasks, make_env_factory(train_cfg.env), k=1,
        concurrency=32, agent_cls=AGENT_REGISTRY[train_cfg.agent],
    )
    print("RESULT", split, n, json.dumps(r))


asyncio.run(main())
