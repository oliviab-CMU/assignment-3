"""Probe: initialize the backend from a training client, then evaluate.

Standalone base-model evaluation (evaluate_checkpoint.py without
--model-path) never calls create_model, so SkyRLTrainBackend never starts
Ray or the vLLM engines and every sample fails with "no proxy URL
published". Creating a fresh LoRA training client does create the model,
so a fresh (untrained) adapter snapshot samples from the base weights.
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
    config = load_config("configs/p5_textcraft_grpo.yaml")
    model_cfg, train_cfg = config.model, config.train
    svc = make_service_client()

    print("creating fresh LoRA training client...", flush=True)
    tc = await svc.create_lora_training_client_async(
        base_model=model_cfg.base_model, rank=model_cfg.lora_rank
    )
    print("training client ready; snapshotting...", flush=True)
    fut = await tc.save_weights_for_sampler_async(name="probe_base")
    res = await fut.result_async()
    print("snapshot:", res.path, flush=True)

    tokenizer = tc.get_tokenizer()
    rendering = dict(
        max_tokens=model_cfg.max_tokens,
        temperature=model_cfg.temperature,
        chat_template=load_chat_template(model_cfg.chat_template),
        enable_thinking=model_cfg.enable_thinking,
        preserve_thinking_history=model_cfg.preserve_thinking_history,
        stop_sequences=model_cfg.stop_sequences,
    )
    policy = TinkerPolicy(svc.create_sampling_client(model_path=res.path), tokenizer, **rendering)

    out = {}
    for split, n in (("data/textcraft_val.jsonl", 4), ("data/textcraft_val_hard.jsonl", 2)):
        tasks = load_tasks(split)[:n]
        r = await evaluate_policy(
            policy, tasks, make_env_factory(train_cfg.env), k=1,
            concurrency=8, agent_cls=AGENT_REGISTRY[train_cfg.agent],
        )
        out[split] = r
        print(split, json.dumps(r), flush=True)
    print("RESULT", json.dumps(out))


asyncio.run(main())
