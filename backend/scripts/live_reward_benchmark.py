"""Run a small live Tinker RL benchmark and compare base vs trained metrics."""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime

import numpy as np
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

import config
from cache.store import read_json, write_json
from rl.grading import grade_response_exact
from rl.reward import compute_reward
from rl.train import run_training
from tts.scorer import compute_tts_for_cot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("TINKER_API_KEY", config.TINKER_API_KEY)


async def evaluate_model(
    service_client,
    model_ref: str,
    problems: list[dict],
    renderer,
    tokenizer,
    *,
    is_base_model: bool,
    max_tokens: int,
) -> dict:
    if is_base_model:
        sampling_client = await service_client.create_sampling_client_async(base_model=model_ref)
    else:
        sampling_client = await service_client.create_sampling_client_async(model_path=model_ref)

    rows = []
    for prob in problems:
        logger.info("Evaluating %s on %s", "base" if is_base_model else "trained", prob["id"])
        prompt = renderer.build_generation_prompt([{"role": "user", "content": prob["question"]}])
        result = await sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=renderer.get_stop_sequences(),
                max_tokens=max_tokens,
                temperature=0.0,
            ),
        )

        text = tokenizer.decode(result.sequences[0].tokens)
        import re

        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        thinking_text = think_match.group(1).strip() if think_match else text
        tts_result = await compute_tts_for_cot(
            sampling_client,
            renderer,
            prob["question"],
            prob["answer"],
            thinking_text,
            seed=42,
        )
        exact_correct = grade_response_exact(text, prob["answer"])
        step_texts = [s.step_text for s in tts_result.step_scores]
        reward = compute_reward(
            answer_correct=exact_correct,
            mean_tts=tts_result.mean_tts,
            decorative_fraction=tts_result.fraction_decorative,
            step_texts=step_texts,
        )

        rows.append({
            "id": prob["id"],
            "difficulty": prob.get("difficulty", ""),
            "correct": exact_correct,
            "early_exit_correct": tts_result.model_correct,
            "reward": reward,
            "mean_tts": tts_result.mean_tts,
            "n_steps": len(tts_result.step_scores),
            "decorative_fraction": tts_result.fraction_decorative,
            "frac_high_tts": tts_result.fraction_high_tts,
        })

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows])) if rows else 0.0

    return {
        "metrics": {
            "accuracy": mean("correct"),
            "mean_reward": mean("reward"),
            "mean_tts": mean("mean_tts"),
            "mean_steps": mean("n_steps"),
            "mean_decorative": mean("decorative_fraction"),
            "frac_high_tts": mean("frac_high_tts"),
        },
        "rows": rows,
    }


def metric_deltas(base: dict, trained: dict) -> dict:
    return {
        key: trained["metrics"][key] - base["metrics"][key]
        for key in base["metrics"]
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-problems", type=int, default=1)
    parser.add_argument("--eval-problems", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    args = parser.parse_args()

    with open(config.DATA_DIR / "examples.json") as f:
        all_problems = json.load(f)

    train_problems = all_problems[:args.train_problems]
    eval_problems = all_problems[:args.eval_problems]
    run_id = f"live_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    service_client = tinker.ServiceClient()
    renderer_name = model_info.get_recommended_renderer_name(config.BASE_MODEL)
    tokenizer = get_tokenizer(config.BASE_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    logger.info("Running baseline evaluation on %d problems", len(eval_problems))
    base_eval = await evaluate_model(
        service_client,
        config.BASE_MODEL,
        eval_problems,
        renderer,
        tokenizer,
        is_base_model=True,
        max_tokens=args.max_tokens,
    )

    logger.info("Running live RL training: run_id=%s", run_id)
    await run_training(
        run_id=run_id,
        problems=train_problems,
        num_steps=args.steps,
        group_size=args.group_size,
        groups_per_batch=args.groups_per_batch,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        max_tokens=args.max_tokens,
    )

    checkpoint = read_json(config.RUNS_DIR / run_id / "checkpoint.json")
    if not checkpoint:
        raise RuntimeError(f"Missing checkpoint for {run_id}")

    logger.info("Running trained checkpoint evaluation on %d problems", len(eval_problems))
    trained_eval = await evaluate_model(
        service_client,
        checkpoint["sampler_path"],
        eval_problems,
        renderer,
        tokenizer,
        is_base_model=False,
        max_tokens=args.max_tokens,
    )

    report = {
        "run_id": run_id,
        "model": config.BASE_MODEL,
        "timestamp": datetime.now().isoformat(),
        "train_problem_ids": [p["id"] for p in train_problems],
        "eval_problem_ids": [p["id"] for p in eval_problems],
        "params": vars(args),
        "checkpoint": checkpoint,
        "base": base_eval,
        "trained": trained_eval,
        "delta": metric_deltas(base_eval, trained_eval),
    }
    out_path = config.RUNS_DIR / run_id / "live_reward_benchmark.json"
    write_json(out_path, report)
    logger.info("Benchmark report written to %s", out_path)
    logger.info("Base metrics: %s", base_eval["metrics"])
    logger.info("Trained metrics: %s", trained_eval["metrics"])
    logger.info("Delta: %s", report["delta"])


if __name__ == "__main__":
    asyncio.run(main())
