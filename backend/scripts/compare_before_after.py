"""Benchmark: Compare baseline vs trained model on same problems."""

import asyncio
import json
import logging
import os
from datetime import datetime

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

import config
from rl.train import run_training
from tts.scorer import compute_tts_for_cot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("TINKER_API_KEY", config.TINKER_API_KEY)


async def evaluate_model(service_client, model_path, problems, tokenizer, renderer, is_baseline=False):
    """Evaluate a model on a set of problems."""
    if is_baseline:
        sampling_client = await service_client.create_sampling_client_async(base_model=model_path)
    else:
        sampling_client = await service_client.create_sampling_client_async(model_path=model_path)
    
    results = []
    for prob in problems:
        try:
            messages = [{"role": "user", "content": prob["question"]}]
            prompt = renderer.build_generation_prompt(messages)
            stop_seqs = renderer.get_stop_sequences()
            
            result = await sampling_client.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    stop=stop_seqs,
                    max_tokens=2048,
                    temperature=0.0,
                ),
            )
            
            text = tokenizer.decode(result.sequences[0].tokens)
            import re
            think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            thinking_text = think_match.group(1).strip() if think_match else text
            
            tts_result = await compute_tts_for_cot(
                sampling_client, renderer, prob["question"], prob["answer"], thinking_text, seed=42
            )
            
            results.append({
                "id": prob["id"],
                "correct": tts_result.model_correct,
                "mean_tts": tts_result.mean_tts,
                "n_steps": len(tts_result.step_scores),
                "decorative_frac": tts_result.fraction_decorative,
                "reward": 0.6 * (1.0 if tts_result.model_correct else 0.0) + 
                         0.3 * min(tts_result.mean_tts / 0.15, 1.0) -
                         0.1 * tts_result.fraction_decorative,
            })
        except Exception as e:
            logger.error(f"Eval failed for {prob['id']}: {e}")
    
    return results


async def main():
    with open(config.DATA_DIR / "examples.json") as f:
        all_problems = json.load(f)
    
    # Use first 3 problems for benchmark
    test_problems = all_problems[:3]
    
    service_client = tinker.ServiceClient()
    renderer_name = model_info.get_recommended_renderer_name(config.BASE_MODEL)
    tokenizer = get_tokenizer(config.BASE_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    
    logger.info("=== BASELINE EVALUATION ===")
    baseline_results = await evaluate_model(
        service_client, config.BASE_MODEL, test_problems, tokenizer, renderer, is_baseline=True
    )
    
    baseline_metrics = {
        "accuracy": sum(1 for r in baseline_results if r["correct"]) / len(baseline_results),
        "mean_tts": sum(r["mean_tts"] for r in baseline_results) / len(baseline_results),
        "mean_steps": sum(r["n_steps"] for r in baseline_results) / len(baseline_results),
        "mean_decorative": sum(r["decorative_frac"] for r in baseline_results) / len(baseline_results),
        "mean_reward": sum(r["reward"] for r in baseline_results) / len(baseline_results),
    }
    
    logger.info(f"Baseline: {baseline_metrics}")
    
    # Run training
    logger.info("\n=== TRAINING ===")
    run_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    await run_training(
        run_id=run_id,
        problems=test_problems,
        num_steps=3,
        group_size=3,
        learning_rate=1e-5,
        lora_rank=8,
    )
    
    # Load checkpoint
    import sys
    sys.path.insert(0, str(config.PROJECT_ROOT / "backend"))
    from cache.store import read_json
    checkpoint = read_json(config.RUNS_DIR / run_id / "checkpoint.json")
    
    logger.info("\n=== POST-TRAINING EVALUATION ===")
    trained_results = await evaluate_model(
        service_client, checkpoint["sampler_path"], test_problems, tokenizer, renderer
    )
    
    trained_metrics = {
        "accuracy": sum(1 for r in trained_results if r["correct"]) / len(trained_results),
        "mean_tts": sum(r["mean_tts"] for r in trained_results) / len(trained_results),
        "mean_steps": sum(r["n_steps"] for r in trained_results) / len(trained_results),
        "mean_decorative": sum(r["decorative_frac"] for r in trained_results) / len(trained_results),
        "mean_reward": sum(r["reward"] for r in trained_results) / len(trained_results),
    }
    
    logger.info(f"Trained: {trained_metrics}")
    
    # Compare
    logger.info("\n=== COMPARISON ===")
    for key in baseline_metrics:
        delta = trained_metrics[key] - baseline_metrics[key]
        pct = (delta / baseline_metrics[key] * 100) if baseline_metrics[key] != 0 else 0
        logger.info(f"{key}: {baseline_metrics[key]:.4f} → {trained_metrics[key]:.4f} ({delta:+.4f}, {pct:+.1f}%)")
    
    # Per-problem comparison
    logger.info("\n=== PER-PROBLEM ===")
    for b, t in zip(baseline_results, trained_results):
        logger.info(f"{b['id']}: reward={b['reward']:.3f}→{t['reward']:.3f}, "
                   f"tts={b['mean_tts']:.3f}→{t['mean_tts']:.3f}, "
                   f"steps={b['n_steps']}→{t['n_steps']}, "
                   f"dec={b['decorative_frac']:.2f}→{t['decorative_frac']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
