"""Precompute baseline TTS analyses for all curated problems."""

import asyncio
import json
import logging
from datetime import datetime

import tinker

import config
from cache.store import write_json
from tts.client import generate_cot_and_compute_tts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def precompute_baseline():
    """Generate baseline TTS for all problems in examples.json."""
    problems_path = config.DATA_DIR / "examples.json"
    with open(problems_path) as f:
        problems = json.load(f)

    logger.info(f"Precomputing baseline for {len(problems)} problems...")

    service_client = tinker.ServiceClient()
    analyses = []

    for i, problem in enumerate(problems):
        logger.info(f"[{i + 1}/{len(problems)}] {problem['question'][:60]}...")
        try:
            result = await generate_cot_and_compute_tts(
                service_client=service_client,
                model_name=config.BASE_MODEL,
                question=problem["question"],
                answer_str=problem["answer"],
                renderer_name=config.RENDERER_NAME,
            )
            result["id"] = problem["id"]
            analyses.append(result)
            logger.info(f"  -> {result['n_steps']} steps, mean_tts={result['mean_tts']:.4f}")
        except Exception as e:
            logger.error(f"  -> Failed: {e}")
            analyses.append({
                "id": problem["id"],
                "question": problem["question"],
                "answer": problem["answer"],
                "error": str(e),
            })

    baseline_data = {
        "model": config.BASE_MODEL,
        "timestamp": datetime.now().isoformat(),
        "n_problems": len(problems),
        "analyses": analyses,
    }

    write_json(config.BASELINE_DIR / "baseline_analysis.json", baseline_data)
    logger.info(f"Baseline saved to {config.BASELINE_DIR / 'baseline_analysis.json'}")


if __name__ == "__main__":
    asyncio.run(precompute_baseline())
