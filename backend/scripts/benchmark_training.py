"""Benchmark: Run 5-step training on 2 problems and compare before/after."""

import asyncio
import json
import logging
import os
from datetime import datetime

import config
from rl.train import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("TINKER_API_KEY", config.TINKER_API_KEY)


async def main():
    with open(config.DATA_DIR / "examples.json") as f:
        all_problems = json.load(f)
    
    # Use first 2 problems
    test_problems = all_problems[:2]
    
    logger.info(f"Running benchmark: {len(test_problems)} problems, 5 steps...")
    
    await run_training(
        run_id=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        problems=test_problems,
        num_steps=5,
        group_size=2,
        learning_rate=1e-5,
        lora_rank=8,
    )
    
    logger.info("Benchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())
