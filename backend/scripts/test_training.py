"""Quick test: Run a tiny training job to verify the loop works."""

import asyncio
import logging
import os
from datetime import datetime

import config
from rl.train import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("TINKER_API_KEY", config.TINKER_API_KEY)


async def main():
    """Run a minimal training job: 2 problems, 2 steps."""
    import json
    
    with open(config.DATA_DIR / "examples.json") as f:
        all_problems = json.load(f)
    
    # Use only 2 simple problems for testing
    test_problems = all_problems[:2]
    
    logger.info(f"Running mini training job on {len(test_problems)} problems...")
    logger.info("This will take a few minutes and consume API credits.")
    
    await run_training(
        run_id=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        problems=test_problems,
        num_steps=2,
        group_size=2,
        learning_rate=1e-5,
        lora_rank=8,
    )
    
    logger.info("Training complete! Check data/runs/ for results.")


if __name__ == "__main__":
    asyncio.run(main())
