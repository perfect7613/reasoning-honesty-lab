"""Quick verification: 1 step, 1 problem."""

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
    import json
    
    with open(config.DATA_DIR / "examples.json") as f:
        all_problems = json.load(f)
    
    # Use just 1 problem, 1 step for quick verification
    test_problems = all_problems[:1]
    
    logger.info(f"Running verification on {len(test_problems)} problem, 1 step...")
    
    await run_training(
        run_id=f"verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        problems=test_problems,
        num_steps=1,
        group_size=1,
        learning_rate=1e-5,
        lora_rank=8,
    )
    
    logger.info("Verification complete! Training loop works end-to-end.")


if __name__ == "__main__":
    asyncio.run(main())
