"""Quick test: Generate one completion and compute TTS reward."""

import asyncio
import logging
import os

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

import config
from rl.grading import grade_response_exact
from rl.reward import compute_reward
from tts.client import generate_cot_and_compute_tts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("TINKER_API_KEY", config.TINKER_API_KEY)

TEST_PROBLEMS = [
    {
        "id": "test_001",
        "question": "How many positive integers less than 100 are divisible by 3, 5, or 7?",
        "answer": "54",
    },
]


async def test_single_analysis():
    """Test generating CoT and computing TTS for one problem."""
    service_client = tinker.ServiceClient()
    problem = TEST_PROBLEMS[0]
    
    logger.info(f"Testing problem: {problem['question'][:60]}...")
    logger.info(f"Expected answer: {problem['answer']}")
    
    result = await generate_cot_and_compute_tts(
        service_client=service_client,
        model_name=config.BASE_MODEL,
        question=problem["question"],
        answer_str=problem["answer"],
        renderer_name=config.RENDERER_NAME,
    )
    
    logger.info("\n=== RESULT ===")
    logger.info(f"Correct: {result['model_correct']}")
    logger.info(f"Steps: {result['n_steps']}")
    logger.info(f"Mean TTS: {result['mean_tts']:.4f}")
    logger.info(f"High TTS (>=0.7): {result['frac_high_tts']:.1%}")
    logger.info(f"Decorative (<=0.005): {result['frac_decorative']:.1%}")
    logger.info(f"Self-verification steps: {result['n_self_verification']}")
    logger.info(f"SV decorative: {result['n_sv_decorative']}")
    
    reward = compute_reward(
        answer_correct=result["model_correct"],
        mean_tts=result["mean_tts"],
        decorative_fraction=result["frac_decorative"],
    )
    logger.info(f"\nReward: {reward:.4f}")
    
    return result


async def test_training_step():
    """Test one training step: generate, score, and update."""
    service_client = tinker.ServiceClient()
    problem = TEST_PROBLEMS[0]
    
    logger.info("=== Testing Training Step ===")
    
    # Create training client
    logger.info("Creating LoRA training client...")
    training_client = await service_client.create_lora_training_client_async(
        config.BASE_MODEL, rank=8  # Small rank for testing
    )
    
    # Save weights and get sampling client
    logger.info("Saving weights...")
    sampling_client = training_client.save_weights_and_get_sampling_client(name="test-step-0")
    
    # Get renderer
    renderer_name = model_info.get_recommended_renderer_name(config.BASE_MODEL)
    tokenizer = get_tokenizer(config.BASE_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    
    # Build prompt
    messages = [{"role": "user", "content": problem["question"]}]
    model_input = renderer.build_generation_prompt(messages)
    stop_seqs = renderer.get_stop_sequences()
    
    # Generate 2 rollouts
    logger.info("Generating 2 rollouts...")
    rollouts = []
    for i in range(2):
        result = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop_seqs,
                max_tokens=1024,
                temperature=0.7,
            ),
        )
        tokens = result.sequences[0].tokens
        text = tokenizer.decode(tokens)
        rollouts.append({"tokens": tokens, "text": text})
        logger.info(f"Rollout {i+1}: {len(tokens)} tokens")
    
    # Score each rollout with TTS
    logger.info("Computing TTS for each rollout...")
    rewards = []
    for i, rollout in enumerate(rollouts):
        try:
            # We need to re-analyze with TTS
            # For simplicity, extract answer and check correctness
            import re
            think_match = re.search(r"<think>(.*?)</think>", rollout["text"], re.DOTALL)
            thinking_text = think_match.group(1).strip() if think_match else rollout["text"]
            
            # Compute TTS
            from tts.scorer import compute_tts_for_cot
            tts_result = await compute_tts_for_cot(
                sampling_client, renderer, problem["question"], problem["answer"], thinking_text, seed=42
            )
            exact_correct = grade_response_exact(rollout["text"], problem["answer"])
            
            reward = compute_reward(
                answer_correct=exact_correct,
                mean_tts=tts_result.mean_tts,
                decorative_fraction=tts_result.fraction_decorative,
                step_texts=[s.step_text for s in tts_result.step_scores],
            )
            rewards.append(reward)
            
            logger.info(f"\nRollout {i+1}:")
            logger.info(f"  Correct: {exact_correct}")
            logger.info(f"  Steps: {len(tts_result.step_scores)}")
            logger.info(f"  Mean TTS: {tts_result.mean_tts:.4f}")
            logger.info(f"  Decorative: {tts_result.fraction_decorative:.1%}")
            logger.info(f"  Reward: {reward:.4f}")
            
        except Exception as e:
            logger.error(f"TTS failed for rollout {i+1}: {e}")
            rewards.append(0.0)
    
    # Compute advantages
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    advantages = [r - mean_reward for r in rewards]
    logger.info(f"\nMean reward: {mean_reward:.4f}")
    logger.info(f"Advantages: {[f'{a:.4f}' for a in advantages]}")
    
    # For a real training step, we would:
    # 1. Compute logprobs for each rollout under the current policy
    # 2. Build Datum objects with tokens, logprobs, advantages
    # 3. Call forward_backward with importance_sampling
    # 4. Call optim_step
    
    logger.info("\nNote: For demo purposes, skipping actual weight update.")
    logger.info("Full training loop would call forward_backward + optim_step here.")
    
    return rewards


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        asyncio.run(test_training_step())
    else:
        asyncio.run(test_single_analysis())
