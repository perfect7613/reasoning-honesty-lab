"""Reward function for TTS-aware RL training."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Correctness is a hard gate. TTS only ranks traces that already answer correctly.
CORRECT_BASE_REWARD = 0.55
TTS_WEIGHT = 0.25
QUALITY_WEIGHT = 0.10
DECORATIVE_PENALTY_WEIGHT = 0.05
DEPTH_PENALTY_WEIGHT = 0.05
INCORRECT_BASE_REWARD = -0.20
INCORRECT_DECORATIVE_PENALTY_WEIGHT = 0.10

# TTS normalization constant
TTS_NORMALIZATION_MAX = 0.15

# Gaming detection thresholds
MIN_STEP_COUNT = 4
MIN_AVG_STEP_WORDS = 8


def compute_reward(
    answer_correct: bool,
    mean_tts: float,
    decorative_fraction: float,
    step_texts: list[str] | None = None,
    correct_base_reward: float = CORRECT_BASE_REWARD,
    tts_weight: float = TTS_WEIGHT,
    quality_weight: float = QUALITY_WEIGHT,
    decorative_penalty_weight: float = DECORATIVE_PENALTY_WEIGHT,
    depth_penalty_weight: float = DEPTH_PENALTY_WEIGHT,
    incorrect_base_reward: float = INCORRECT_BASE_REWARD,
    incorrect_decorative_penalty_weight: float = INCORRECT_DECORATIVE_PENALTY_WEIGHT,
) -> float:
    """Compute composite reward for a reasoning trace.
    
    Correctness is a hard gate: TTS can improve a correct answer, but it cannot
    rescue an incorrect answer. This prevents high-TTS-looking wrong traces from
    outranking correct traces.
    
    Args:
        answer_correct: Whether the final answer matches ground truth.
        mean_tts: Mean True-Thinking Score across all steps.
        decorative_fraction: Fraction of steps with TTS <= 0.005.
        step_texts: List of reasoning step texts for depth checking.
        correct_base_reward: Base reward for an exactly correct final answer.
        tts_weight: Weight for mean TTS (normalized to [0, 1]).
        quality_weight: Weight for (1 - decorative) as explicit quality bonus.
        decorative_penalty_weight: Weight for decorative step penalty.
        depth_penalty_weight: Weight for too-short rollout penalty.
        incorrect_base_reward: Base reward for an incorrect final answer.
        incorrect_decorative_penalty_weight: Extra penalty for decorative wrong traces.
    
    Returns:
        Composite reward scalar, clipped to [-0.3, 1.0].
    """
    normalized_tts = min(mean_tts / TTS_NORMALIZATION_MAX, 1.0)
    
    # Gaming / depth penalty
    depth_penalty = 0.0
    if step_texts:
        n_steps = len(step_texts)
        avg_words = np.mean([len(s.split()) for s in step_texts]) if step_texts else 0
        
        if n_steps < MIN_STEP_COUNT:
            depth_penalty += 0.20
            logger.info(f"Depth penalty: only {n_steps} steps (min {MIN_STEP_COUNT})")
        if avg_words < MIN_AVG_STEP_WORDS:
            depth_penalty += 0.15
            logger.info(f"Depth penalty: avg {avg_words:.1f} words/step (min {MIN_AVG_STEP_WORDS})")
    
    if not answer_correct:
        reward = (
            incorrect_base_reward
            - incorrect_decorative_penalty_weight * decorative_fraction
            - depth_penalty_weight * depth_penalty
        )
    else:
        reward = (
            correct_base_reward
            + tts_weight * normalized_tts
            + quality_weight * (1.0 - decorative_fraction)
            - decorative_penalty_weight * decorative_fraction
            - depth_penalty_weight * depth_penalty
        )
    
    # Clip to prevent extreme values
    reward = float(np.clip(reward, -0.3, 1.0))
    
    logger.info(
        f"Reward: {reward:.4f} (correct={answer_correct}) = "
        f"tts {tts_weight}*{normalized_tts:.4f} + quality {quality_weight}*{1.0 - decorative_fraction:.4f} - "
        f"{decorative_penalty_weight}*{decorative_fraction:.4f} - {depth_penalty_weight}*{depth_penalty:.4f}"
    )
    
    return reward
