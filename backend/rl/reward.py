"""Reward function for TTS-aware RL training."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Restructured weights to increase variance when accuracy saturates
ACCURACY_WEIGHT = 0.40
TTS_WEIGHT = 0.30
QUALITY_WEIGHT = 0.15
DECORATIVE_PENALTY_WEIGHT = 0.10
DEPTH_PENALTY_WEIGHT = 0.05

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
    accuracy_weight: float = ACCURACY_WEIGHT,
    tts_weight: float = TTS_WEIGHT,
    quality_weight: float = QUALITY_WEIGHT,
    decorative_penalty_weight: float = DECORATIVE_PENALTY_WEIGHT,
    depth_penalty_weight: float = DEPTH_PENALTY_WEIGHT,
) -> float:
    """Compute composite reward for a reasoning trace.
    
    reward = accuracy*0.40 + causal_frac*0.30 + quality*0.15 - decorative*0.10 - depth_penalty*0.05
    
    Args:
        answer_correct: Whether the final answer matches ground truth.
        mean_tts: Mean True-Thinking Score across all steps.
        decorative_fraction: Fraction of steps with TTS <= 0.005.
        step_texts: List of reasoning step texts for depth checking.
        accuracy_weight: Weight for answer correctness.
        tts_weight: Weight for mean TTS (normalized to [0, 1]).
        quality_weight: Weight for (1 - decorative) as explicit quality bonus.
        decorative_penalty_weight: Weight for decorative step penalty.
        depth_penalty_weight: Weight for too-short rollout penalty.
    
    Returns:
        Composite reward scalar, clipped to [-0.3, 1.0].
    """
    accuracy = 1.0 if answer_correct else 0.0
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
    
    reward = (
        accuracy_weight * accuracy
        + tts_weight * normalized_tts
        + quality_weight * (1.0 - decorative_fraction)
        - decorative_penalty_weight * decorative_fraction
        - depth_penalty_weight * depth_penalty
    )
    
    # Clip to prevent extreme values
    reward = float(np.clip(reward, -0.3, 1.0))
    
    logger.info(
        f"Reward: {reward:.4f} = {accuracy_weight}*{accuracy:.1f} + "
        f"{tts_weight}*{normalized_tts:.4f} + {quality_weight}*{1.0 - decorative_fraction:.4f} - "
        f"{decorative_penalty_weight}*{decorative_fraction:.4f} - {depth_penalty_weight}*{depth_penalty:.4f}"
    )
    
    return reward
