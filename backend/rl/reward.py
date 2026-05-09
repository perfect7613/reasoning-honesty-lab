"""Reward function for TTS-aware RL training."""

import logging

logger = logging.getLogger(__name__)

# Default reward weights (configurable)
ACCURACY_WEIGHT = 0.60
TTS_WEIGHT = 0.30
DECORATIVE_PENALTY_WEIGHT = 0.10

# TTS normalization constant (DeepSeek-V3.1 max observed mean TTS ≈ 0.144)
TTS_NORMALIZATION_MAX = 0.15


def compute_reward(
    answer_correct: bool,
    mean_tts: float,
    decorative_fraction: float,
    accuracy_weight: float = ACCURACY_WEIGHT,
    tts_weight: float = TTS_WEIGHT,
    decorative_penalty_weight: float = DECORATIVE_PENALTY_WEIGHT,
) -> float:
    """Compute composite reward for a reasoning trace.
    
    reward = accuracy_weight * accuracy + tts_weight * normalized_mean_tts - decorative_penalty_weight * decorative_fraction
    
    Args:
        answer_correct: Whether the final answer matches ground truth.
        mean_tts: Mean True-Thinking Score across all steps.
        decorative_fraction: Fraction of steps with TTS <= 0.005.
        accuracy_weight: Weight for answer correctness.
        tts_weight: Weight for mean TTS (normalized to [0, 1]).
        decorative_penalty_weight: Weight for decorative step penalty.
    
    Returns:
        Composite reward scalar.
    """
    accuracy = 1.0 if answer_correct else 0.0
    normalized_tts = min(mean_tts / TTS_NORMALIZATION_MAX, 1.0)

    reward = (
        accuracy_weight * accuracy
        + tts_weight * normalized_tts
        - decorative_penalty_weight * decorative_fraction
    )

    logger.debug(
        f"Reward: {reward:.4f} = {accuracy_weight}*{accuracy:.1f} + "
        f"{tts_weight}*{normalized_tts:.4f} - {decorative_penalty_weight}*{decorative_fraction:.4f}"
    )

    return reward
