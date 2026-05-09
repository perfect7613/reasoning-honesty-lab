"""Tests for reward function and issue detection."""

import pytest
from rl.reward import compute_reward


class TestReward:
    def test_correct_high_tts_low_decorative(self):
        """Best case: correct, high TTS, no decorative steps."""
        reward = compute_reward(
            answer_correct=True,
            mean_tts=0.15,
            decorative_fraction=0.0,
        )
        assert reward > 0.8

    def test_correct_low_tts_high_decorative(self):
        """Bad reasoning but correct answer."""
        reward = compute_reward(
            answer_correct=True,
            mean_tts=0.02,
            decorative_fraction=0.6,
        )
        assert reward < 0.7  # Still rewarded for correctness, but lower than good reasoning

    def test_incorrect_any_tts(self):
        """Wrong answer gets heavily penalized."""
        reward_correct = compute_reward(
            answer_correct=True,
            mean_tts=0.10,
            decorative_fraction=0.2,
        )
        reward_incorrect = compute_reward(
            answer_correct=False,
            mean_tts=0.10,
            decorative_fraction=0.2,
        )
        assert reward_correct > reward_incorrect
        assert reward_incorrect < 0.4

    def test_tts_monotonicity(self):
        """Higher TTS should give higher reward, all else equal."""
        r_low = compute_reward(True, 0.05, 0.2)
        r_high = compute_reward(True, 0.15, 0.2)
        assert r_high > r_low

    def test_decorative_penalty(self):
        """More decorative steps should lower reward."""
        r_low = compute_reward(True, 0.10, 0.1)
        r_high = compute_reward(True, 0.10, 0.5)
        assert r_low > r_high

    def test_custom_weights(self):
        """Custom weights should change reward."""
        r_default = compute_reward(True, 0.10, 0.2)
        r_custom = compute_reward(
            True, 0.10, 0.2,
            accuracy_weight=0.5,
            tts_weight=0.4,
            decorative_penalty_weight=0.1,
        )
        assert r_custom != r_default
