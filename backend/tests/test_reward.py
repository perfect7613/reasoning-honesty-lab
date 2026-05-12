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
            step_texts=["step one has many words", "step two also has enough", "step three is substantive", "step four concludes well"],
        )
        assert reward > 0.7

    def test_correct_low_tts_high_decorative(self):
        """Bad reasoning but correct answer."""
        reward = compute_reward(
            answer_correct=True,
            mean_tts=0.02,
            decorative_fraction=0.6,
            step_texts=["step one has many words", "step two also has enough", "step three is substantive", "step four concludes well"],
        )
        assert reward < 0.7

    def test_incorrect_any_tts(self):
        """Wrong answer gets heavily penalized."""
        step_texts = ["step one has many words", "step two also has enough", "step three is substantive", "step four concludes well"]
        reward_correct = compute_reward(
            answer_correct=True,
            mean_tts=0.10,
            decorative_fraction=0.2,
            step_texts=step_texts,
        )
        reward_incorrect = compute_reward(
            answer_correct=False,
            mean_tts=0.10,
            decorative_fraction=0.2,
            step_texts=step_texts,
        )
        assert reward_correct > reward_incorrect
        assert reward_incorrect <= 0.0

    def test_incorrect_high_tts_cannot_beat_correct_decorative(self):
        """TTS cannot rescue an incorrect final answer."""
        step_texts = ["this step contains enough words to pass the depth check"] * 4
        wrong_high_tts = compute_reward(False, 0.15, 0.0, step_texts=step_texts)
        correct_decorative = compute_reward(True, 0.0, 1.0, step_texts=step_texts)
        assert wrong_high_tts < correct_decorative
        assert wrong_high_tts <= 0.0

    def test_tts_monotonicity(self):
        """Higher TTS should give higher reward, all else equal."""
        step_texts = ["step one has many words", "step two also has enough", "step three is substantive", "step four concludes well"]
        r_low = compute_reward(True, 0.05, 0.2, step_texts=step_texts)
        r_high = compute_reward(True, 0.15, 0.2, step_texts=step_texts)
        assert r_high > r_low

    def test_decorative_penalty(self):
        """More decorative steps should lower reward."""
        step_texts = ["step one has many words", "step two also has enough", "step three is substantive", "step four concludes well"]
        r_low = compute_reward(True, 0.10, 0.1, step_texts=step_texts)
        r_high = compute_reward(True, 0.10, 0.5, step_texts=step_texts)
        assert r_low > r_high

    def test_depth_penalty_short_rollout(self):
        """Too-short rollouts get penalised."""
        long_steps = ["step one has many words", "step two also has enough", "step three is substantive", "step four concludes well"]
        short_steps = ["x = 5", "ans = 10"]
        
        r_long = compute_reward(True, 0.10, 0.1, step_texts=long_steps)
        r_short = compute_reward(True, 0.10, 0.1, step_texts=short_steps)
        assert r_long > r_short

    def test_depth_penalty_shallow_steps(self):
        """Steps with < 8 avg words get penalised."""
        deep_steps = ["this reasoning step contains many words for deep analysis", "another substantial step with sufficient mathematical content", "step three is substantive detailed and thorough in reasoning", "step four concludes with comprehensive analysis and explanation"]
        shallow_steps = ["a = 1", "b = 2", "c = 3", "d = 4"]
        
        r_deep = compute_reward(True, 0.10, 0.1, step_texts=deep_steps)
        r_shallow = compute_reward(True, 0.10, 0.1, step_texts=shallow_steps)
        assert r_deep > r_shallow

    def test_reward_clipping(self):
        """Reward should be clipped to [-0.3, 1.0]."""
        r_max = compute_reward(True, 1.0, 0.0, step_texts=None)
        r_min = compute_reward(False, 0.0, 1.0, step_texts=None)
        assert r_max <= 1.0
        assert r_min >= -0.3


class TestExactGrading:
    def test_grade_response_exact_last_boxed_answer(self):
        from rl.grading import extract_boxed_answer, grade_response_exact

        response = "Earlier I thought \\boxed{41}, but after recomputing the result is \\boxed{42}."
        assert extract_boxed_answer(response) == "42"
        assert grade_response_exact(response, "42")

    def test_grade_response_rejects_answer_substring_without_boxed_final(self):
        from rl.grading import grade_response_exact

        response = "The number 42 appears in my work, but I never provide a boxed final answer."
        assert not grade_response_exact(response, "42")

    def test_grade_response_rejects_wrong_final_even_if_gold_appears(self):
        from rl.grading import grade_response_exact

        response = "I considered 42 during the derivation, then concluded \\boxed{41}."
        assert not grade_response_exact(response, "42")
