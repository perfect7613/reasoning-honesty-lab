"""Tests for TTS segmenter and scorer."""

import pytest
from tts.segmenter import is_self_verification_step, segment_cot_steps
from tts.perturb import has_numbers, perturb_numbers


class TestSegmenter:
    def test_segment_cot_steps_basic(self):
        cot = """Let me think about this problem.
We need to find the sum of all integers.
First, consider the range from 1 to 10.
Then, use the formula n*(n+1)/2.
Therefore, the answer is 55."""
        steps = segment_cot_steps(cot)
        assert len(steps) >= 3
        assert any("formula" in s for s in steps)

    def test_segment_cot_steps_single_step(self):
        cot = "The answer is simply 42."
        steps = segment_cot_steps(cot)
        assert len(steps) == 1
        assert steps[0] == "The answer is simply 42."

    def test_segment_cot_steps_empty(self):
        steps = segment_cot_steps("")
        assert steps == []

    def test_is_self_verification_wait(self):
        assert is_self_verification_step("Wait, let me re-check that.")
        assert is_self_verification_step("Hmm, that doesn't seem right.")
        assert not is_self_verification_step("The sum is 42.")

    def test_is_self_verification_double_check(self):
        assert is_self_verification_step("Let me double-check my work.")
        assert is_self_verification_step("Actually, I made a mistake.")


class TestPerturb:
    def test_has_numbers_yes(self):
        assert has_numbers("The sum is 42.")
        assert has_numbers("x = 3.14")

    def test_has_numbers_no(self):
        assert not has_numbers("The answer is clear.")
        assert not has_numbers("Let me think about this.")

    def test_perturb_numbers_basic(self):
        import random
        rng = random.Random(42)
        result = perturb_numbers("The sum is 42 and the product is 10.", rng)
        assert result != "The sum is 42 and the product is 10."
        assert "42" not in result or "10" not in result

    def test_perturb_numbers_no_numbers(self):
        text = "There are no numbers here."
        assert perturb_numbers(text) == text


class TestScorerMock:
    """Mock-based tests for the scorer logic."""

    @pytest.mark.asyncio
    async def test_compute_tts_for_cot_simple(self, monkeypatch):
        """Test that TTS scoring produces expected ordinal relationships."""
        from tts.scorer import compute_tts_for_cot, StepTTS

        # Mock compute_tts_for_step to return known values directly
        call_idx = 0
        async def mock_step(*args, **kwargs):
            nonlocal call_idx
            step_index = kwargs.get("step_index", args[6] if len(args) > 6 else 0)
            step_text = kwargs.get("step", args[5] if len(args) > 5 else "")
            
            # Step 0: low TTS (decorative intro)
            # Step 1: high TTS (actual calculation)
            if step_index == 0:
                tts = 0.02
            else:
                tts = 0.85
            
            call_idx += 1
            return StepTTS(
                step_index=step_index,
                step_text=step_text,
                tts=tts,
                s1_c1=0.9 if tts > 0.5 else 0.3,
                s0_c1=0.1,
                s1_c0=0.9 if tts > 0.5 else 0.3,
                s0_c0=0.1,
                is_self_verification=False,
            )

        monkeypatch.setattr("tts.scorer.compute_tts_for_step", mock_step)

        result = await compute_tts_for_cot(
            sampling_client=None,
            renderer=None,
            question="What is 2+2?",
            answer_str="4",
            cot_text="Let me think about this.\nFirst, 2 plus 2 equals 4.",
            seed=42,
        )

        assert result.question == "What is 2+2?"
        assert result.answer == "4"
        assert len(result.step_scores) == 2
        # First step is decorative (low TTS)
        assert result.step_scores[0].tts < 0.1
        # Second step is causal (high TTS)
        assert result.step_scores[1].tts > 0.5
        assert result.model_correct

    @pytest.mark.asyncio
    async def test_compute_tts_for_cot_empty(self):
        """Test empty CoT handling."""
        from tts.scorer import compute_tts_for_cot

        result = await compute_tts_for_cot(
            sampling_client=None,
            renderer=None,
            question="What is 2+2?",
            answer_str="4",
            cot_text="",
            seed=42,
        )

        assert result.step_scores == []
        assert not result.model_correct

    @pytest.mark.asyncio
    async def test_compute_tts_decorative_step(self, monkeypatch):
        """Test that a decorative step gets low TTS."""
        from tts.scorer import compute_tts_for_cot

        async def mock_confidence(*args, **kwargs):
            # All perturbations give same confidence -> TTS = 0
            return 0.5

        monkeypatch.setattr("tts.scorer.compute_early_exit_confidence", mock_confidence)

        result = await compute_tts_for_cot(
            sampling_client=None,
            renderer=None,
            question="What is 2+2?",
            answer_str="4",
            cot_text="Let me understand the problem.\n2 plus 2 equals 4.",
            seed=42,
        )

        # First step is decorative (intro filler)
        assert result.step_scores[0].tts < 0.01
