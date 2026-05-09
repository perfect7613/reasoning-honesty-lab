"""Tests for issue detection and metrics computation."""

import pytest
from rl.issues import detect_issues, compute_metrics
from tts.scorer import StepTTS


class TestDetectIssues:
    def test_no_issues(self):
        steps = [
            StepTTS(0, "Step 1: calculate", 0.8, 0.9, 0.1, 0.9, 0.1, False),
            StepTTS(1, "Step 2: result", 0.9, 0.9, 0.1, 0.9, 0.1, False),
        ]
        issues = detect_issues(steps)
        assert len(issues) == 0

    def test_decorative_steps(self):
        steps = [
            StepTTS(0, "Let me think", 0.001, 0.9, 0.1, 0.9, 0.1, False),
            StepTTS(1, "Actually", 0.002, 0.9, 0.1, 0.9, 0.1, False),
            StepTTS(2, "Result is 42", 0.9, 0.9, 0.1, 0.9, 0.1, False),
        ]
        issues = detect_issues(steps)
        assert any("decorative" in i for i in issues)
        assert any("2 decorative" in i for i in issues)

    def test_fake_self_verification(self):
        steps = [
            StepTTS(0, "Calculate 5*5", 0.8, 0.9, 0.1, 0.9, 0.1, False),
            StepTTS(1, "Wait, let me re-check", 0.001, 0.9, 0.1, 0.9, 0.1, True),
        ]
        issues = detect_issues(steps)
        assert any("self-verification" in i for i in issues)
        assert any("performative" in i for i in issues)

    def test_verbose_reasoning(self):
        steps = [StepTTS(i, f"Step {i}", 0.1, 0.9, 0.1, 0.9, 0.1, False) for i in range(20)]
        issues = detect_issues(steps)
        assert any("verbose" in i for i in issues)


class TestComputeMetrics:
    def test_basic_metrics(self):
        steps = [
            StepTTS(0, "Step 1", 0.8, 0.9, 0.1, 0.9, 0.1, False),
            StepTTS(1, "Step 2", 0.002, 0.9, 0.1, 0.9, 0.1, False),
        ]
        metrics = compute_metrics(steps, model_correct=True)
        assert metrics["accuracy"] == 1.0
        assert metrics["mean_tts"] == pytest.approx(0.401, abs=0.01)
        assert metrics["frac_high_tts"] == 0.5
        assert metrics["frac_decorative"] == 0.5
        assert metrics["n_steps"] == 2

    def test_empty_steps(self):
        metrics = compute_metrics([], model_correct=False)
        assert metrics["accuracy"] == 0.0
        assert metrics["mean_tts"] == 0.0
        assert metrics["n_steps"] == 0
