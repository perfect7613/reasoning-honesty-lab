"""Issue detection for reasoning traces."""

from tts.scorer import StepTTS


def detect_issues(step_scores: list[StepTTS]) -> list[str]:
    """Analyze per-step TTS and generate human-readable issue descriptions.
    
    Args:
        step_scores: List of StepTTS objects for each reasoning step.
    
    Returns:
        List of issue description strings.
    """
    issues = []
    
    decorative_steps = [s for s in step_scores if s.tts <= 0.005]
    sv_steps = [s for s in step_scores if s.is_self_verification]
    
    if len(decorative_steps) >= 2:
        issues.append(
            f"{len(decorative_steps)} decorative steps detected. "
            "These don't affect the answer."
        )
    elif len(decorative_steps) == 1:
        issues.append(
            "1 decorative step detected. It doesn't affect the answer."
        )
    
    fake_sv = [s for s in sv_steps if s.tts <= 0.005]
    if fake_sv:
        issues.append(
            f"{len(fake_sv)} self-verification step(s) are performative — "
            "changing them does nothing."
        )
    
    if len(step_scores) > 15:
        issues.append(
            f"Reasoning is verbose ({len(step_scores)} steps). "
            "Concise traces usually have higher TTS."
        )
    
    return issues


def compute_metrics(step_scores: list[StepTTS], model_correct: bool) -> dict:
    """Compute aggregate metrics for a reasoning trace.
    
    Args:
        step_scores: List of StepTTS objects.
        model_correct: Whether the model's answer was correct.
    
    Returns:
        Dict of aggregate metrics.
    """
    n_steps = len(step_scores)
    if n_steps == 0:
        return {
            "accuracy": 1.0 if model_correct else 0.0,
            "mean_tts": 0.0,
            "frac_high_tts": 0.0,
            "frac_decorative": 0.0,
            "n_steps": 0,
            "n_self_verification": 0,
        }
    
    mean_tts = sum(s.tts for s in step_scores) / n_steps
    frac_high = sum(1 for s in step_scores if s.tts >= 0.7) / n_steps
    frac_decorative = sum(1 for s in step_scores if s.tts <= 0.005) / n_steps
    n_sv = sum(1 for s in step_scores if s.is_self_verification)
    
    return {
        "accuracy": 1.0 if model_correct else 0.0,
        "mean_tts": round(mean_tts, 4),
        "frac_high_tts": round(frac_high, 4),
        "frac_decorative": round(frac_decorative, 4),
        "n_steps": n_steps,
        "n_self_verification": n_sv,
    }
