"""Step segmentation for chain-of-thought reasoning."""

import re

STEP_SPLIT_PATTERN = re.compile(
    r"(?:^|\n)"
    r"(?="
    r"(?:"
    r"\d+[\.\)]\s|"
    r"\*\*\w|"
    r"#{1,3}\s|"
    r"Step \d|First|Second|Third|Fourth|Fifth|Next|Then|Now|Finally|"
    r"So |Therefore|Thus|Hence|Let me|Let's|Wait|Hmm|Actually|"
    r"We (?:can|have|know|need|get|see|find|note|observe)|"
    r"I (?:can|need|should|will|notice|think|see)|"
    r"This (?:means|gives|implies|shows|is)|"
    r"Since |Because |Given |If |But |However |Note |"
    r"For |Using |Substitut|Comput|Calculat|Evaluat|Apply|Recall )"
    r")",
    re.MULTILINE,
)

SELF_VERIFICATION_PATTERN = re.compile(
    r"(?:Wait|Hmm|Actually|Let me (?:re-?check|re-?evaluate|verify|double.?check|reconsider)|"
    r"Hold on|No,? that's|That (?:doesn't|can't) be right|"
    r"Let me (?:think|reconsider) (?:about|again)|"
    r"I (?:made|think there's) (?:a|an) (?:mistake|error))",
    re.IGNORECASE,
)


def segment_cot_steps(cot_text: str, min_step_chars: int = 20) -> list[str]:
    """Split chain-of-thought text into reasoning steps.
    
    Uses heuristic patterns (step markers, discourse cues) to identify step boundaries.
    Merges very short segments with the previous step.
    """
    parts = STEP_SPLIT_PATTERN.split(cot_text)
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return [cot_text.strip()] if cot_text.strip() else []

    merged: list[str] = []
    for part in parts:
        if merged and len(part) < min_step_chars:
            merged[-1] = merged[-1] + "\n" + part
        else:
            merged.append(part)

    return merged


def is_self_verification_step(step_text: str) -> bool:
    """Check if a step looks like a self-verification / 'aha moment'."""
    return bool(SELF_VERIFICATION_PATTERN.search(step_text))
