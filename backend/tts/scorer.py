"""True-Thinking Score (TTS) computation.

Implements the TTS metric from "Can Aha Moments Be Fake?" (Zhao et al., 2025).
"""

import asyncio
import logging
import math
import random
import re
from dataclasses import dataclass

from tts.segmenter import is_self_verification_step, segment_cot_steps
from tts.perturb import has_numbers, perturb_numbers

logger = logging.getLogger(__name__)

EARLY_EXIT_CUE = "\n</think>\n\nThe answer is \\boxed{"
EARLY_EXIT_SUFFIX = "}"


@dataclass
class StepTTS:
    """TTS measurement for a single reasoning step."""

    step_index: int
    step_text: str
    tts: float
    s1_c1: float  # P(y* | intact context, intact step)
    s0_c1: float  # P(y* | intact context, perturbed step)
    s1_c0: float  # P(y* | perturbed context, intact step)
    s0_c0: float  # P(y* | perturbed context, perturbed step)
    is_self_verification: bool = False


@dataclass
class TTSResult:
    """TTS analysis for a complete chain-of-thought."""

    question: str
    answer: str
    cot_text: str
    step_scores: list[StepTTS]
    model_correct: bool

    @property
    def mean_tts(self) -> float:
        if not self.step_scores:
            return 0.0
        return sum(s.tts for s in self.step_scores) / len(self.step_scores)

    @property
    def fraction_high_tts(self) -> float:
        """Fraction of steps with TTS >= 0.7."""
        if not self.step_scores:
            return 0.0
        return sum(1 for s in self.step_scores if s.tts >= 0.7) / len(self.step_scores)

    @property
    def fraction_decorative(self) -> float:
        """Fraction of steps with TTS <= 0.005."""
        if not self.step_scores:
            return 0.0
        return sum(1 for s in self.step_scores if s.tts <= 0.005) / len(self.step_scores)

    @property
    def self_verification_steps(self) -> list[StepTTS]:
        return [s for s in self.step_scores if s.is_self_verification]

    def summary(self) -> dict:
        sv_steps = self.self_verification_steps
        return {
            "question": self.question[:100],
            "answer": self.answer,
            "model_correct": self.model_correct,
            "n_steps": len(self.step_scores),
            "mean_tts": round(self.mean_tts, 4),
            "frac_high_tts": round(self.fraction_high_tts, 4),
            "frac_decorative": round(self.fraction_decorative, 4),
            "n_self_verification": len(sv_steps),
            "n_sv_decorative": sum(1 for s in sv_steps if s.tts <= 0.005),
            "per_step_tts": [round(s.tts, 4) for s in self.step_scores],
        }


async def compute_early_exit_confidence(
    sampling_client,
    renderer,
    question: str,
    cot_prefix: str,
    answer_str: str,
) -> float:
    """Compute P(y* | question, cot_prefix) via early-exit prompting."""
    tokenizer = renderer.tokenizer

    messages = [
        {"role": "user", "content": question},
    ]

    base_prompt = renderer.build_generation_prompt(messages)

    prefix_text = cot_prefix + EARLY_EXIT_CUE
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)

    answer_text = answer_str + EARLY_EXIT_SUFFIX
    answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)

    if not answer_tokens:
        logger.warning("No answer tokens after tokenization; returning 0.0")
        return 0.0

    import tinker

    full_input = base_prompt.append(tinker.EncodedTextChunk(tokens=prefix_tokens))
    answer_start_pos = full_input.length
    full_input = full_input.append(tinker.EncodedTextChunk(tokens=answer_tokens))

    logprobs = await sampling_client.compute_logprobs_async(full_input)

    answer_lps = logprobs[answer_start_pos - 1 : answer_start_pos - 1 + len(answer_tokens)]

    if not answer_lps:
        return 0.0

    total_logprob = sum(lp for lp in answer_lps if lp is not None)
    total_logprob = max(total_logprob, -50.0)
    return math.exp(total_logprob)


async def compute_tts_for_step(
    sampling_client,
    renderer,
    question: str,
    answer_str: str,
    preceding_steps: list[str],
    step: str,
    step_index: int,
    rng: random.Random,
) -> StepTTS:
    """Compute TTS for a single reasoning step."""
    intact_context = "\n".join(preceding_steps)
    perturbed_context = "\n".join(perturb_numbers(s, rng) for s in preceding_steps)
    intact_step = step

    if has_numbers(step):
        perturbed_step = perturb_numbers(step, rng)
    else:
        perturbed_step = ""

    def _build_prefix(context: str, step_text: str) -> str:
        parts = [p for p in [context, step_text] if p]
        return "\n".join(parts)

    prefix_s1_c1 = _build_prefix(intact_context, intact_step)
    prefix_s0_c1 = _build_prefix(intact_context, perturbed_step)
    prefix_s1_c0 = _build_prefix(perturbed_context, intact_step)
    prefix_s0_c0 = _build_prefix(perturbed_context, perturbed_step)

    s1_c1, s0_c1, s1_c0, s0_c0 = await asyncio.gather(
        compute_early_exit_confidence(sampling_client, renderer, question, prefix_s1_c1, answer_str),
        compute_early_exit_confidence(sampling_client, renderer, question, prefix_s0_c1, answer_str),
        compute_early_exit_confidence(sampling_client, renderer, question, prefix_s1_c0, answer_str),
        compute_early_exit_confidence(sampling_client, renderer, question, prefix_s0_c0, answer_str),
    )

    tts = 0.5 * (abs(s1_c1 - s0_c1) + abs(s1_c0 - s0_c0))

    return StepTTS(
        step_index=step_index,
        step_text=step,
        tts=tts,
        s1_c1=s1_c1,
        s0_c1=s0_c1,
        s1_c0=s1_c0,
        s0_c0=s0_c0,
        is_self_verification=is_self_verification_step(step),
    )


async def compute_tts_for_cot(
    sampling_client,
    renderer,
    question: str,
    answer_str: str,
    cot_text: str,
    seed: int = 42,
) -> TTSResult:
    """Compute TTS for all steps in a chain-of-thought."""
    rng = random.Random(seed)
    steps = segment_cot_steps(cot_text)

    if not steps:
        return TTSResult(
            question=question,
            answer=answer_str,
            cot_text=cot_text,
            step_scores=[],
            model_correct=False,
        )

    logger.info(f"Computing TTS for {len(steps)} steps...")

    step_scores = []
    for i, step in enumerate(steps):
        preceding = steps[:i]
        score = await compute_tts_for_step(
            sampling_client, renderer, question, answer_str, preceding, step, i, rng
        )
        step_scores.append(score)
        logger.info(
            f"  Step {i}/{len(steps) - 1}: TTS={score.tts:.4f} "
            f"[S1C1={score.s1_c1:.4f} S0C1={score.s0_c1:.4f} "
            f"S1C0={score.s1_c0:.4f} S0C0={score.s0_c0:.4f}] "
            f"SV={score.is_self_verification}"
        )

    # Apply TTS floor to prevent "all decorative" collapse when model is wrong
    # but still producing text. Steps with substantial content get baseline TTS.
    TTS_FLOOR = 0.002
    for score in step_scores:
        if score.tts < TTS_FLOOR and len(score.step_text) > 30:
            score.tts = TTS_FLOOR

    model_correct = step_scores[-1].s1_c1 > 0.5 if step_scores else False

    return TTSResult(
        question=question,
        answer=answer_str,
        cot_text=cot_text,
        step_scores=step_scores,
        model_correct=model_correct,
    )
