"""Tinker client wrapper for generating CoT and computing TTS."""

import logging
import re

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from rl.grading import grade_response_exact
from tts.scorer import compute_tts_for_cot

logger = logging.getLogger(__name__)


async def generate_cot_and_compute_tts(
    service_client: tinker.ServiceClient,
    model_name: str,
    question: str,
    answer_str: str,
    max_tokens: int = 4096,
    seed: int = 42,
    renderer_name: str | None = None,
) -> dict:
    """Generate chain-of-thought and compute TTS for all steps.
    
    End-to-end: generates CoT from the model via greedy decoding,
    then computes TTS for each reasoning step.
    
    Returns the TTSResult.summary() dict.
    """
    if renderer_name is None:
        renderer_name = model_info.get_recommended_renderer_name(model_name)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    sampling_client = await service_client.create_sampling_client_async(base_model=model_name)

    messages = [
        {"role": "user", "content": question},
    ]
    model_input = renderer.build_generation_prompt(messages)
    stop_seqs = renderer.get_stop_sequences()

    logger.info(f"Generating CoT for: {question[:80]}...")
    sample_result = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            stop=stop_seqs,
            max_tokens=max_tokens,
            temperature=0.0,
        ),
    )

    cot_tokens = sample_result.sequences[0].tokens
    cot_text_raw = tokenizer.decode(cot_tokens)
    assert isinstance(cot_text_raw, str)
    cot_text: str = cot_text_raw
    logger.info(f"Generated {len(cot_tokens)} tokens of CoT")

    think_match = re.search(r"<think>(.*?)</think>", cot_text, re.DOTALL)
    if think_match:
        thinking_text = think_match.group(1).strip()
        final_part = cot_text[think_match.end() :].strip()
    else:
        thinking_text = cot_text
        final_part = ""

    logger.info(f"CoT: {len(thinking_text)} chars")
    if final_part:
        logger.info(f"Final answer: {final_part[:200]}")

    tts_result = await compute_tts_for_cot(
        sampling_client, renderer, question, answer_str, thinking_text, seed=seed
    )

    # Text-based answer verification for UI metrics. Training uses this exact
    # boxed-answer grade directly instead of early-exit confidence.
    text_correct = grade_response_exact(cot_text_raw, answer_str)
    if tts_result.model_correct and not text_correct:
        logger.info(
            "Early-exit confidence favored the gold answer, but the generated final boxed answer was not exact."
        )

    summary = tts_result.summary()
    summary["cot_text"] = tts_result.cot_text
    summary["early_exit_correct"] = tts_result.model_correct
    summary["model_correct"] = text_correct
    return summary
