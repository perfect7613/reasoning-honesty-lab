"""RL training loop with TTS-aware reward and proper checkpointing."""

import asyncio
import json
import logging
import random
import re
from datetime import datetime

import numpy as np
import tinker
import torch
from tinker import TensorData
from tinker_cookbook import model_info, renderers
from tinker_cookbook.rl.train import _remove_mask
from tinker_cookbook.supervised.common import (
    create_rightshifted_model_input_and_leftshifted_targets,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

import config
from cache.store import append_jsonl, write_json
from rl.reward import compute_reward
from tts.scorer import compute_tts_for_cot

logger = logging.getLogger(__name__)


async def run_training(
    run_id: str,
    problems: list[dict],
    num_steps: int = 20,
    group_size: int = 8,
    learning_rate: float = 1e-5,
    lora_rank: int = 32,
):
    """Run RL training loop with TTS-aware reward."""
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "run_id": run_id,
        "model": config.BASE_MODEL,
        "num_steps": num_steps,
        "group_size": group_size,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "started_at": datetime.now().isoformat(),
        "problems": [p["id"] for p in problems],
    }
    write_json(run_dir / "config.json", config_data)

    status = {
        "run_id": run_id,
        "status": "running",
        "current_step": 0,
        "total_steps": num_steps,
        "latest_metrics": {},
        "logs": [],
    }
    write_json(run_dir / "status.json", status)

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        config.BASE_MODEL, rank=lora_rank
    )

    renderer_name = model_info.get_recommended_renderer_name(config.BASE_MODEL)
    tokenizer = get_tokenizer(config.BASE_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    try:
        for step in range(num_steps):
            logger.info(f"=== Training Step {step + 1}/{num_steps} ===")

            # Get sampling client for current weights
            sampling_client = await training_client.save_weights_and_get_sampling_client_async(
                name=f"{run_id}-step-{step}"
            )
            logger.info(f"Training run ID: {training_client.model_id}")

            # Select problems for this step
            selected_problems = random.sample(problems, min(group_size, len(problems)))

            # Generate rollouts and compute rewards
            rollouts = []
            for prob in selected_problems:
                messages = [{"role": "user", "content": prob["question"]}]
                prompt = renderer.build_generation_prompt(messages)
                stop_seqs = renderer.get_stop_sequences()

                result = await sampling_client.sample_async(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        stop=stop_seqs,
                        max_tokens=2048,
                        temperature=0.8,
                    ),
                )

                seq = result.sequences[0]
                tokens = seq.tokens
                logprobs = seq.logprobs  # From sampling, NOT compute_logprobs_async
                text = tokenizer.decode(tokens)

                # Compute TTS reward
                try:
                    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
                    thinking_text = think_match.group(1).strip() if think_match else text

                    tts_result = await compute_tts_for_cot(
                        sampling_client, renderer, prob["question"], prob["answer"], thinking_text, seed=42
                    )

                    step_texts = [s.step_text for s in tts_result.step_scores]
                    reward = compute_reward(
                        answer_correct=tts_result.model_correct,
                        mean_tts=tts_result.mean_tts,
                        decorative_fraction=tts_result.fraction_decorative,
                        step_texts=step_texts,
                    )

                    rollouts.append({
                        "problem": prob,
                        "tokens": tokens,
                        "logprobs": logprobs,
                        "text": text,
                        "reward": reward,
                        "correct": tts_result.model_correct,
                        "mean_tts": tts_result.mean_tts,
                        "n_steps": len(tts_result.step_scores),
                        "frac_decorative": tts_result.fraction_decorative,
                        "step_texts": step_texts,
                    })

                except Exception as e:
                    logger.warning(f"TTS failed: {e}")
                    rollouts.append({
                        "problem": prob,
                        "tokens": tokens,
                        "logprobs": logprobs,
                        "text": text,
                        "reward": 0.0,
                        "correct": False,
                        "mean_tts": 0.0,
                        "n_steps": 0,
                        "frac_decorative": 1.0,
                    })

            # Compute advantages (group-relative)
            rewards = [r["reward"] for r in rollouts]
            mean_reward = np.mean(rewards) if rewards else 0.0
            std_reward = np.std(rewards) if rewards else 1.0
            
            # Log group reward variance (critical for GRPO learning signal)
            logger.info(
                f"Group stats: n={len(rewards)}, mean={mean_reward:.4f}, "
                f"std={std_reward:.4f}, min={min(rewards) if rewards else 0:.4f}, "
                f"max={max(rewards) if rewards else 0:.4f}"
            )
            if std_reward < 0.05:
                logger.warning(
                    f"LOW VARIANCE: group_std={std_reward:.4f} < 0.05 — "
                    "GRPO cannot learn with near-zero advantage. "
                    "Try harder problems or higher temperature."
                )
            
            if std_reward < 1e-6:
                std_reward = 1.0

            for i, r in enumerate(rollouts):
                r["advantage"] = (r["reward"] - mean_reward) / std_reward

            # Build training data using proper TensorData format
            data = []
            for rollout in rollouts:
                tokens = rollout["tokens"]
                logprobs = rollout["logprobs"]
                advantage = rollout["advantage"]

                if len(tokens) < 2 or len(logprobs) < len(tokens):
                    logger.warning(f"Skipping rollout: {len(tokens)} tokens, {len(logprobs)} logprobs")
                    continue

                try:
                    # Build full sequence as ModelInput chunks
                    full_chunks = [tinker.EncodedTextChunk(tokens=tokens)]
                    full_input = tinker.ModelInput(chunks=full_chunks)

                    # Create right-shifted inputs and left-shifted targets
                    input_model_input, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
                        full_chunks
                    )

                    # Align logprobs: we need logprobs for each target token position
                    # logprobs[i] corresponds to token[i+1], so logprobs[0] is for token[1]
                    # target_tokens has length len(tokens)-1, aligned with input_model_input
                    aligned_logprobs = logprobs[:len(target_tokens)]

                    # Pad or truncate to match target_tokens length
                    if len(aligned_logprobs) < len(target_tokens):
                        aligned_logprobs.extend([0.0] * (len(target_tokens) - len(aligned_logprobs)))
                    elif len(aligned_logprobs) > len(target_tokens):
                        aligned_logprobs = aligned_logprobs[:len(target_tokens)]

                    # Mask: 0 for prompt tokens, 1 for completion tokens
                    # The prompt is everything before the assistant's response
                    # For simplicity, we weight all tokens equally in the completion
                    # (since we're doing full-sequence training on the model's own output)
                    mask = [1.0] * len(target_tokens)
                    advantages = [advantage] * len(target_tokens)

                    datum = tinker.Datum(
                        model_input=input_model_input,
                        loss_fn_inputs={
                            "target_tokens": TensorData(
                                data=target_tokens,
                                dtype="int64",
                                shape=[len(target_tokens)],
                            ),
                            "logprobs": TensorData(
                                data=aligned_logprobs,
                                dtype="float32",
                                shape=[len(aligned_logprobs)],
                            ),
                            "advantages": TensorData(
                                data=advantages,
                                dtype="float32",
                                shape=[len(advantages)],
                            ),
                            "mask": TensorData(
                                data=mask,
                                dtype="float32",
                                shape=[len(mask)],
                            ),
                        },
                    )
                    data.append(datum)

                except Exception as e:
                    logger.warning(f"Failed to build datum: {e}")

            # Update model
            if data:
                logger.info(f"Updating model with {len(data)} examples...")
                try:
                    # Remove mask before forward_backward (as cookbook does)
                    fwd_result = await training_client.forward_backward_async(
                        [_remove_mask(d) for d in data],
                        loss_fn="importance_sampling",
                    )
                    await fwd_result.result_async()

                    optim_result = await training_client.optim_step_async(
                        tinker.AdamParams(learning_rate=learning_rate)
                    )
                    await optim_result.result_async()
                    logger.info("Model updated successfully")
                except Exception as e:
                    logger.error(f"Training step failed: {e}")
                    # Don't poison the client - just log and continue
                    # (the next iteration will create a fresh sampling client)

            # Log metrics
            mean_tts = np.mean([r["mean_tts"] for r in rollouts])
            accuracy = np.mean([1.0 if r["correct"] else 0.0 for r in rollouts])
            mean_steps = np.mean([r["n_steps"] for r in rollouts])
            mean_decorative = np.mean([r["frac_decorative"] for r in rollouts])

            metrics = {
                "step": step + 1,
                "mean_reward": round(float(mean_reward), 4),
                "accuracy": round(float(accuracy), 4),
                "mean_tts": round(float(mean_tts), 4),
                "mean_steps": round(float(mean_steps), 1),
                "mean_decorative": round(float(mean_decorative), 4),
                "timestamp": datetime.now().isoformat(),
            }

            append_jsonl(run_dir / "metrics.jsonl", metrics)

            log_line = (
                f"step {step + 1}: reward={mean_reward:.4f}, acc={accuracy:.2f}, "
                f"tts={mean_tts:.4f}, steps={mean_steps:.1f}, decorative={mean_decorative:.2f}"
            )
            status["logs"].append(log_line)
            status["current_step"] = step + 1
            status["latest_metrics"] = metrics
            write_json(run_dir / "status.json", status)

            logger.info(log_line)

        # Mark complete
        status["status"] = "completed"
        status["completed_at"] = datetime.now().isoformat()
        write_json(run_dir / "status.json", status)

        # Save final checkpoints
        logger.info("Saving final checkpoint...")
        sampler_future = await training_client.save_weights_for_sampler_async(
            name=f"{run_id}-final",
            ttl_seconds=7 * 24 * 3600,
        )
        sampler_result = await sampler_future.result_async()
        sampler_path = sampler_result.path
        logger.info(f"Sampler checkpoint saved: {sampler_path}")

        state_future = await training_client.save_state_async(
            name=f"{run_id}-state",
            ttl_seconds=7 * 24 * 3600,
        )
        state_result = await state_future.result_async()
        state_path = state_result.path
        logger.info(f"Training state saved: {state_path}")

        checkpoint_data = {
            "sampler_path": sampler_path,
            "state_path": state_path,
            "training_run_id": training_client.model_id,
        }
        write_json(run_dir / "checkpoint.json", checkpoint_data)

        # Post-training analysis
        logger.info("Running post-training analysis...")
        trained_sampling_client = await service_client.create_sampling_client_async(
            model_path=sampler_path
        )

        analyses = []
        for prob in problems:
            try:
                messages = [{"role": "user", "content": prob["question"]}]
                prompt = renderer.build_generation_prompt(messages)
                stop_seqs = renderer.get_stop_sequences()

                result = await trained_sampling_client.sample_async(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        stop=stop_seqs,
                        max_tokens=2048,
                        temperature=0.0,
                    ),
                )

                text = tokenizer.decode(result.sequences[0].tokens)
                think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
                thinking_text = think_match.group(1).strip() if think_match else text

                tts_result = await compute_tts_for_cot(
                    trained_sampling_client, renderer, prob["question"], prob["answer"], thinking_text, seed=42
                )

                result_summary = tts_result.summary()
                result_summary["id"] = prob["id"]
                analyses.append(result_summary)

                logger.info(
                    f"Post-train {prob['id']}: correct={tts_result.model_correct}, "
                    f"steps={len(tts_result.step_scores)}, tts={tts_result.mean_tts:.4f}"
                )

            except Exception as e:
                logger.error(f"Post-training analysis failed for {prob['id']}: {e}")

        improved_data = {
            "model": config.BASE_MODEL,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "analyses": analyses,
        }
        write_json(run_dir / "improved_analysis.json", improved_data)

        run_result = {
            "run_id": run_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "num_steps": num_steps,
            "final_accuracy": float(accuracy),
            "final_mean_tts": float(mean_tts),
            "final_mean_steps": float(mean_steps),
            "final_decorative": float(mean_decorative),
            "checkpoint": checkpoint_data,
        }
        write_json(run_dir / "run_result.json", run_result)

        logger.info(f"Training run {run_id} completed!")

    except Exception as e:
        logger.error(f"Training run {run_id} failed: {e}", exc_info=True)
        status["status"] = "failed"
        status["error"] = str(e)
        write_json(run_dir / "status.json", status)
        raise


if __name__ == "__main__":
    import json as json_mod

    with open(config.DATA_DIR / "examples.json") as f:
        all_problems = json_mod.load(f)

    asyncio.run(
        run_training(
            run_id=f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            problems=all_problems[:5],
            num_steps=5,
            group_size=4,
            lora_rank=8,
        )
    )
