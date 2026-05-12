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
from tinker_cookbook.tokenizer_utils import get_tokenizer

import config
from cache.store import append_jsonl, write_json
from rl.grading import grade_response_exact
from rl.reward import compute_reward
from tts.scorer import compute_tts_for_cot

logger = logging.getLogger(__name__)


async def run_training(
    run_id: str,
    problems: list[dict],
    num_steps: int = 20,
    group_size: int = 8,
    groups_per_batch: int = 2,
    learning_rate: float = 1e-5,
    lora_rank: int = 32,
    max_tokens: int = 2048,
    sampling_temperature: float = 0.8,
):
    """Run RL training loop with TTS-aware reward."""
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_data = {
        "run_id": run_id,
        "model": config.BASE_MODEL,
        "num_steps": num_steps,
        "group_size": group_size,
        "groups_per_batch": groups_per_batch,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "max_tokens": max_tokens,
        "sampling_temperature": sampling_temperature,
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

            # Select prompt groups for this step. Each prompt gets group_size
            # completions so advantages compare rollouts for the same problem.
            selected_problems = random.sample(problems, min(groups_per_batch, len(problems)))

            # Generate rollouts and compute rewards
            rollouts = []
            n_degenerate = 0
            for prob in selected_problems:
                messages = [{"role": "user", "content": prob["question"]}]
                prompt = renderer.build_generation_prompt(messages)
                stop_seqs = renderer.get_stop_sequences()

                result = await sampling_client.sample_async(
                    prompt=prompt,
                    num_samples=group_size,
                    sampling_params=tinker.SamplingParams(
                        stop=stop_seqs,
                        max_tokens=max_tokens,
                        temperature=sampling_temperature,
                    ),
                )

                group_rollouts = []
                for seq in result.sequences:
                    tokens = seq.tokens
                    logprobs = seq.logprobs  # From sampling, NOT compute_logprobs_async
                    text = tokenizer.decode(tokens)

                    # Compute exact correctness and TTS shaping reward.
                    try:
                        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
                        thinking_text = think_match.group(1).strip() if think_match else text

                        tts_result = await compute_tts_for_cot(
                            sampling_client, renderer, prob["question"], prob["answer"], thinking_text, seed=42
                        )

                        exact_correct = grade_response_exact(text, prob["answer"])
                        step_texts = [s.step_text for s in tts_result.step_scores]
                        reward = compute_reward(
                            answer_correct=exact_correct,
                            mean_tts=tts_result.mean_tts,
                            decorative_fraction=tts_result.fraction_decorative,
                            step_texts=step_texts,
                        )

                        group_rollouts.append({
                            "problem": prob,
                            "prompt": prompt,
                            "tokens": tokens,
                            "logprobs": logprobs,
                            "text": text,
                            "reward": reward,
                            "correct": exact_correct,
                            "mean_tts": tts_result.mean_tts,
                            "n_steps": len(tts_result.step_scores),
                            "frac_decorative": tts_result.fraction_decorative,
                            "step_texts": step_texts,
                        })

                    except Exception as e:
                        logger.warning(f"TTS failed: {e}")
                        group_rollouts.append({
                            "problem": prob,
                            "prompt": prompt,
                            "tokens": tokens,
                            "logprobs": logprobs,
                            "text": text,
                            "reward": -0.3,
                            "correct": False,
                            "mean_tts": 0.0,
                            "n_steps": 0,
                            "frac_decorative": 1.0,
                            "step_texts": [],
                        })

                group_rewards = [r["reward"] for r in group_rollouts]
                group_mean = float(np.mean(group_rewards)) if group_rewards else 0.0
                group_advantages = [r - group_mean for r in group_rewards]
                if group_advantages and all(abs(a) < 1e-6 for a in group_advantages):
                    n_degenerate += 1
                    logger.info(f"Skipping degenerate group for {prob['id']}: all rewards={group_mean:.4f}")
                    continue

                for rollout, advantage in zip(group_rollouts, group_advantages):
                    rollout["advantage"] = advantage
                    rollouts.append(rollout)

            # Log batch reward variance (critical for learning signal)
            rewards = [r["reward"] for r in rollouts]
            mean_reward = np.mean(rewards) if rewards else 0.0
            std_reward = np.std(rewards) if rewards else 1.0
            
            logger.info(
                f"Batch stats: n_rollouts={len(rewards)}, groups={len(selected_problems)}, "
                f"degenerate={n_degenerate}, mean={mean_reward:.4f}, std={std_reward:.4f}, "
                f"min={min(rewards) if rewards else 0:.4f}, max={max(rewards) if rewards else 0:.4f}"
            )
            if std_reward < 0.05:
                logger.warning(
                    f"LOW VARIANCE: batch_std={std_reward:.4f} < 0.05 — "
                    "GRPO cannot learn with near-zero advantage. "
                    "Try harder problems or higher temperature."
                )

            # Build training data using proper TensorData format
            data = []
            for rollout in rollouts:
                prompt = rollout["prompt"]
                tokens = rollout["tokens"]
                logprobs = list(rollout["logprobs"])
                advantage = rollout["advantage"]

                if len(tokens) < 2:
                    logger.warning(f"Skipping rollout: {len(tokens)} tokens, {len(logprobs)} logprobs")
                    continue

                try:
                    if len(logprobs) < len(tokens):
                        logprobs.extend([0.0] * (len(tokens) - len(logprobs)))
                    elif len(logprobs) > len(tokens):
                        logprobs = logprobs[:len(tokens)]

                    model_input = prompt.append(tinker.EncodedTextChunk(tokens=tokens[:-1]))
                    prompt_target_len = prompt.length - 1
                    target_tokens = [0] * prompt_target_len + tokens
                    padded_logprobs = [0.0] * prompt_target_len + logprobs
                    padded_advantages = [0.0] * prompt_target_len + [advantage] * len(tokens)

                    datum = tinker.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                            "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                            "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                        },
                    )
                    data.append(datum)

                except Exception as e:
                    logger.warning(f"Failed to build datum: {e}")

            # Update model
            if data:
                logger.info(f"Updating model with {len(data)} examples...")
                try:
                    fwd_result = await training_client.forward_backward_async(
                        data,
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
            mean_tts = np.mean([r["mean_tts"] for r in rollouts]) if rollouts else 0.0
            accuracy = np.mean([1.0 if r["correct"] else 0.0 for r in rollouts]) if rollouts else 0.0
            mean_steps = np.mean([r["n_steps"] for r in rollouts]) if rollouts else 0.0
            mean_decorative = np.mean([r["frac_decorative"] for r in rollouts]) if rollouts else 0.0

            metrics = {
                "step": step + 1,
                "mean_reward": round(float(mean_reward), 4),
                "accuracy": round(float(accuracy), 4),
                "mean_tts": round(float(mean_tts), 4),
                "mean_steps": round(float(mean_steps), 1),
                "mean_decorative": round(float(mean_decorative), 4),
                "n_degenerate_groups": n_degenerate,
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
                        max_tokens=max_tokens,
                        temperature=0.0,
                    ),
                )

                text = tokenizer.decode(result.sequences[0].tokens)
                think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
                thinking_text = think_match.group(1).strip() if think_match else text

                tts_result = await compute_tts_for_cot(
                    trained_sampling_client, renderer, prob["question"], prob["answer"], thinking_text, seed=42
                )

                exact_correct = grade_response_exact(text, prob["answer"])
                result_summary = tts_result.summary()
                result_summary["id"] = prob["id"]
                result_summary["early_exit_correct"] = tts_result.model_correct
                result_summary["model_correct"] = exact_correct
                analyses.append(result_summary)

                logger.info(
                    f"Post-train {prob['id']}: correct={exact_correct}, "
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
