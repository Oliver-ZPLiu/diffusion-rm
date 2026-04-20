#!/usr/bin/env python
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-GPU PickScore evaluation script for checkpoints.
Usage: accelerate launch --num_processes=8 scripts/eval_pickscore_checkpoints_multinode.py [args]
"""

import os
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import StableDiffusion3Pipeline
from torch.utils.data import Dataset, DataLoader
from torch.distributed.fsdp import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import ShardingStrategy

from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob_fast import pipeline_with_logprob as sd3_pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt as sd3_encode_prompt
from flow_grpo.pickscore_scorer import PickScoreScorer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on PickScore (Multi-GPU)")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoints directory")
    parser.add_argument("--base_model_path", type=str,
                        default="/models/stable-diffusion-3.5-medium",
                        help="Base model path for pipeline")
    parser.add_argument("--start_step", type=int, default=120,
                        help="Starting checkpoint step")
    parser.add_argument("--end_step", type=int, default=None,
                        help="Ending checkpoint step (None = last available)")
    parser.add_argument("--eval_interval", type=int, default=120,
                        help="Evaluation interval in steps")
    parser.add_argument("--test_prompts_path", type=str,
                        default="dataset/pickscore/test.txt",
                        help="Path to test prompts file")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per GPU for generation")
    parser.add_argument("--num_inference_steps", type=int, default=40,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=4.5,
                        help="Guidance scale for generation")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Image resolution")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save results as JSON")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type (float32, bfloat16, float16)")
    parser.add_argument("--noise_level", type=float, default=0,
                        help="Noise level for SDE sampling (0 for evaluation)")
    parser.add_argument("--sde_window_size", type=int, default=0,
                        help="SDE window size (0 for evaluation)")
    parser.add_argument("--sde_type", type=str, default="cps",
                        help="SDE type (cps, sde)")
    return parser.parse_args()


class PromptDataset(Dataset):
    def __init__(self, file_path):
        self.prompts = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.prompts.append(line)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def find_available_checkpoints(checkpoint_dir, start_step, end_step, interval):
    checkpoints = []
    if not os.path.exists(checkpoint_dir):
        return checkpoints

    for item in os.listdir(checkpoint_dir):
        if item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[1])
                if step >= start_step:
                    if end_step is None or step <= end_step:
                        if (step - start_step) % interval == 0:
                            checkpoints.append((step, os.path.join(checkpoint_dir, item)))
            except ValueError:
                continue

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def main():
    args = parse_args()

    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    # Initialize accelerator with multi-GPU support
    accelerator = Accelerator()

    # Only main process prints
    if accelerator.is_main_process:
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Process index: {accelerator.process_index}")

    # Find available checkpoints (only on main process)
    checkpoints = find_available_checkpoints(
        args.checkpoint_dir, args.start_step, args.end_step, args.eval_interval
    )

    if not checkpoints:
        if accelerator.is_main_process:
            print(f"No checkpoints found in {args.checkpoint_dir}")
        return

    if accelerator.is_main_process:
        print(f"Found {len(checkpoints)} checkpoints to evaluate:")
        for step, path in checkpoints:
            print(f"  - checkpoint-{step}")
        print()

    # Load base pipeline (only once, shared across processes)
    if accelerator.is_main_process:
        print(f"Loading base pipeline: {args.base_model_path}...")

    # Each process loads the pipeline to its device
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype,
    )
    pipeline.pipeline_id = args.base_model_path

    # Move all components to GPU (consistent with training eval phase)
    pipeline.vae.to(accelerator.device, dtype=torch_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=torch_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=torch_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=torch_dtype)
    pipeline.transformer.to(accelerator.device, dtype=torch_dtype)

    # Load test prompts
    prompts_path = args.test_prompts_path
    if not os.path.isabs(prompts_path):
        prompts_path = os.path.join(os.path.dirname(__file__), "..", prompts_path)

    dataset = PromptDataset(prompts_path)

    # Use DistributedSampler for multi-GPU
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
        drop_last=True
    )
    sampler.set_epoch(0)  # Required for DistributedSampler
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )

    if accelerator.is_main_process:
        print(f"Loaded {len(dataset)} test prompts")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batches per GPU: {len(dataloader)}")
        print()

    # Initialize PickScore scorer (each process has its own)
    pickscore_scorer = PickScoreScorer(
        device=accelerator.device,
        dtype=torch.float32
    )

    # Text encoders and tokenizers for SD3
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    # Pre-compute negative embeddings from empty string (consistent with training)
    neg_prompt_embeds, neg_pooled_prompt_embeds = sd3_encode_prompt(
        text_encoders, tokenizers, [""], 128, device=accelerator.device
    )
    neg_prompt_embeds_batch = neg_prompt_embeds.repeat(args.batch_size, 1, 1)
    neg_pooled_prompt_embeds_batch = neg_pooled_prompt_embeds.repeat(args.batch_size, 1)

    # Store results (only on main process)
    results = {}

    # Evaluate each checkpoint
    for step, checkpoint_path in checkpoints:
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"Evaluating checkpoint-{step}")
            print(f"{'='*60}")

        # Synchronize before loading LoRA
        accelerator.wait_for_everyone()

        # Load LoRA weights (all processes load simultaneously)
        lora_path = os.path.join(checkpoint_path, "lora")
        if os.path.exists(lora_path):
            pipeline.load_lora_weights(lora_path)
        elif accelerator.is_main_process:
            print(f"Warning: LoRA path not found at {lora_path}, using base model")

        # Synchronize after loading LoRA
        accelerator.wait_for_everyone()

        # Evaluate
        all_pickscores = []

        with torch.autocast("cuda", dtype=torch_dtype), torch.no_grad():
            for batch_idx, prompts in enumerate(dataloader):
                # Compute text embeddings
                prompt_embeds, pooled_prompt_embeds = sd3_encode_prompt(
                    text_encoders, tokenizers, prompts, 128, device=accelerator.device
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)

                # Slice negative embeddings to match actual batch size
                bs = prompt_embeds.shape[0]
                neg_embeds = neg_prompt_embeds_batch[:bs]
                neg_pooled_embeds = neg_pooled_prompt_embeds_batch[:bs]

                # Generate images
                images = sd3_pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=neg_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=neg_pooled_embeds,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    output_type="pt",
                    height=args.resolution,
                    width=args.resolution,
                    noise_level=args.noise_level,
                    sde_window_size=args.sde_window_size,
                    sde_type=args.sde_type,
                )[0]

                # Compute PickScore
                images = images.float()
                scores = pickscore_scorer(prompts, images)
                all_pickscores.extend(scores.cpu().tolist())

        # Gather results from all processes
        all_pickscores = accelerator.gather(torch.tensor(all_pickscores, device=accelerator.device)).cpu().numpy()

        # Only main process computes statistics
        if accelerator.is_main_process:
            mean_score = np.mean(all_pickscores)
            std_score = np.std(all_pickscores)

            results[step] = {
                "mean": float(mean_score),
                "std": float(std_score),
                "num_samples": len(all_pickscores)
            }

            print(f"\n  checkpoint-{step}: PickScore = {mean_score:.4f} (±{std_score:.4f})")

        # Synchronize before unloading LoRA
        accelerator.wait_for_everyone()

        # Unload LoRA
        if os.path.exists(lora_path):
            pipeline.unload_lora_weights()

    # Synchronize before printing summary
    accelerator.wait_for_everyone()

    # Print summary (only on main process)
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"{'Checkpoint':<20} {'PickScore':<15} {'Std':<10} {'Samples':<10}")
        print("-"*60)
        for step in sorted(results.keys()):
            r = results[step]
            print(f"checkpoint-{step:<7} {r['mean']:.4f} (±{r['std']:.4f})   {r['std']:.4f}     {r['num_samples']}")

        # Save results to JSON
        if args.output_json:
            output_path = args.output_json
            if not os.path.isabs(output_path):
                output_path = os.path.join(os.path.dirname(__file__), "..", output_path)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
