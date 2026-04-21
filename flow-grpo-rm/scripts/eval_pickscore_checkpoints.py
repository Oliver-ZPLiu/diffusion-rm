#!/usr/bin/env python
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import json
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from torch.utils.data import Dataset, DataLoader

from flow_grpo.diffusers_patch.flux_pipeline_with_logprob_fast import pipeline_with_logprob as flux_pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob_fast import pipeline_with_logprob as sd3_pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt as flux_encode_prompt
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt as sd3_encode_prompt
from flow_grpo.pickscore_scorer import PickScoreScorer
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file


def is_sd3_model(model_path):
    """Check if the model path is for SD3"""
    sd3_identifiers = ["stable-diffusion-3", "sd3", "SD3"]
    return any(identifier.lower() in model_path.lower() for identifier in sd3_identifiers)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on PickScore")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoints directory")
    parser.add_argument("--base_model_path", type=str,
                        default="black-forest-labs/FLUX.1-dev",
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
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                        help="Guidance scale for generation")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Image resolution")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save results as JSON")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
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


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device, is_sd3=False):
    with torch.no_grad():
        if is_sd3:
            prompt_embeds, pooled_prompt_embeds = sd3_encode_prompt(
                text_encoders, tokenizers, prompt, max_sequence_length, device=device
            )
        else:
            prompt_embeds, pooled_prompt_embeds = flux_encode_prompt(
                text_encoders, tokenizers, prompt, max_sequence_length
            )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def main():
    args = parse_args()

    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    # Initialize accelerator
    accelerator = Accelerator()

    # Find available checkpoints
    checkpoints = find_available_checkpoints(
        args.checkpoint_dir, args.start_step, args.end_step, args.eval_interval
    )

    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        print(f"Start: {args.start_step}, End: {args.end_step}, Interval: {args.eval_interval}")
        return

    print(f"Found {len(checkpoints)} checkpoints to evaluate:")
    for step, path in checkpoints:
        print(f"  - checkpoint-{step}")
    print()

    # Load base pipeline
    print(f"Loading base pipeline: {args.base_model_path}...")
    is_sd3 = is_sd3_model(args.base_model_path)

    if is_sd3:
        from diffusers import StableDiffusion3Pipeline
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
    else:
        pipeline = FluxPipeline.from_pretrained(
            args.base_model_path,
            torch_dtype=torch_dtype,
            variant="fp16" if torch_dtype == torch.float16 else None,
        )
        pipeline = pipeline.to(accelerator.device)

    # Store model info for later use
    pipeline.is_sd3 = is_sd3

    # Load test prompts
    prompts_path = args.test_prompts_path
    if not os.path.isabs(prompts_path):
        prompts_path = os.path.join(os.path.dirname(__file__), "..", prompts_path)

    dataset = PromptDataset(prompts_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Loaded {len(dataset)} test prompts")
    print()

    # Initialize PickScore scorer
    print("Initializing PickScore scorer...")
    pickscore_scorer = PickScoreScorer(
        device=accelerator.device,
        dtype=torch.float32
    )

    # Pre-compute negative embeddings for SD3
    if is_sd3:
        sd3_text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
        sd3_tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
        neg_prompt_embeds, neg_pooled_prompt_embeds = sd3_encode_prompt(
            sd3_text_encoders, sd3_tokenizers, [""], 128, device=accelerator.device
        )
        neg_prompt_embeds_batch = neg_prompt_embeds.repeat(args.batch_size, 1, 1)
        neg_pooled_prompt_embeds_batch = neg_pooled_prompt_embeds.repeat(args.batch_size, 1)

    # Store results
    results = {}

    # Initialize PeftModel with first checkpoint (same pattern as training code)
    peft_initialized = False

    # Evaluate each checkpoint
    for step, checkpoint_path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Evaluating checkpoint-{step}")
        print(f"{'='*60}")

        # Load LoRA weights (PEFT format saved by train_sd3_fast_rm.py)
        lora_path = os.path.join(checkpoint_path, "lora")
        if os.path.exists(lora_path):
            if not peft_initialized:
                # First checkpoint: wrap transformer with PeftModel
                pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, lora_path)
                pipeline.transformer.set_adapter("default")
                peft_initialized = True
            else:
                # Subsequent checkpoints: replace adapter weights in-place
                adapter_file = os.path.join(lora_path, "adapter_model.safetensors")
                if not os.path.exists(adapter_file):
                    adapter_file = os.path.join(lora_path, "adapter_model.bin")
                state_dict = load_file(adapter_file) if adapter_file.endswith(".safetensors") else torch.load(adapter_file, map_location="cpu")
                set_peft_model_state_dict(pipeline.transformer, state_dict)
        else:
            print(f"Warning: LoRA path not found at {lora_path}, using base model")

        # Evaluate
        all_pickscores = []

        with accelerator.autocast(), torch.no_grad():
            for batch_idx, prompts in enumerate(dataloader):
                if accelerator.is_local_main_process:
                    print(f"  Batch {batch_idx + 1}/{len(dataloader)}", end="")

                # Get correct text encoders and tokenizers based on model type
                if is_sd3:
                    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
                    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
                else:
                    text_encoders = pipeline.text_encoder
                    tokenizers = pipeline.tokenizer

                # Compute text embeddings
                prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                    prompts,
                    text_encoders,
                    tokenizers,
                    max_sequence_length=128,
                    device=accelerator.device,
                    is_sd3=is_sd3
                )

                # Generate images using the appropriate pipeline function
                if is_sd3:
                    bs = prompt_embeds.shape[0]
                    neg_embeds = neg_prompt_embeds_batch[:bs]
                    neg_pooled_embeds = neg_pooled_prompt_embeds_batch[:bs]
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
                else:
                    images = flux_pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
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

                if accelerator.is_local_main_process:
                    print(f" - Mean PickScore: {np.mean(scores.cpu().tolist()):.4f}")

        # Gather results from all processes
        all_pickscores = accelerator.gather(torch.tensor(all_pickscores, device=accelerator.device)).cpu().numpy()

        mean_score = np.mean(all_pickscores)
        std_score = np.std(all_pickscores)

        results[step] = {
            "mean": float(mean_score),
            "std": float(std_score),
            "num_samples": len(all_pickscores)
        }

        print(f"\n  checkpoint-{step}: PickScore = {mean_score:.4f} (±{std_score:.4f})")

        # No need to unload — PeftModel stays, weights get replaced next iteration

    # Print summary
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
