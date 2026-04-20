#!/usr/bin/env python
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Download official Pick-a-Pic validation set and create test set excluding training prompts.
"""

import os
from datasets import load_dataset

# 1. 先下载官方数据集，查看划分
print("=" * 60)
print("Loading official Pick-a-Pic v1 dataset info...")
print("=" * 60)

# 加载整个数据集看看官方划分
dataset = load_dataset('yuvalkirstain/pickapic_v1', streaming=True)

print(f"\nAvailable splits: {dataset}")
for split_name, ds in dataset.items():
    print(f"  - {split_name}: {ds.num_rows} samples" if hasattr(ds, 'num_rows') else f"  - {split_name}: streaming")

# 2. 加载官方 train 和 validation
print("\n" + "=" * 60)
print("Loading official train split...")
print("=" * 60)

official_train = load_dataset('yuvalkirstain/pickapic_v1', split='train', streaming=True)
official_train_captions = []
for example in official_train:
    caption = example['caption'].strip()
    if caption and caption not in official_train_captions:
        official_train_captions.append(caption)

print(f"Unique captions in official train: {len(official_train_captions)}")

# 3. 加载官方 validation
print("\n" + "=" * 60)
print("Loading official validation split...")
print("=" * 60)

official_val = load_dataset('yuvalkirstain/pickapic_v1', split='validation', streaming=True)
official_val_captions = []
for example in official_val:
    caption = example['caption'].strip()
    if caption and caption not in official_val_captions:
        official_val_captions.append(caption)

print(f"Unique captions in official validation: {len(official_val_captions)}")

# 4. 加载现有的 train.txt
print("\n" + "=" * 60)
print("Loading existing train.txt...")
print("=" * 60)

train_file = os.path.join(os.path.dirname(__file__), 'train.txt')
with open(train_file, 'r', encoding='utf-8') as f:
    existing_train_prompts = set(line.strip() for line in f if line.strip())

print(f"Existing train.txt prompts: {len(existing_train_prompts)}")

# 5. 分析重叠情况
print("\n" + "=" * 60)
print("Analyzing overlaps...")
print("=" * 60)

# 官方 train 和官方 validation 是否有重叠
official_overlap = set(official_train_captions) & set(official_val_captions)
print(f"Official train ∩ official validation: {len(official_overlap)}")

# 现有 train.txt 与官方 train 的重叠
existing_vs_official_train = existing_train_prompts & set(official_train_captions)
print(f"Existing train.txt ∩ official train: {len(existing_vs_official_train)}")

# 现有 train.txt 与官方 validation 的重叠
existing_vs_official_val = existing_train_prompts & set(official_val_captions)
print(f"Existing train.txt ∩ official validation: {len(existing_vs_official_val)}")

# 6. 创建评测集（使用官方 validation，排除与现有训练集重叠的）
print("\n" + "=" * 60)
print("Creating evaluation set...")
print("=" * 60)

eval_captions = [c for c in official_val_captions if c not in existing_train_prompts]
print(f"Official validation captions (excluding train overlap): {len(eval_captions)}")

# 保存完整版本
output_file = os.path.join(os.path.dirname(__file__), 'official_validation.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    for caption in eval_captions:
        f.write(caption + '\n')
print(f"\nSaved {len(eval_captions)} prompts to {output_file}")

# 保存快速评测版本（2000条）
test_output_file = os.path.join(os.path.dirname(__file__), 'official_validation_test.txt')
test_size = min(2000, len(eval_captions))
with open(test_output_file, 'w', encoding='utf-8') as f:
    for caption in eval_captions[:test_size]:
        f.write(caption + '\n')
print(f"Saved first {test_size} prompts to {test_output_file}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Official validation: {len(official_val_captions)} unique captions")
print(f"After excluding existing train.txt overlap: {len(eval_captions)} captions")
