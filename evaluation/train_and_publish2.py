"""
Train a model (minimal SFT), save checkpoint, and publish it.

NOTE: This is a TOY EXAMPLE that trains for a few steps on dummy data
to verify the full workflow end-to-end. You should replace the training
data and training logic with your own implementation.

TODO:
  - Replace DEMO_CONVERSATIONS with your task-specific training data
  - Tune hyperparameters (learning rate, batch size, number of steps, LoRA rank)
  - Add validation / early stopping as needed

Usage:
    python evaluation/train_and_publish.py
    python evaluation/train_and_publish.py --num_steps 20
    python evaluation/train_and_publish.py --no_publish   # skip publishing
"""

import argparse
import json
import os
import random

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

from datasets import load_dataset
import datafile

#MODEL = "meta-llama/Llama-3.2-3B"
# MODEL = "meta-llama/Llama-3.2-1B"    # Smaller, faster for development
MODEL = "meta-llama/Llama-3.1-8B"    # Recommended for final submission

pre_weights = "tinker://a8b434f4-ebcc-58d3-9009-45404f75175b:train:0/sampler_weights/0420-8B"

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

NVIDIA_NUM = 10_000
OPENAI_NUM = 6_000
ALLENAI_NUM = 5_000
METAMATH_NUM = 13_000
WIZARD_NUM = 10_000

TOTAL_NUM = NVIDIA_NUM + OPENAI_NUM + ALLENAI_NUM + METAMATH_NUM

NVIDIA_CONVERSATIONS = datafile.sample_from_jsonl("evaluation/data2/nvidia50k.jsonl", NVIDIA_NUM)
OPENAI_CONVERSATIONS = datafile.sample_from_jsonl("evaluation/data2/openai.jsonl", OPENAI_NUM)
ALLENAI_CONVERSATIONS = datafile.sample_from_jsonl("evaluation/data2/allenai50k.jsonl", ALLENAI_NUM)

ALL_CONVERSATIONS = NVIDIA_CONVERSATIONS + OPENAI_CONVERSATIONS + ALLENAI_CONVERSATIONS

#optional metaMath
#WIZARD_CONVERSATIONS = datafile.sample_from_jsonl("evaluation/data2/wizard50k.jsonl", WIZARD_NUM)
METAMATH_CONVERSATIONS = datafile.sample_from_jsonl("evaluation/data2/metamath50k.jsonl", METAMATH_NUM)
ALL_CONVERSATIONS = ALL_CONVERSATIONS + METAMATH_CONVERSATIONS# + WIZARD_CONVERSATIONS


random.shuffle(ALL_CONVERSATIONS)


def main():
    parser = argparse.ArgumentParser(description="Train, save, and publish a checkpoint")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--checkpoint_name", type=str, default="demo", help="Checkpoint name")
    parser.add_argument("--no_publish", action="store_true", help="Skip publishing")
    args = parser.parse_args()

    # Setup
    print(f"Model: {MODEL}")
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # Prepare training data
    print("Preparing training data...")
    all_data = []
    for convo in ALL_CONVERSATIONS:
        datum = conversation_to_datum(
            convo, renderer, max_length=512, train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )
        all_data.append(datum)
    print(f"  {len(all_data)} training examples prepared")

    # Create training client
    print(f"Creating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=MODEL, rank=args.rank)
    print("  Training client ready")

    # Train
    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    print(f"\nTraining for {args.num_steps} steps (batch_size={args.batch_size}, lr={args.lr})...")

    for step in range(args.num_steps):
        # Cycle through data
        start = (step * args.batch_size) % len(all_data)
        batch = [all_data[i % len(all_data)] for i in range(start, start + args.batch_size)]

        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = tc.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        # Compute loss
        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
        weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
        loss = -np.dot(logprobs, weights) / max(weights.sum(), 1)
        print(f"  Step {step+1}/{args.num_steps} | Loss: {loss:.4f}")
        
        if (step + 1) % 750 == 0:
            periodic_ckpt_name = (
                f"{args.checkpoint_name}"
                f"_bs{args.batch_size}"
                f"_lr{args.lr}"
                f"_r{args.rank}"
                f"_ns{len(all_data)}"
                f"_step{step+1}"
            )

            periodic_ckpt = tc.save_weights_for_sampler(name=periodic_ckpt_name).result()
            periodic_ckpt_path = periodic_ckpt.path
            print(f"  [Checkpoint saved @ step {step+1}] -> {periodic_ckpt_path}")

            # Publish immediately
            if not args.no_publish:
                rest_client = sc.create_rest_client()
                rest_client.publish_checkpoint_from_tinker_path(periodic_ckpt_path).result()
                print(f"  [Checkpoint published @ step {step+1}]")

    # Save checkpoint
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Checkpoint saved: {checkpoint_path}")

    # Publish
    if not args.no_publish:
        print("\nPublishing checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print("  Published successfully!")
    else:
        print("\nSkipping publish (--no_publish).")

    # Save checkpoint info
    info = {
        "checkpoint_path": checkpoint_path,
        "base_model": MODEL,
        "renderer_name": renderer_name,
        "training": {
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "lora_rank": args.rank,
        },
        "published": not args.no_publish,
    }
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nCheckpoint info saved to {info_path}")
    print(f"\nNext: evaluate your checkpoint with")
    print(f"  python -m evaluation.eval_all --checkpoint_path \"{checkpoint_path}\" --base_model {MODEL}")


if __name__ == "__main__":
    random.seed(542)
    np.random.seed(542)
    main()