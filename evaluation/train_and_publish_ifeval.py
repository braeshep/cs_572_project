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
import datafile

MODEL = "meta-llama/Llama-3.1-8B"

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------
# DATA SPLITS
# ---------------------------

def load_pretrain_data():
    NVIDIA_NUM = 10_000
    OPENAI_NUM = 6_000
    ALLENAI_NUM = 5_000
    METAMATH_NUM = 13_000
    WIZARD_NUM = 10_000

    nvidia = datafile.sample_from_jsonl("evaluation/data2/nvidia50k.jsonl", NVIDIA_NUM)
    openai = datafile.sample_from_jsonl("evaluation/data2/openai.jsonl", OPENAI_NUM)
    allenai = datafile.sample_from_jsonl("evaluation/data2/allenai50k.jsonl", ALLENAI_NUM)
    metamath = datafile.sample_from_jsonl("evaluation/data2/metamath50k.jsonl", METAMATH_NUM)
    wizard = datafile.sample_from_jsonl("evaluation/data2/wizard50k.jsonl", WIZARD_NUM)

    all_data = nvidia + openai + allenai + metamath + wizard
    random.shuffle(all_data)
    return all_data


def load_ifeval_data():
    # round 2 specialization split
    nvidia = datafile.sample_from_jsonl("evaluation/data2/nvidia50k.jsonl", 1500)   # code
    allenai = datafile.sample_from_jsonl("evaluation/data2/allenai50k.jsonl", 6000) # IF tasks
    metamath = datafile.sample_from_jsonl("evaluation/data2/metamath50k.jsonl", 1500) # math

    all_data = nvidia + allenai + metamath
    random.shuffle(all_data)
    return all_data


# ---------------------------
# TRAIN LOOP
# ---------------------------

def train_stage(
    name,
    tc,
    data,
    renderer,
    steps,
    batch_size,
    lr,
    save_every=100,
):
    adam_params = types.AdamParams(
        learning_rate=lr,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8
    )

    all_data = [
        conversation_to_datum(
            convo,
            renderer,
            max_length=512,
            train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )
        for convo in data
    ]

    print(f"\n=== {name} training ===")
    print(f"steps={steps}, batch_size={batch_size}, lr={lr}")

    for step in range(steps):
        if (step + 1) % 50 == 0:
            print(f"Training Step: {step}")
        start = (step * batch_size) % len(all_data)
        batch = [all_data[i % len(all_data)] for i in range(start, start + batch_size)]

        fwd = tc.forward_backward(batch, loss_fn="cross_entropy")
        opt = tc.optim_step(adam_params)

        fwd.result()
        opt.result()

        if (step + 1) % save_every == 0:
            ckpt_name = f"{name}_step{step+1}"
            ckpt = tc.save_weights_for_sampler(name=ckpt_name).result()
            print(f"[{name}] checkpoint saved: {ckpt.path}")

            rest = tinker.ServiceClient().create_rest_client()
            rest.publish_checkpoint_from_tinker_path(ckpt.path).result()

    final_ckpt = tc.save_weights_for_sampler(name=f"{name}_final").result()
    print(f"[{name}] FINAL checkpoint: {final_ckpt.path}")

    return final_ckpt.path


# ---------------------------
# MAIN
# ---------------------------

def main():
    sc = tinker.ServiceClient()

    tokenizer = get_tokenizer(MODEL)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(MODEL),
        tokenizer
    )

    # create model
    tc = sc.create_lora_training_client(base_model=MODEL, rank=64)

    # ---------------------------
    # STAGE 1: PRE-IFEVAL
    # ---------------------------
    pre_data = load_pretrain_data()

    pre_ckpt = train_stage(
        name="pre_ifeval_focus",
        tc=tc,
        data=pre_data,
        renderer=renderer,
        steps=5500,
        batch_size=16,
        lr=3e-5,
        save_every=500,   # less frequent for long run
    )

    # save snapshot marker
    with open(os.path.join(EVAL_DIR, "pre_ifeval_ckpt.txt"), "w") as f:
        f.write(pre_ckpt)

    # ---------------------------
    # IMPORTANT:
    # DO NOT reload optimizer or state
    # just continue SAME tc object
    # ---------------------------

    # ---------------------------
    # STAGE 2: IFEVAL FOCUS
    # ---------------------------
    ifeval_data = load_ifeval_data()

    post_ckpt = train_stage(
        name="post_ifeval_focus",
        tc=tc,
        data=ifeval_data,
        renderer=renderer,
        steps= 1500,
        batch_size=8,
        lr=3e-6,
        save_every=100,
    )

    with open(os.path.join(EVAL_DIR, "post_ifeval_ckpt.txt"), "w") as f:
        f.write(post_ckpt)

    print("\nDONE")
    print("Pre-IFEval:", pre_ckpt)
    print("Post-IFEval:", post_ckpt)


if __name__ == "__main__":
    random.seed()
    np.random.seed()
    main()