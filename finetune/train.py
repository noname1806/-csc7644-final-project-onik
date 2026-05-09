"""LoRA fine-tuning for the permission-to-policy auditor.

Defaults to Llama 3.2 1B Instruct (small enough to fine-tune on a 6-8 GB
consumer GPU without 4-bit quantization). For the 8B model from the proposal,
pass --base-model meta-llama/Llama-3.1-8B-Instruct --load-in-4bit.

Usage:
    python -m finetune.train
    python -m finetune.train --base-model meta-llama/Llama-3.1-8B-Instruct --load-in-4bit
    python -m finetune.train --base-model meta-llama/Llama-3.2-3B-Instruct --epochs 5

Requirements (install separately, see finetune/requirements.txt):
    transformers>=4.45 peft>=0.13 trl>=0.11 datasets accelerate
    bitsandbytes>=0.43  # only needed for --load-in-4bit
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "finetune" / "dataset"
DEFAULT_OUT = ROOT / "finetune" / "checkpoints"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--train-file", type=Path, default=DATA_DIR / "train.jsonl")
    ap.add_argument("--val-file", type=Path, default=DATA_DIR / "val.jsonl")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    # Imports are deferred so `--help` works without the heavy stack installed.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    print(f"[setup] base_model={args.base_model}")
    print(f"[setup] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[setup] device={torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    print(f"[setup] loading model (4bit={args.load_in_4bit}) ...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    print(f"[data] train={args.train_file} val={args.val_file}")
    train_ds = load_dataset("json", data_files=str(args.train_file), split="train")
    val_ds = load_dataset("json", data_files=str(args.val_file), split="train") if args.val_file.exists() else None
    print(f"[data] train_examples={len(train_ds)} val_examples={len(val_ds) if val_ds else 0}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sft_cfg = SFTConfig(
        output_dir=str(args.out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=torch.cuda.is_available(),
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds is not None else "no",
        max_seq_length=args.max_seq_len,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_cfg,
    )

    print(f"[train] starting (epochs={args.epochs}, lr={args.lr}, lora_r={args.lora_r}) ...")
    trainer.train()

    final_dir = args.out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[done] adapter + tokenizer saved to {final_dir}")

    # Write a small metadata file the analyzer can read.
    (final_dir / "meta.json").write_text(
        '{"base_model": "' + args.base_model + '", "lora_r": ' + str(args.lora_r) + "}",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
