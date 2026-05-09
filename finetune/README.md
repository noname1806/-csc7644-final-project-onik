# Fine-tuning track

This is the Option B extension of the project: fine-tune a small open-weight
LLM (LoRA) on the permission-to-policy auditing task and add it as a 5th
backend (`finetuned`) alongside `claude` / `llama` / `heuristic` /
`claude-cached`.

## Why this exists

The base project *uses* an LLM via prompting; this directory *trains* one.
That gives the project a real ML contribution — adapter weights produced from
hand-labeled supervision — and a fair head-to-head comparison: untuned base
model vs. fine-tuned same model vs. Claude.

## Pipeline

```
eval/ground_truth.json
        │
        ▼
prepare_data.py     ──▶  finetune/dataset/{train,val}.jsonl   (chat-format SFT data)
        │
        ▼
train.py            ──▶  finetune/checkpoints/final/          (LoRA adapter + tokenizer)
        │
        ▼
infer.py + analyzer.py finetuned backend  ──▶ AppReport
```

## Data: 56 train / 14 val

The eval set has 10 apps × ~5 perms = 51 verdicts — too small to fine-tune on
directly. `prepare_data.py` augments per-app by:
1. Emitting one full-permission training example per app.
2. Emitting `--augmentations N` random-subset examples (asks the model to
   audit only some of the app's permissions). This teaches the model that the
   audit list is parametric, not memorized.
3. Holding out 2 apps (`weather_b`, `kids_b` by default) entirely as
   validation — they never appear in any training subset.

The assistant target JSON is generated deterministically from the ground-truth
verdicts: covered/unclear → first relevant policy sentence as `policy_mention`;
mismatch → `"NOT MENTIONED"`. This keeps the supervision faithful to the
strict-quoting rule in `src/prompts.py`.

## Prerequisites

```bash
pip install -r finetune/requirements.txt
huggingface-cli login   # needed for gated Llama models
```

Llama 3.2 / 3.1 weights on HuggingFace are gated — you must accept Meta's
license on the model page once before downloading.

## Run

```bash
# 1) Build dataset
python -m finetune.prepare_data

# 2) Train (default: Llama 3.2 1B, ~5-10 min on a 6-8 GB GPU)
python -m finetune.train

# 2b) Or, the 8B model from the proposal — needs ~12-16 GB VRAM with 4-bit:
python -m finetune.train \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --load-in-4bit \
    --epochs 3

# 3) Use the fine-tuned adapter on Lyft (or any app)
python check.py --app-id me.lyft.android --backend finetuned

# 4) Evaluate on the held-out apps + full eval set
python -m eval.run_eval --backend finetuned
python -m eval.compare_backends   # extend RESULTS.md by hand
```

## Defaults and why

| Setting | Default | Reason |
|---|---|---|
| Base model | `meta-llama/Llama-3.2-1B-Instruct` | Fits on a 6-8 GB GPU at full precision; trains in ~10 min on 56 examples |
| LoRA rank | 16 | Standard for instruction-style fine-tuning |
| LoRA alpha | 32 | 2× rank, common ratio |
| LoRA dropout | 0.05 | Mild regularization given the small dataset |
| Target modules | all 7 linear projections | Captures both attention and MLP adaptation |
| Epochs | 4 | Higher than usual because the dataset is small |
| Batch size | 1 + 4-step grad accum | Effective batch 4; safe on small GPUs |
| LR | 2e-4 | Standard LoRA LR |
| Max seq len | 2048 | Plenty for ~5K-char prompts |

## Limits to be honest about

- **Tiny supervision set.** 56 training examples is small even for LoRA. The
  fine-tuned model is unlikely to beat Claude (1.000 F1) but should beat the
  heuristic (0.495 F1). The interesting comparison is **Llama 3.2 base vs.
  Llama 3.2 fine-tuned** — does our supervision actually help?
- **Synthetic policy quotes.** The assistant target's `policy_mention` is
  picked by a deterministic heuristic, not a human. A human-relabeled
  training set would be stronger.
- **No domain-shift test.** The held-out apps are from the same synthetic
  family as the training apps. Real Play Store apps are out-of-distribution.
  Adding 5-10 real apps to the eval set is the next step.

## Files

- `prepare_data.py` — ground truth → JSONL chat-template SFT data
- `train.py` — TRL SFTTrainer + PEFT LoRA
- `infer.py` — adapter-loading helper used by the `finetuned` backend
- `requirements.txt` — extra deps for fine-tuning
- `dataset/` — generated train/val JSONL (created by `prepare_data.py`)
- `checkpoints/` — saved LoRA adapters (created by `train.py`)
