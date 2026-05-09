"""Inference helper for a fine-tuned LoRA adapter.

Used by the `finetuned` backend in src.analyzer. Loads the base model + adapter
once per process (cached in a module-level dict keyed by adapter path).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_CACHE: dict[str, tuple] = {}


def _resolve_base_model(adapter_dir: Path, override: Optional[str]) -> str:
    if override:
        return override
    meta = adapter_dir / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8")).get(
            "base_model", "meta-llama/Llama-3.2-1B-Instruct"
        )
    cfg = adapter_dir / "adapter_config.json"
    if cfg.exists():
        return json.loads(cfg.read_text(encoding="utf-8")).get(
            "base_model_name_or_path", "meta-llama/Llama-3.2-1B-Instruct"
        )
    return "meta-llama/Llama-3.2-1B-Instruct"


def _load(adapter_path: Optional[str], base_model: Optional[str]):
    """Load base model with optional LoRA adapter.

    `adapter_path=None` returns the untrained base model — useful for
    ablation experiments (base vs. fine-tuned on the same eval set).
    """
    cache_key = adapter_path or f"__base__:{base_model}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if adapter_path:
        adapter_dir = Path(adapter_path)
        base = _resolve_base_model(adapter_dir, base_model)
    else:
        base = base_model or "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    _CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def generate(
    system_prompt: str,
    user_prompt: str,
    adapter_path: Optional[str],
    base_model: Optional[str] = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
) -> str:
    """Run a single prompt through the model, return raw generated text.

    Pass `adapter_path=None` to use the untrained base model (ablation mode).
    """
    import torch

    model, tokenizer = _load(adapter_path, base_model)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = out[0][inputs.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
