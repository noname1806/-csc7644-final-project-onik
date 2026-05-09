"""LLM-based permission/policy analyzer.

Single-shot backends:
  * `claude`        - Anthropic API (Claude Sonnet) one-shot prompt
  * `groq`          - Groq cloud API (Llama 3.3 70B) one-shot prompt — FREE TIER
  * `llama`         - Local Llama 3.1 via Ollama, one-shot prompt
  * `heuristic`     - Offline keyword-grounded fallback
  * `claude-cached` - Replays saved Claude predictions for deterministic eval
  * `finetuned`     - Locally fine-tuned LoRA adapter (see finetune/)
  * `local-hf`      - Untrained HuggingFace base model (for fine-tune ablation)

Agent backends (LLM drives the audit via tool calls):
  * `agent-claude`  - Claude as an agent with native tool use
  * `agent-groq`    - Llama 3.3 70B as an agent via Groq — FREE TIER, recommended
  * `agent-llama`   - Llama 3.1 8B as an agent (local, manual JSON protocol)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .prompts import PERMISSION_GLOSSARY, SYSTEM_PROMPT, build_user_prompt
from .schema import AppReport, PermissionVerdict, Verdict


@dataclass
class AnalyzerConfig:
    backend: str = "auto"  # one of the values listed in the module docstring
    claude_model: str = "claude-sonnet-4-5"
    llama_model: str = "llama3.1:8b"
    ollama_base: str = "http://localhost:11434/v1"
    groq_model: str = "llama-3.3-70b-versatile"  # free-tier Llama 3.3 70B on Groq
    groq_base: str = "https://api.groq.com/openai/v1"
    max_tokens: int = 8192
    temperature: float = 0.0
    cache_path: Optional[str] = None  # used by claude-cached backend
    ft_adapter_path: str = "finetune/checkpoints/final"  # used by finetuned backend
    ft_base_model: Optional[str] = None  # override base model (defaults to adapter meta.json)


# ----------------------------------------------------------------------------
# Backend: Claude (Anthropic API)
# ----------------------------------------------------------------------------
def _call_claude(system_prompt: str, user_prompt: str, cfg: AnalyzerConfig) -> str:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=cfg.claude_model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    return "".join(parts)


# ----------------------------------------------------------------------------
# Backend: Groq (Llama 3.3 70B via OpenAI-compatible API, free tier)
# ----------------------------------------------------------------------------
def _call_groq(system_prompt: str, user_prompt: str, cfg: AnalyzerConfig) -> str:
    import time

    import requests

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY env var not set. Get a free key at https://console.groq.com/keys"
        )
    payload = {
        "model": cfg.groq_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for attempt in range(6):
        r = requests.post(
            f"{cfg.groq_base}/chat/completions",
            json=payload,
            headers=headers,
            timeout=120,
        )
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        retry_after = r.headers.get("retry-after")
        try:
            wait = float(retry_after) if retry_after else (2 ** attempt)
        except ValueError:
            wait = 2 ** attempt
        time.sleep(min(wait + 0.5, 30))
    r.raise_for_status()  # final attempt error
    return r.json()["choices"][0]["message"]["content"]


# ----------------------------------------------------------------------------
# Backend: fine-tuned local LoRA adapter
# ----------------------------------------------------------------------------
def _call_finetuned(system_prompt: str, user_prompt: str, cfg: AnalyzerConfig) -> str:
    from finetune.infer import generate

    return generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        adapter_path=cfg.ft_adapter_path,
        base_model=cfg.ft_base_model,
        max_new_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )


# ----------------------------------------------------------------------------
# Backend: Llama (Ollama OpenAI-compatible endpoint)
# ----------------------------------------------------------------------------
def _call_llama(system_prompt: str, user_prompt: str, cfg: AnalyzerConfig) -> str:
    """Use Ollama's native /api/chat endpoint.

    The OpenAI-compatible endpoint silently caps output around 4-8K tokens
    regardless of `max_tokens`, which breaks audits of apps with many
    permissions. The native API respects `options.num_predict` and
    `options.num_ctx` reliably.
    """
    import requests

    base = cfg.ollama_base.replace("/v1", "")  # native API root
    payload = {
        "model": cfg.llama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",  # force JSON output
        "options": {
            "temperature": cfg.temperature,
            "num_predict": cfg.max_tokens,
            "num_ctx": 16384,  # extend context to fit policy + full audit JSON
        },
    }
    # First call after model load can take several minutes on consumer hardware;
    # subsequent calls are usually <30 s.
    r = requests.post(f"{base}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


# ----------------------------------------------------------------------------
# Backend: heuristic fallback (deterministic, offline)
# ----------------------------------------------------------------------------
KEYWORD_RULES = {
    # permission_substring: list of policy keywords that count as disclosure.
    # Phrases use word-boundaries; multi-word phrases avoid casual word clashes
    # ("contact us" should NOT match the CONTACTS permission).
    "LOCATION": ["location", "gps", "geolocat", "geographic"],
    "CAMERA": ["camera", "take a photo", "take photos", "capture photo", "capture image"],
    "RECORD_AUDIO": ["microphone", "audio recording", "voice", "record audio"],
    "CONTACTS": ["your contacts", "contact list", "address book", "phone book"],
    "CALENDAR": ["calendar", "calendar event"],
    "SMS": ["sms", "text message"],
    "PHONE_STATE": ["phone number", "imei", "device identifier", "phone state"],
    "STORAGE": ["storage", "files on your device", "media file", "your files"],
    "MEDIA_IMAGES": ["photos", "images", "media"],
    "MEDIA_VIDEO": ["videos", "media"],
    "MEDIA_AUDIO": ["audio file", "media", "voice"],
    "BLUETOOTH": ["bluetooth"],
    "WIFI": ["wi-fi", "wifi", "wireless network"],
    "INTERNET": ["network", "transmit", "send to our server", "share with"],
    "ACCOUNTS": ["account", "google account"],
    "CALL_LOG": ["call history", "call log", "calls you make"],
    "ACTIVITY_RECOGNITION": ["fitness", "step count", "physical activity"],
    "BODY_SENSORS": ["heart rate", "biometric", "health sensor"],
    "POST_NOTIFICATIONS": ["push notification", "send notifications"],
    "BILLING": ["purchase", "billing", "payment", "in-app"],
    "AD_ID": ["advertising", "ad id", "admob", "ad network"],
}


def _heuristic_verdict(permission: str, policy_text: str) -> PermissionVerdict:
    text_l = policy_text.lower()
    matched_key = None
    matched_quote = "NOT MENTIONED"

    for key, keywords in KEYWORD_RULES.items():
        if key in permission.upper():
            matched_key = key
            for kw in keywords:
                idx = text_l.find(kw.lower())
                if idx >= 0:
                    start = max(0, idx - 60)
                    end = min(len(policy_text), idx + len(kw) + 60)
                    snippet = policy_text[start:end].replace("\n", " ").strip()
                    matched_quote = snippet
                    break
            if matched_quote != "NOT MENTIONED":
                break

    if matched_quote == "NOT MENTIONED":
        verdict = Verdict.MISMATCH
        reasoning = (
            f"Policy does not mention {matched_key.lower() if matched_key else 'this data category'}."
        )
    else:
        # Cheap proxy for "vague" disclosure: short total policy or generic terms only
        vague_terms = ["device information", "personal information", "usage data"]
        if any(t in text_l for t in vague_terms) and matched_key in {
            "PHONE_STATE",
            "ACCOUNTS",
            "ACTIVITY_RECOGNITION",
        }:
            verdict = Verdict.UNCLEAR
            reasoning = "Policy mentions related data only via generic catch-all language."
        else:
            verdict = Verdict.COVERED
            reasoning = "Policy explicitly references this data category."

    return PermissionVerdict(
        permission=permission,
        data_access=PERMISSION_GLOSSARY.get(permission, "(see Android docs)"),
        policy_mention=matched_quote,
        verdict=verdict,
        reasoning=reasoning,
    )


def _heuristic_report(app_title: str, permissions: list[str], policy_text: str) -> dict:
    verdicts = [_heuristic_verdict(p, policy_text) for p in permissions]
    covered = sum(1 for v in verdicts if v.verdict == Verdict.COVERED)
    unclear = sum(1 for v in verdicts if v.verdict == Verdict.UNCLEAR)
    mismatch = sum(1 for v in verdicts if v.verdict == Verdict.MISMATCH)
    summary = (
        f"This app requests {len(permissions)} permissions. The privacy policy "
        f"clearly covers {covered}, leaves {unclear} ambiguous, and does not "
        f"address {mismatch} at all."
    )
    return {
        "permissions": [v.model_dump() for v in verdicts],
        "summary": summary,
    }


# ----------------------------------------------------------------------------
# JSON parsing helpers
# ----------------------------------------------------------------------------
JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_model_json(raw: str) -> dict:
    raw = raw.strip()
    m = JSON_FENCE.search(raw)
    if m:
        raw = m.group(1)
    # Find first { ... last } to be tolerant of leading/trailing prose
    if not raw.startswith("{"):
        first = raw.find("{")
        last = raw.rfind("}")
        if first >= 0 and last > first:
            raw = raw[first : last + 1]
    return json.loads(raw)


# ----------------------------------------------------------------------------
# Agent backend dispatch
# ----------------------------------------------------------------------------
def _run_agent_backend(
    backend: str,
    app_title: str,
    app_id: str,
    permissions: list[str],
    policy_text: str,
    policy_url: Optional[str],
    cfg: AnalyzerConfig,
) -> AppReport:
    from .agent import run_agent
    from .agent_tools import AgentSession
    from .scraper import AppMetadata

    session = AgentSession(
        app_id=app_id,
        preloaded_metadata=AppMetadata(
            app_id=app_id,
            title=app_title,
            developer="",
            content_rating="",
            permissions=permissions,
            privacy_policy_url=policy_url or "",
        ),
        preloaded_policy=policy_text,
    )

    try:
        run_agent(
            backend=backend,
            session=session,
            app_title=app_title,
            permissions=permissions,
            claude_model=cfg.claude_model,
            llama_model=cfg.llama_model,
            ollama_base=cfg.ollama_base,
            groq_model=cfg.groq_model,
            groq_base=cfg.groq_base,
        )
    except Exception as e:  # noqa: BLE001
        return AppReport(
            app_id=app_id,
            app_title=app_title,
            model=backend,
            permissions=session.verdicts,
            summary=session.summary,
            policy_chars=len(policy_text),
            policy_url=policy_url,
            error=f"Agent run failed: {e}",
        )

    if backend == "agent-claude":
        model_label = f"agent-claude ({cfg.claude_model})"
    elif backend == "agent-groq":
        model_label = f"agent-groq ({cfg.groq_model})"
    else:
        model_label = f"agent-llama ({cfg.llama_model})"
    return AppReport(
        app_id=app_id,
        app_title=app_title,
        model=model_label,
        permissions=session.verdicts,
        summary=session.summary,
        policy_chars=len(policy_text),
        policy_url=policy_url,
    )


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def _select_backend(cfg: AnalyzerConfig) -> str:
    if cfg.backend != "auto":
        return cfg.backend
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    # Try ollama probe
    try:
        import requests

        r = requests.get(cfg.ollama_base.replace("/v1", "") + "/api/tags", timeout=2)
        if r.ok:
            return "llama"
    except Exception:
        pass
    return "heuristic"


def analyze(
    app_title: str,
    app_id: str,
    permissions: list[str],
    policy_text: str,
    cfg: Optional[AnalyzerConfig] = None,
    policy_url: Optional[str] = None,
) -> AppReport:
    cfg = cfg or AnalyzerConfig()
    backend = _select_backend(cfg)

    if not policy_text or len(policy_text.strip()) < 50:
        return AppReport(
            app_id=app_id,
            app_title=app_title,
            model=f"{backend} (skipped)",
            permissions=[],
            summary="",
            policy_chars=len(policy_text or ""),
            policy_url=policy_url,
            error="Policy text too short or missing — flagged rather than verdict-forced.",
        )

    if backend in ("agent-claude", "agent-groq", "agent-llama"):
        return _run_agent_backend(
            backend, app_title, app_id, permissions, policy_text, policy_url, cfg
        )

    if backend == "heuristic":
        result = _heuristic_report(app_title, permissions, policy_text)
        model_label = "heuristic-keyword"
    elif backend == "claude-cached":
        cache_file = cfg.cache_path or "eval/claude_predictions.json"
        cache = json.loads(Path(cache_file).read_text(encoding="utf-8"))
        if app_id not in cache:
            return AppReport(
                app_id=app_id, app_title=app_title, model="claude-cached",
                permissions=[], summary="", policy_chars=len(policy_text),
                policy_url=policy_url,
                error=f"No cached Claude prediction for {app_id} in {cache_file}",
            )
        result = cache[app_id]
        model_label = "claude-sonnet-4-5 (cached)"
    else:
        user_prompt = build_user_prompt(app_title, permissions, policy_text)
        if backend == "claude":
            raw = _call_claude(SYSTEM_PROMPT, user_prompt, cfg)
            model_label = cfg.claude_model
        elif backend == "groq":
            raw = _call_groq(SYSTEM_PROMPT, user_prompt, cfg)
            model_label = cfg.groq_model
        elif backend == "llama":
            raw = _call_llama(SYSTEM_PROMPT, user_prompt, cfg)
            model_label = cfg.llama_model
        elif backend == "finetuned":
            raw = _call_finetuned(SYSTEM_PROMPT, user_prompt, cfg)
            model_label = f"finetuned ({cfg.ft_adapter_path})"
        elif backend == "local-hf":
            from finetune.infer import generate as _hf_generate

            raw = _hf_generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                adapter_path=None,
                base_model=cfg.ft_base_model or "meta-llama/Llama-3.2-1B-Instruct",
                max_new_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )
            model_label = f"local-hf ({cfg.ft_base_model or 'meta-llama/Llama-3.2-1B-Instruct'})"
        else:
            raise ValueError(f"Unknown backend: {backend}")
        try:
            result = _parse_model_json(raw)
        except json.JSONDecodeError as e:
            return AppReport(
                app_id=app_id,
                app_title=app_title,
                model=model_label,
                permissions=[],
                summary="",
                policy_chars=len(policy_text),
                policy_url=policy_url,
                error=f"Model returned invalid JSON: {e}. Raw: {raw[:300]}",
            )

    perm_models = [PermissionVerdict(**p) for p in result.get("permissions", [])]
    return AppReport(
        app_id=app_id,
        app_title=app_title,
        model=model_label,
        permissions=perm_models,
        summary=result.get("summary", ""),
        policy_chars=len(policy_text),
        policy_url=policy_url,
    )
