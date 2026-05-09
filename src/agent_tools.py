"""Tools for the privacy-auditor agent.

The agent receives a goal (audit app X) and a fixed set of tools it can call
to make progress. Each tool is a thin wrapper around the existing pipeline
(scraper, extractor) plus two new capabilities — full-text policy search
and inline quote verification — that an agent needs but a single-shot
prompt does not.

In live mode the tools hit the network. In eval mode the session is
pre-loaded with metadata + policy text, so the agent calls behave the
same but read from cache. The model never sees the difference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .extractor import extract_policy_text as _fetch_policy_text
from .scraper import AppMetadata, fetch_app_metadata as _fetch_meta
from .schema import PermissionVerdict, Verdict


@dataclass
class AgentSession:
    """Shared state across one agent run."""

    app_id: str
    preloaded_metadata: Optional[AppMetadata] = None
    preloaded_policy: Optional[str] = None

    metadata: Optional[AppMetadata] = None
    policy_text: Optional[str] = None
    verdicts: List[PermissionVerdict] = field(default_factory=list)
    summary: str = ""
    tool_calls: int = 0
    trace: List[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Pre-populate state from preloaded data so verbatim-quote checks and
        # search_policy work even if the model skips fetch_app_metadata or
        # fetch_policy. The fetch tools still work and return the same data.
        if self.metadata is None and self.preloaded_metadata is not None:
            self.metadata = self.preloaded_metadata
        if self.policy_text is None and self.preloaded_policy is not None:
            self.policy_text = self.preloaded_policy


# ----------------------------------------------------------------------------
# Tool implementations
# ----------------------------------------------------------------------------
def tool_fetch_app_metadata(session: AgentSession, app_id: str) -> dict:
    if session.preloaded_metadata is not None:
        meta = session.preloaded_metadata
    else:
        meta = _fetch_meta(app_id)
    session.metadata = meta
    return {
        "app_id": meta.app_id,
        "title": meta.title,
        "developer": meta.developer,
        "content_rating": meta.content_rating,
        "permissions": meta.permissions,
        "privacy_policy_url": meta.privacy_policy_url,
    }


def tool_fetch_policy(session: AgentSession, url: str) -> dict:
    if session.preloaded_policy is not None:
        session.policy_text = session.preloaded_policy
    else:
        # Larger budget than the one-shot pipeline because the agent will
        # search rather than read it all into one prompt.
        session.policy_text = _fetch_policy_text(url, max_chars=50_000)
    return {
        "policy_chars": len(session.policy_text),
        "preview": session.policy_text[:500],
    }


def tool_search_policy(
    session: AgentSession, query: str, max_results: int = 3
) -> dict:
    if not session.policy_text:
        return {"error": "Call fetch_policy first."}

    text = session.policy_text
    text_lower = text.lower()
    keywords = [query.lower()]
    keywords.extend(w for w in query.lower().split() if len(w) > 3)

    results: List[dict] = []
    seen_buckets: set[int] = set()

    for kw in keywords:
        start = 0
        while len(results) < max_results:
            idx = text_lower.find(kw, start)
            if idx < 0:
                break
            sent_start = max(0, text.rfind(".", 0, idx) + 1)
            sent_end = text.find(".", idx + len(kw))
            sent_end = sent_end + 1 if sent_end >= 0 else min(len(text), idx + 300)
            bucket = sent_start // 100
            if bucket not in seen_buckets:
                seen_buckets.add(bucket)
                snippet = text[sent_start:sent_end].strip()
                if snippet:
                    results.append({"offset": sent_start, "text": snippet})
            start = idx + len(kw)

    return {"query": query, "matches": results[:max_results]}


def tool_read_policy_section(
    session: AgentSession, start_offset: int, length: int = 1000
) -> dict:
    if not session.policy_text:
        return {"error": "Call fetch_policy first."}
    text = session.policy_text
    end = min(len(text), start_offset + length)
    return {
        "start": start_offset,
        "end": end,
        "section": text[start_offset:end],
    }


def tool_verify_quote(session: AgentSession, quote: str) -> dict:
    if quote == "NOT MENTIONED":
        return {"verbatim": True}
    if not session.policy_text:
        return {"error": "Call fetch_policy first."}
    return {"verbatim": quote in session.policy_text}


def tool_submit_verdict(
    session: AgentSession,
    permission: str,
    data_access: str,
    policy_mention: str,
    verdict: str,
    reasoning: str,
) -> dict:
    try:
        v = Verdict(verdict.lower())
    except ValueError:
        return {
            "error": f"verdict must be one of: covered, unclear, mismatch (got: {verdict!r})"
        }

    if (
        policy_mention != "NOT MENTIONED"
        and session.policy_text
        and policy_mention not in session.policy_text
    ):
        return {
            "error": (
                "policy_mention quote is not verbatim in the policy. "
                "Use search_policy then verify_quote before submitting."
            )
        }

    pv = PermissionVerdict(
        permission=permission,
        data_access=data_access,
        policy_mention=policy_mention,
        verdict=v,
        reasoning=reasoning,
    )
    session.verdicts.append(pv)
    return {"recorded": True, "n_verdicts": len(session.verdicts)}


# ----------------------------------------------------------------------------
# Tool registry (single source of truth, transformed per-backend)
# ----------------------------------------------------------------------------
TOOL_DEFS: list[dict] = [
    {
        "name": "fetch_app_metadata",
        "description": (
            "Fetch the app's title, developer, requested permissions, and "
            "privacy policy URL from the Play Store. Always call this first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "app_id": {
                    "type": "string",
                    "description": "Play Store package ID, e.g. 'me.lyft.android'",
                }
            },
            "required": ["app_id"],
        },
    },
    {
        "name": "fetch_policy",
        "description": (
            "Fetch and load the full privacy policy text. Call after "
            "fetch_app_metadata. The text is held in session and made "
            "queryable via search_policy."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Privacy policy URL"}
            },
            "required": ["url"],
        },
    },
    {
        "name": "search_policy",
        "description": (
            "Search the loaded policy for terms relevant to a permission's "
            "data category. Returns up to 3 matching sentences with offsets. "
            "Use this rather than reading the entire policy."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "search terms, e.g. 'microphone audio voice'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "max spans (default 3)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_policy_section",
        "description": (
            "Read a contiguous section of the policy starting at a given "
            "offset. Use after search_policy when you need surrounding context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_offset": {"type": "integer"},
                "length": {
                    "type": "integer",
                    "description": "default 1000",
                },
            },
            "required": ["start_offset"],
        },
    },
    {
        "name": "verify_quote",
        "description": (
            "Check that a quote appears verbatim in the policy. Always call "
            "before submit_verdict for any covered/unclear verdict."
        ),
        "parameters": {
            "type": "object",
            "properties": {"quote": {"type": "string"}},
            "required": ["quote"],
        },
    },
    {
        "name": "submit_verdict",
        "description": (
            "Record one final verdict for one permission. Call exactly once "
            "per requested permission, after searching and verifying."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "permission": {
                    "type": "string",
                    "description": "Android permission identifier",
                },
                "data_access": {
                    "type": "string",
                    "description": "plain-language description of what the permission grants",
                },
                "policy_mention": {
                    "type": "string",
                    "description": "verbatim policy quote, or 'NOT MENTIONED'",
                },
                "verdict": {
                    "type": "string",
                    "enum": ["covered", "unclear", "mismatch"],
                },
                "reasoning": {
                    "type": "string",
                    "description": "one-sentence justification",
                },
            },
            "required": [
                "permission",
                "data_access",
                "policy_mention",
                "verdict",
                "reasoning",
            ],
        },
    },
]

TOOL_EXECUTORS = {
    "fetch_app_metadata": tool_fetch_app_metadata,
    "fetch_policy": tool_fetch_policy,
    "search_policy": tool_search_policy,
    "read_policy_section": tool_read_policy_section,
    "verify_quote": tool_verify_quote,
    "submit_verdict": tool_submit_verdict,
}


def execute_tool(session: AgentSession, name: str, args: dict) -> dict:
    session.tool_calls += 1
    fn = TOOL_EXECUTORS.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        result = fn(session, **args)
    except TypeError as e:
        result = {"error": f"bad arguments for {name}: {e}"}
    except Exception as e:  # noqa: BLE001
        result = {"error": f"tool {name} failed: {e}"}
    session.trace.append({"tool": name, "args": args, "result": _short(result)})
    return result


def _short(obj: Any, n: int = 200) -> Any:
    if isinstance(obj, str):
        return obj if len(obj) <= n else obj[:n] + "…"
    if isinstance(obj, dict):
        return {k: _short(v, n) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_short(x, n) for x in obj[:5]]
    return obj


# ----------------------------------------------------------------------------
# Per-backend schema transforms
# ----------------------------------------------------------------------------
def claude_tool_schema() -> list[dict]:
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in TOOL_DEFS
    ]


def openai_tool_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in TOOL_DEFS
    ]
