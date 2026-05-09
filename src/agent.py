"""Agent loop for the privacy auditor.

Two backends supported with the same tool set:
  * `agent-claude` — Anthropic native tool use
  * `agent-llama` — Ollama native tool use (Llama 3.1 supports function calling)

The agent receives a goal ("audit this app") and orchestrates the work
itself: fetch metadata, fetch policy, search per permission, verify quote,
submit verdict, summarize. The classic single-shot prompt becomes a
multi-step loop.
"""
from __future__ import annotations

import json
from typing import Optional

from .agent_tools import (
    TOOL_EXECUTORS,
    AgentSession,
    claude_tool_schema,
    execute_tool,
    openai_tool_schema,
)
from .prompts import PERMISSION_GLOSSARY


def _parse_tool_calls_from_text(text: str) -> list[dict]:
    """Extract tool calls that the model emitted as JSON inside its text content.

    Smaller open-weight models (e.g. Llama 3.1 8B via Ollama) often understand
    the workflow but ignore the structured `tool_calls` API, dumping calls as
    JSON inside the message body. We rescue those.
    """
    valid_tools = set(TOOL_EXECUTORS.keys())
    out: list[dict] = []

    # Strip common wrappers like <tool_call>{...}</tool_call>
    cleaned = text.replace("<tool_call>", "").replace("</tool_call>", "")

    # Greedy bracket-match: find every top-level {...} in the string
    depth = 0
    start = -1
    blocks: list[str] = []
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                blocks.append(cleaned[start : i + 1])
                start = -1

    for block in blocks:
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or (obj.get("function") or {}).get("name")
        if name not in valid_tools:
            continue
        args = obj.get("parameters") or obj.get("arguments") or {}
        if isinstance(args, dict) and "arguments" in args and len(args) == 1:
            args = args["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        out.append({"name": name, "args": args})

    return out


AGENT_SYSTEM_PROMPT = """You are a privacy auditor agent. For each Android permission an app requests, you must determine whether the app's privacy policy adequately discloses the corresponding data collection.

Workflow (follow exactly):
1. Call `fetch_app_metadata` to get the permission list and policy URL.
2. Call `fetch_policy` to load the full privacy policy text.
3. For EACH requested permission, in order:
   a. Call `search_policy` with keywords matching that permission's data category.
   b. Optionally call `read_policy_section` for more context if a match is partial.
   c. Decide the verdict:
      - "covered": policy clearly states this data is collected, used, or accessed.
      - "unclear": policy mentions the category vaguely or only via generic catch-all (e.g. "device information").
      - "mismatch": policy does not mention this data category, or contradicts the permission.
   d. If covered/unclear: pick a verbatim quote from the policy and call `verify_quote` on it.
   e. Call `submit_verdict` with the result.
4. After EVERY permission has been audited, write a 2-3 sentence plain-language summary as your FINAL response (no more tool calls).

Strict rules:
- A "covered" or "unclear" verdict requires a verbatim quote that you have verified with `verify_quote`. The `policy_mention` field must be a substring that exists in the policy.
- For "mismatch", set `policy_mention` to "NOT MENTIONED".
- Call `submit_verdict` exactly once per permission. Do not skip any. Do not duplicate.
- Do not infer disclosure from the app's category, name, or behavior — only from the policy text returned by your tools.
"""


def _build_initial_user_msg(app_title: str, app_id: str, permissions: list[str]) -> str:
    glossary_lines = []
    for perm in permissions:
        gloss = PERMISSION_GLOSSARY.get(perm, "(see Android docs)")
        glossary_lines.append(f"  - {perm} -> {gloss}")
    glossary = "\n".join(glossary_lines)

    return (
        f"Audit the app '{app_title}' (Play Store ID: {app_id}).\n\n"
        f"It requests these {len(permissions)} permissions (with grounded data-access reference):\n"
        f"{glossary}\n\n"
        "Begin by calling fetch_app_metadata, then fetch_policy. "
        "Then audit each permission per the system prompt."
    )


# ----------------------------------------------------------------------------
# Claude (Anthropic native tool use)
# ----------------------------------------------------------------------------
def run_claude_agent(
    session: AgentSession,
    app_title: str,
    permissions: list[str],
    claude_model: str = "claude-sonnet-4-5",
    max_iterations: int = 60,
) -> None:
    import anthropic

    client = anthropic.Anthropic()
    tools = claude_tool_schema()
    messages = [
        {
            "role": "user",
            "content": _build_initial_user_msg(app_title, session.app_id, permissions),
        }
    ]

    for _ in range(max_iterations):
        resp = client.messages.create(
            model=claude_model,
            max_tokens=4096,
            system=AGENT_SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        # Append assistant turn (Claude requires exact echoing of content blocks).
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            session.summary = text.strip()
            return

        tool_results: list[dict] = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            result = execute_tool(session, block.name, block.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
        messages.append({"role": "user", "content": tool_results})

    session.summary = session.summary or "(agent hit max_iterations without final summary)"


# ----------------------------------------------------------------------------
# Llama (Ollama native tool use)
# ----------------------------------------------------------------------------
_LLAMA_AGENT_SYSTEM = AGENT_SYSTEM_PROMPT + """

Tool-calling protocol (STRICT — small open-weight models cannot use the
structured tool API reliably, so we use a manual JSON protocol):

Each of your turns must contain EITHER:
  (A) Exactly one tool call as a single JSON object on its own:
        {"name": "<tool_name>", "arguments": {...}}
      Nothing else in the turn. No prose, no explanation, no markdown fences.
  (B) Your final summary as plain text (only after every permission is verdicted).

After your tool call, the user will respond with the tool result. Then you
emit your next single tool call, and so on. Do NOT batch multiple tool calls.

Available tools:
"""


def _llama_tool_help() -> str:
    lines = []
    for t in openai_tool_schema():
        f = t["function"]
        params = f.get("parameters", {}).get("properties", {})
        sig = ", ".join(params.keys())
        lines.append(f"- {f['name']}({sig}): {f['description']}")
    return "\n".join(lines)


def run_llama_agent(
    session: AgentSession,
    app_title: str,
    permissions: list[str],
    llama_model: str = "llama3.1:8b",
    ollama_base: str = "http://localhost:11434",
    max_iterations: int = 200,
) -> None:
    """Manual JSON tool-calling loop for small open-weight models.

    Llama 3.1 8B doesn't reliably use Ollama's structured `tools` API. Instead
    we ask it to emit one JSON tool call per turn, parse it ourselves, and
    append the tool result as a user message — keeping the conversation as
    standard user/assistant turns the model handles well.
    """
    import requests

    base = ollama_base.replace("/v1", "")
    system = _LLAMA_AGENT_SYSTEM + _llama_tool_help()
    messages: list[dict] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": _build_initial_user_msg(app_title, session.app_id, permissions),
        },
    ]

    for _ in range(max_iterations):
        payload = {
            "model": llama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 16384,
                "num_predict": 1024,
            },
        }
        r = requests.post(f"{base}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        content = (r.json()["message"].get("content") or "").strip()
        messages.append({"role": "assistant", "content": content})

        text_calls = _parse_tool_calls_from_text(content)
        if not text_calls:
            session.summary = content
            return

        # Honor only the first tool call per turn (protocol says one at a time).
        # If the model still batched, we drop the extras and let a fresh turn
        # ask for them — keeps the conversation grounded.
        tc = text_calls[0]
        result = execute_tool(session, tc["name"], tc["args"])
        # Compact preview to keep context small; the model gets enough to react.
        result_str = json.dumps(result, ensure_ascii=False)
        if len(result_str) > 4000:
            result_str = result_str[:4000] + "…(truncated)"
        messages.append(
            {"role": "user", "content": f"Tool result:\n{result_str}\n\nEmit your next tool call (one JSON object) or your final summary."}
        )

    session.summary = session.summary or "(agent hit max_iterations without final summary)"


# ----------------------------------------------------------------------------
# Groq (OpenAI-compatible structured tool use, FREE TIER)
# ----------------------------------------------------------------------------
def run_groq_agent(
    session: AgentSession,
    app_title: str,
    permissions: list[str],
    groq_model: str = "llama-3.3-70b-versatile",
    groq_base: str = "https://api.groq.com/openai/v1",
    max_iterations: int = 80,
) -> None:
    import os

    import requests

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY env var not set. Get a free key at https://console.groq.com/keys"
        )

    tools = openai_tool_schema()
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_initial_user_msg(app_title, session.app_id, permissions),
        },
    ]

    import time

    for _ in range(max_iterations):
        payload = {
            "model": groq_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_tokens": 2048,
        }

        # Retry-with-backoff on 429 (free tier is 30 req/min for Llama 3.3 70B).
        for attempt in range(6):
            r = requests.post(
                f"{groq_base}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120,
            )
            if r.status_code != 429:
                break
            retry_after = r.headers.get("retry-after")
            try:
                wait = float(retry_after) if retry_after else (2 ** attempt)
            except ValueError:
                wait = 2 ** attempt
            time.sleep(min(wait + 0.5, 30))
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        # Small inter-turn pacing to keep RPM under the cap.
        time.sleep(2.2)
        messages.append(msg)

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            session.summary = (msg.get("content") or "").strip()
            return

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = dict(raw_args)
            result = execute_tool(session, name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    session.summary = session.summary or "(agent hit max_iterations without final summary)"


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def run_agent(
    backend: str,
    session: AgentSession,
    app_title: str,
    permissions: list[str],
    claude_model: Optional[str] = None,
    llama_model: Optional[str] = None,
    ollama_base: Optional[str] = None,
    groq_model: Optional[str] = None,
    groq_base: Optional[str] = None,
) -> None:
    if backend == "agent-claude":
        run_claude_agent(
            session,
            app_title,
            permissions,
            claude_model=claude_model or "claude-sonnet-4-5",
        )
    elif backend == "agent-groq":
        run_groq_agent(
            session,
            app_title,
            permissions,
            groq_model=groq_model or "llama-3.3-70b-versatile",
            groq_base=groq_base or "https://api.groq.com/openai/v1",
        )
    elif backend == "agent-llama":
        run_llama_agent(
            session,
            app_title,
            permissions,
            llama_model=llama_model or "llama3.1:8b",
            ollama_base=ollama_base or "http://localhost:11434",
        )
    else:
        raise ValueError(f"Unknown agent backend: {backend}")
