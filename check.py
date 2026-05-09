"""CLI entry point for the Permission-to-Policy Checker.

Usage:
    python check.py --app-id com.whatsapp
    python check.py --app-id com.whatsapp --backend claude
    python check.py --app-id com.whatsapp --offline-policy data/samples/whatsapp_policy.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.analyzer import AnalyzerConfig, analyze
from src.extractor import PolicyFetchError, extract_policy_text
from src.scraper import (
    AppMetadata,
    cache_metadata,
    fetch_app_metadata,
    load_cached,
)

CACHE = Path("data/cache")


def _load_offline_metadata(path: Path) -> AppMetadata:
    data = json.loads(path.read_text(encoding="utf-8"))
    return AppMetadata(**data)


def main() -> int:
    p = argparse.ArgumentParser(description="Permission-to-Policy Checker")
    p.add_argument("--app-id", required=True, help="Play Store package id (e.g. com.whatsapp)")
    p.add_argument(
        "--backend",
        default="auto",
        choices=[
            "auto",
            "claude",
            "groq",
            "llama",
            "heuristic",
            "claude-cached",
            "finetuned",
            "local-hf",
            "agent-claude",
            "agent-groq",
            "agent-llama",
        ],
    )
    p.add_argument("--offline-meta", type=Path, help="Skip scraping; load metadata JSON from file")
    p.add_argument("--offline-policy", type=Path, help="Skip fetching; load policy text from file")
    p.add_argument("--out", type=Path, help="Write JSON report to file")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    print(f"[1/4] Loading metadata for {args.app_id} ...", file=sys.stderr)
    if args.offline_meta:
        meta = _load_offline_metadata(args.offline_meta)
    else:
        meta = None if args.no_cache else load_cached(args.app_id, CACHE)
        if meta is None:
            try:
                meta = fetch_app_metadata(args.app_id)
                cache_metadata(meta, CACHE)
            except Exception as e:
                print(f"  scrape failed: {e}", file=sys.stderr)
                return 2

    print(f"      title={meta.title!r} permissions={len(meta.permissions)}", file=sys.stderr)

    print("[2/4] Loading privacy policy text ...", file=sys.stderr)
    if args.offline_policy:
        policy_text = args.offline_policy.read_text(encoding="utf-8")
    else:
        try:
            policy_text = extract_policy_text(meta.privacy_policy_url)
        except PolicyFetchError as e:
            print(f"  policy fetch failed: {e}", file=sys.stderr)
            policy_text = ""

    print(f"      policy_chars={len(policy_text)}", file=sys.stderr)

    print(f"[3/4] Analyzing with backend={args.backend} ...", file=sys.stderr)
    cfg = AnalyzerConfig(backend=args.backend)
    report = analyze(
        app_title=meta.title,
        app_id=meta.app_id,
        permissions=meta.permissions,
        policy_text=policy_text,
        cfg=cfg,
        policy_url=meta.privacy_policy_url,
    )

    print("[4/4] Done. Verdict counts:", file=sys.stderr)
    print(
        f"      covered={report.covered_count} unclear={report.unclear_count} "
        f"mismatch={report.mismatch_count}",
        file=sys.stderr,
    )

    out_json = report.model_dump_json(indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out_json, encoding="utf-8")
        print(f"      wrote {args.out}", file=sys.stderr)
    else:
        print(out_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
