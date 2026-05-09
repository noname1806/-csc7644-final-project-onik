"""Verify every Claude-quoted policy span actually appears verbatim in source policy text.

This is the hallucination-control check from the Module 5 plan.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GT = json.loads((ROOT / "ground_truth.json").read_text(encoding="utf-8"))
PRED = json.loads((ROOT / "claude_predictions.json").read_text(encoding="utf-8"))


def main() -> int:
    failures: list[dict] = []
    total = 0
    verified = 0

    for app in GT["apps"]:
        app_id = app["app_id"]
        policy = app["policy_text"]
        if app_id not in PRED:
            continue
        for v in PRED[app_id]["permissions"]:
            total += 1
            quote = v["policy_mention"]
            if quote == "NOT MENTIONED":
                verified += 1
                continue
            if quote in policy:
                verified += 1
            else:
                failures.append(
                    {
                        "app_id": app_id,
                        "permission": v["permission"],
                        "verdict": v["verdict"],
                        "quote": quote,
                    }
                )

    rate = verified / total if total else 0.0
    print(f"Quote-grounding verification")
    print(f"  total verdicts:        {total}")
    print(f"  verified (verbatim):   {verified}")
    print(f"  hallucinated quotes:   {len(failures)}")
    print(f"  pass rate:             {rate:.3f}")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f['app_id']} :: {f['permission']} :: '{f['quote'][:80]}'")
        return 1

    out = ROOT / "results" / "quote_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"total": total, "verified": verified, "pass_rate": round(rate, 4), "failures": failures},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
