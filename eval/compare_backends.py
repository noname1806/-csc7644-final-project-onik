"""Side-by-side comparison of heuristic baseline vs Claude on the same eval set."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    h = load(RESULTS_DIR / "eval_heuristic.json")
    c = load(RESULTS_DIR / "eval_claude-cached.json")

    print("=" * 72)
    print("BACKEND COMPARISON — Heuristic baseline vs Claude Sonnet 4.5")
    print("=" * 72)
    print(f"Apps:            {h['n_apps']}")
    print(f"Verdicts/app:    {h['n_instances'] / h['n_apps']:.1f} avg")
    print(f"Total verdicts:  {h['n_instances']}")
    print()
    print(f"                    Heuristic    Claude")
    print(f"  Overall accuracy   {h['overall_accuracy']:>6.3f}     {c['overall_accuracy']:>6.3f}")
    print(f"  Macro-F1           {h['macro_f1']:>6.3f}     {c['macro_f1']:>6.3f}")
    print()
    print("Per-class F1:")
    print(f"  {'class':<10} {'Heuristic':>10} {'Claude':>10} {'delta':>8}")
    for cls in ("covered", "unclear", "mismatch"):
        hf = h["per_class"][cls]["f1"]
        cf = c["per_class"][cls]["f1"]
        print(f"  {cls:<10} {hf:>10.3f} {cf:>10.3f} {cf - hf:>+8.3f}")
    print()
    print("Per-category accuracy:")
    print(f"  {'category':<12} {'Heuristic':>10} {'Claude':>10}")
    for cat in h["per_category"]:
        ha = h["per_category"][cat]["accuracy"]
        ca = c["per_category"][cat]["accuracy"]
        print(f"  {cat:<12} {ha:>10.3f} {ca:>10.3f}")

    # Where does the heuristic fail?
    print()
    print("Per-app accuracy gap (Claude - Heuristic):")
    for h_app, c_app in zip(h["per_app"], c["per_app"]):
        gap = c_app["accuracy"] - h_app["accuracy"]
        print(
            f"  {h_app['app_id']:<26} {h_app['accuracy']:>5.2f} -> {c_app['accuracy']:>5.2f}   gap={gap:+.2f}"
        )

    # Disagreement details
    print()
    print("Specific disagreements (heuristic wrong, Claude right):")
    h_per_app = {a["app_id"]: a for a in h["per_app"]}
    c_per_app = {a["app_id"]: a for a in c["per_app"]}
    for app_id in h_per_app:
        h_perms = {p["permission"]: p for p in h_per_app[app_id]["per_permission"]}
        c_perms = {p["permission"]: p for p in c_per_app[app_id]["per_permission"]}
        for perm, hp in h_perms.items():
            cp = c_perms[perm]
            if not hp["match"] and cp["match"]:
                print(f"  {app_id} :: {perm}")
                print(f"    truth={hp['true']}  heuristic={hp['pred']}  claude={cp['pred']}")

    out = RESULTS_DIR / "comparison.json"
    summary = {
        "heuristic": {"accuracy": h["overall_accuracy"], "macro_f1": h["macro_f1"]},
        "claude": {"accuracy": c["overall_accuracy"], "macro_f1": c["macro_f1"]},
        "delta": {
            "accuracy": round(c["overall_accuracy"] - h["overall_accuracy"], 3),
            "macro_f1": round(c["macro_f1"] - h["macro_f1"], 3),
        },
    }
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
