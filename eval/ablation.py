"""Ablation: untrained base LLM vs. fine-tuned LoRA on the same eval set.

Answers the central question of the fine-tuning track: does our supervision
actually help, or is the base model already good enough?

Workflow:
  1. Run eval/run_eval.py on the base model (`--backend local-hf`).
  2. Run eval/run_eval.py on the fine-tuned adapter (`--backend finetuned`).
  3. This script reads both result JSONs and reports the lift.

Usage:
    # First, run both eval backends (training must already be done)
    python -m eval.run_eval --backend local-hf
    python -m eval.run_eval --backend finetuned

    # Then compare
    python -m eval.ablation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    base_path = RESULTS_DIR / "eval_local-hf.json"
    ft_path = RESULTS_DIR / "eval_finetuned.json"

    missing = [p for p in (base_path, ft_path) if not p.exists()]
    if missing:
        for p in missing:
            print(f"  missing: {p}", file=sys.stderr)
        print(
            "\nRun both evals first:\n"
            "  python -m eval.run_eval --backend local-hf\n"
            "  python -m eval.run_eval --backend finetuned",
            file=sys.stderr,
        )
        return 2

    b = load(base_path)
    f = load(ft_path)

    print("=" * 72)
    print("FINE-TUNING ABLATION — base LLM vs. LoRA-tuned same model")
    print("=" * 72)
    print(f"Apps:           {b['n_apps']}")
    print(f"Verdicts:       {b['n_instances']}")
    print()
    print(f"                       Base       Fine-tuned   delta")
    print(
        f"  Overall accuracy   {b['overall_accuracy']:>6.3f}      {f['overall_accuracy']:>6.3f}   "
        f"{f['overall_accuracy'] - b['overall_accuracy']:+.3f}"
    )
    print(
        f"  Macro-F1           {b['macro_f1']:>6.3f}      {f['macro_f1']:>6.3f}   "
        f"{f['macro_f1'] - b['macro_f1']:+.3f}"
    )

    print()
    print("Per-class F1:")
    print(f"  {'class':<10} {'Base':>10} {'Tuned':>10} {'delta':>10}")
    for cls in ("covered", "unclear", "mismatch"):
        bf = b["per_class"][cls]["f1"]
        ff = f["per_class"][cls]["f1"]
        print(f"  {cls:<10} {bf:>10.3f} {ff:>10.3f} {ff - bf:>+10.3f}")

    print()
    print("Per-category accuracy:")
    print(f"  {'category':<12} {'Base':>10} {'Tuned':>10} {'delta':>10}")
    for cat in b["per_category"]:
        ba = b["per_category"][cat]["accuracy"]
        fa = f["per_category"][cat]["accuracy"]
        print(f"  {cat:<12} {ba:>10.3f} {fa:>10.3f} {fa - ba:>+10.3f}")

    # Where did fine-tuning help / hurt?
    print()
    print("Per-app accuracy:")
    print(f"  {'app':<26} {'Base':>6} {'Tuned':>6} {'delta':>6}")
    b_per_app = {a["app_id"]: a for a in b["per_app"]}
    f_per_app = {a["app_id"]: a for a in f["per_app"]}
    for app_id in b_per_app:
        ba = b_per_app[app_id]["accuracy"]
        fa = f_per_app[app_id]["accuracy"]
        print(f"  {app_id:<26} {ba:>6.2f} {fa:>6.2f} {fa - ba:>+6.2f}")

    # Where the fine-tune fixes the base
    print()
    print("Examples fixed by fine-tuning (base wrong, tuned right):")
    fixed = []
    regressed = []
    for app_id, b_app in b_per_app.items():
        f_app = f_per_app[app_id]
        b_perms = {p["permission"]: p for p in b_app["per_permission"]}
        f_perms = {p["permission"]: p for p in f_app["per_permission"]}
        for perm, bp in b_perms.items():
            fp = f_perms[perm]
            if not bp["match"] and fp["match"]:
                fixed.append((app_id, perm, bp["true"], bp["pred"], fp["pred"]))
            elif bp["match"] and not fp["match"]:
                regressed.append((app_id, perm, bp["true"], bp["pred"], fp["pred"]))

    for app_id, perm, t, bp_pred, fp_pred in fixed[:20]:
        print(f"  {app_id} :: {perm}")
        print(f"    truth={t}  base={bp_pred}  tuned={fp_pred}")
    if not fixed:
        print("  (none)")

    print()
    print(f"Examples broken by fine-tuning (base right, tuned wrong): {len(regressed)}")
    for app_id, perm, t, bp_pred, fp_pred in regressed[:10]:
        print(f"  {app_id} :: {perm}")
        print(f"    truth={t}  base={bp_pred}  tuned={fp_pred}")

    out = RESULTS_DIR / "ablation.json"
    summary = {
        "base": {"accuracy": b["overall_accuracy"], "macro_f1": b["macro_f1"]},
        "finetuned": {"accuracy": f["overall_accuracy"], "macro_f1": f["macro_f1"]},
        "delta": {
            "accuracy": round(f["overall_accuracy"] - b["overall_accuracy"], 3),
            "macro_f1": round(f["macro_f1"] - b["macro_f1"], 3),
        },
        "fixed_count": len(fixed),
        "regressed_count": len(regressed),
    }
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
