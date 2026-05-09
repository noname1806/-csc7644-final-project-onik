"""Run the analyzer over the ground-truth eval set and compute precision/recall/F1.

Usage:
    python -m eval.run_eval --backend heuristic
    python -m eval.run_eval --backend claude
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analyzer import AnalyzerConfig, analyze
from src.schema import Verdict

GT_PATH = Path(__file__).parent / "ground_truth.json"
RESULTS_DIR = Path(__file__).parent / "results"
CLASSES = [Verdict.COVERED.value, Verdict.UNCLEAR.value, Verdict.MISMATCH.value]


def precision_recall_f1(y_true: list[str], y_pred: list[str], cls: str) -> tuple[float, float, float, int, int, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1, tp, fp, fn


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict:
    cm = {a: {b: 0 for b in CLASSES} for a in CLASSES}
    for t, p in zip(y_true, y_pred):
        if t in cm and p in cm[t]:
            cm[t][p] += 1
    return cm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        default="heuristic",
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
    parser.add_argument("--limit", type=int, default=0, help="Limit number of apps (0=all)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    apps = gt["apps"]
    if args.limit:
        apps = apps[: args.limit]

    cfg = AnalyzerConfig(backend=args.backend)

    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    per_app: list[dict] = []
    per_category: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"true": [], "pred": []})
    edge_cases: list[dict] = []

    for entry in apps:
        report = analyze(
            app_title=entry["title"],
            app_id=entry["app_id"],
            permissions=entry["permissions"],
            policy_text=entry["policy_text"],
            cfg=cfg,
        )
        pred_map = {v.permission: v.verdict.value for v in report.permissions}
        y_t = []
        y_p = []
        per_perm = []
        for perm in entry["permissions"]:
            t = entry["ground_truth"][perm]
            p = pred_map.get(perm, "mismatch")
            y_t.append(t)
            y_p.append(p)
            per_perm.append({"permission": perm, "true": t, "pred": p, "match": t == p})
        y_true_all.extend(y_t)
        y_pred_all.extend(y_p)
        per_category[entry["category"]]["true"].extend(y_t)
        per_category[entry["category"]]["pred"].extend(y_p)
        accuracy = sum(1 for x in per_perm if x["match"]) / len(per_perm)
        per_app.append(
            {
                "app_id": entry["app_id"],
                "category": entry["category"],
                "title": entry["title"],
                "policy_chars": len(entry["policy_text"]),
                "n_permissions": len(entry["permissions"]),
                "accuracy": round(accuracy, 3),
                "per_permission": per_perm,
            }
        )

        # Edge-case probing
        if len(entry["policy_text"]) < 200:
            edge_cases.append({"app_id": entry["app_id"], "case": "short_policy", "chars": len(entry["policy_text"])})

    # Aggregate metrics
    metrics_by_class = {}
    for cls in CLASSES:
        precision, recall, f1, tp, fp, fn = precision_recall_f1(y_true_all, y_pred_all, cls)
        metrics_by_class[cls] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    macro_f1 = round(sum(m["f1"] for m in metrics_by_class.values()) / len(CLASSES), 3)
    overall_accuracy = round(
        sum(1 for t, p in zip(y_true_all, y_pred_all) if t == p) / len(y_true_all), 3
    )

    # Per-category breakdown
    cat_results = {}
    for cat, lists in per_category.items():
        cat_acc = sum(1 for t, p in zip(lists["true"], lists["pred"]) if t == p) / len(lists["true"])
        cat_results[cat] = {
            "n_instances": len(lists["true"]),
            "accuracy": round(cat_acc, 3),
        }

    cm = confusion_matrix(y_true_all, y_pred_all)

    report = {
        "backend": args.backend,
        "n_apps": len(apps),
        "n_instances": len(y_true_all),
        "overall_accuracy": overall_accuracy,
        "macro_f1": macro_f1,
        "per_class": metrics_by_class,
        "per_category": cat_results,
        "confusion_matrix": cm,
        "edge_cases": edge_cases,
        "per_app": per_app,
    }

    out_path = args.out or RESULTS_DIR / f"eval_{args.backend}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Pretty print summary
    print(f"\n{'=' * 68}")
    print(f"EVAL RESULTS — backend={args.backend}")
    print(f"{'=' * 68}")
    print(f"Apps evaluated:     {len(apps)}")
    print(f"Permission instances: {len(y_true_all)}")
    print(f"Overall accuracy:   {overall_accuracy:.3f}")
    print(f"Macro-F1:           {macro_f1:.3f}")
    print(f"\nPer-class metrics:")
    print(f"  {'class':<10} {'P':>6} {'R':>6} {'F1':>6}   TP/FP/FN")
    for cls, m in metrics_by_class.items():
        print(
            f"  {cls:<10} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}   "
            f"{m['tp']}/{m['fp']}/{m['fn']}"
        )
    print(f"\nPer-category accuracy:")
    for cat, m in cat_results.items():
        print(f"  {cat:<12} n={m['n_instances']:<3} acc={m['accuracy']:.3f}")
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    header = "             " + " ".join(f"{c:>10}" for c in CLASSES)
    print(header)
    for t in CLASSES:
        row = f"  {t:<10} " + " ".join(f"{cm[t][p]:>10}" for p in CLASSES)
        print(row)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
