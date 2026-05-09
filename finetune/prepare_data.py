"""Build a supervised fine-tuning dataset from ground_truth.json.

The eval set has 10 apps with full permission/policy/verdict triples. That is
too small to fine-tune on directly, so we augment by subsetting which
permissions each training example asks about: each app yields the full example
plus N random-subset examples. We also do a leave-some-out split so the held-out
apps are never seen during training.

Output format: JSONL where each line is `{"messages": [...]}` in
chat-template form (system / user / assistant). This is the format SFTTrainer
expects when `dataset_text_field` is unset.

Usage:
    python -m finetune.prepare_data
    python -m finetune.prepare_data --augmentations 8 --holdout social_a kids_b
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402

GT_PATH = ROOT / "eval" / "ground_truth.json"
OUT_DIR = ROOT / "finetune" / "dataset"


def _verdict_record(app: dict, perms_subset: list[str]) -> dict:
    """Build the assistant target JSON for a chosen subset of permissions."""
    permissions_out = []
    for perm in perms_subset:
        v = app["ground_truth"][perm]
        # We don't have per-permission `policy_mention`/`reasoning` in the
        # ground truth file, so synthesize them from the policy text.
        # For "covered"/"unclear" we pick the longest policy sentence that
        # mentions a relevant keyword; for "mismatch" we emit "NOT MENTIONED".
        permissions_out.append(
            {
                "permission": perm,
                "data_access": _data_access_for(perm),
                "policy_mention": _policy_quote_for(perm, v, app["policy_text"]),
                "verdict": v,
                "reasoning": _reasoning_for(perm, v, app["policy_text"]),
            }
        )
    covered = sum(1 for p in permissions_out if p["verdict"] == "covered")
    unclear = sum(1 for p in permissions_out if p["verdict"] == "unclear")
    mismatch = sum(1 for p in permissions_out if p["verdict"] == "mismatch")
    summary = (
        f"{app['title']} requests {len(perms_subset)} permissions in this audit. "
        f"The privacy policy clearly covers {covered}, leaves {unclear} ambiguous, "
        f"and does not address {mismatch}."
    )
    return {"permissions": permissions_out, "summary": summary}


PERMISSION_GLOSSARY_LITE = {
    "ACCESS_FINE_LOCATION": "precise GPS location",
    "ACCESS_COARSE_LOCATION": "approximate (network-based) location",
    "ACCESS_BACKGROUND_LOCATION": "background location",
    "CAMERA": "device camera",
    "RECORD_AUDIO": "device microphone",
    "READ_CONTACTS": "the user's contact list",
    "READ_CALENDAR": "calendar events",
    "READ_PHONE_STATE": "phone identifiers and state",
    "READ_MEDIA_IMAGES": "photos and images on device",
    "READ_MEDIA_VIDEO": "videos on device",
    "READ_MEDIA_AUDIO": "audio files on device",
    "INTERNET": "network communication",
    "ACCESS_NETWORK_STATE": "network connectivity info",
    "ACCESS_WIFI_STATE": "Wi-Fi network info",
    "POST_NOTIFICATIONS": "show notifications",
    "BLUETOOTH": "Bluetooth pairing",
    "WAKE_LOCK": "keep device awake",
    "VIBRATE": "device vibration",
}


def _data_access_for(perm: str) -> str:
    short = perm.rsplit(".", 1)[-1]
    return PERMISSION_GLOSSARY_LITE.get(short, "(see Android docs)")


# Keywords to find a relevant span in the policy for the assistant target.
KEYWORDS = {
    "LOCATION": ["location", "GPS", "approximate", "precise"],
    "CAMERA": ["camera", "photo", "video"],
    "RECORD_AUDIO": ["microphone", "audio", "voice"],
    "CONTACTS": ["contact"],
    "CALENDAR": ["calendar"],
    "PHONE_STATE": ["phone number", "device identifier", "device information"],
    "MEDIA": ["photo", "video", "image", "media"],
    "INTERNET": ["network", "transmit", "share with"],
    "NETWORK": ["network"],
    "WIFI": ["wi-fi", "wifi"],
    "NOTIFICATIONS": ["notification"],
    "BLUETOOTH": ["bluetooth"],
}


def _policy_quote_for(perm: str, verdict: str, policy: str) -> str:
    if verdict == "mismatch":
        return "NOT MENTIONED"
    perm_up = perm.upper()
    for key, kws in KEYWORDS.items():
        if key in perm_up:
            for kw in kws:
                idx = policy.lower().find(kw.lower())
                if idx >= 0:
                    # Find sentence boundaries
                    start = policy.rfind(".", 0, idx) + 1
                    end = policy.find(".", idx)
                    if end == -1:
                        end = len(policy)
                    span = policy[start : end + 1].strip()
                    if span:
                        return span
    # Fallback: first sentence of policy
    return policy.split(".")[0].strip() + "."


def _reasoning_for(perm: str, verdict: str, policy: str) -> str:
    if verdict == "covered":
        return f"Policy explicitly discloses the data category accessed by {perm.rsplit('.', 1)[-1]}."
    if verdict == "unclear":
        return (
            f"Policy mentions related data only via generic or imprecise language for "
            f"{perm.rsplit('.', 1)[-1]}."
        )
    return f"Policy does not address the data category accessed by {perm.rsplit('.', 1)[-1]}."


def _to_messages(app: dict, perms_subset: list[str]) -> dict[str, Any]:
    target = _verdict_record(app, perms_subset)
    user_content = build_user_prompt(app["title"], perms_subset, app["policy_text"])
    assistant_content = json.dumps(target, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def build_examples(app: dict, n_augmentations: int, rng: random.Random) -> list[dict]:
    """Full-perm example + N random-subset examples per app."""
    out = [_to_messages(app, list(app["permissions"]))]
    perms = list(app["permissions"])
    if len(perms) <= 2:
        return out
    for _ in range(n_augmentations):
        k = rng.randint(max(2, len(perms) // 2), len(perms))
        subset = sorted(rng.sample(perms, k))
        out.append(_to_messages(app, subset))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--augmentations", type=int, default=6, help="random-subset examples per app")
    ap.add_argument(
        "--holdout",
        nargs="*",
        default=["weather_b", "kids_b"],
        help="app_id suffixes to hold out as validation",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    apps = gt["apps"]
    holdout_set = set(args.holdout)

    train_examples: list[dict] = []
    val_examples: list[dict] = []

    for app in apps:
        suffix = app["app_id"].rsplit(".", 1)[-1]
        examples = build_examples(app, args.augmentations, rng)
        if suffix in holdout_set:
            val_examples.extend(examples)
        else:
            train_examples.extend(examples)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "val.jsonl"

    train_path.write_text(
        "\n".join(json.dumps(ex, ensure_ascii=False) for ex in train_examples) + "\n",
        encoding="utf-8",
    )
    val_path.write_text(
        "\n".join(json.dumps(ex, ensure_ascii=False) for ex in val_examples) + "\n",
        encoding="utf-8",
    )

    print(f"train: {len(train_examples)} examples -> {train_path}")
    print(f"val:   {len(val_examples)} examples -> {val_path}")
    print(f"holdout app suffixes: {sorted(holdout_set)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
