"""Play Store scraper wrapper."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class AppMetadata:
    app_id: str
    title: str
    developer: str
    content_rating: str
    permissions: List[str]
    privacy_policy_url: Optional[str]
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def fetch_app_metadata(app_id: str, country: str = "us", lang: str = "en") -> AppMetadata:
    """Fetch app metadata + permissions from the Play Store.

    Uses google-play-scraper. Permissions come from a separate endpoint.
    """
    from google_play_scraper import app, permissions

    info = app(app_id, lang=lang, country=country)
    perm_map = permissions(app_id, lang=lang, country=country)

    flat_perms: List[str] = []
    for group, items in perm_map.items():
        if isinstance(items, list):
            flat_perms.extend(items)
        elif isinstance(items, str):
            flat_perms.append(items)

    return AppMetadata(
        app_id=app_id,
        title=info.get("title", app_id),
        developer=info.get("developer", ""),
        content_rating=info.get("contentRating", ""),
        permissions=sorted(set(flat_perms)),
        privacy_policy_url=info.get("privacyPolicy"),
        description=info.get("description", "")[:500],
    )


def cache_metadata(meta: AppMetadata, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{meta.app_id}.json"
    out.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
    return out


def load_cached(app_id: str, cache_dir: Path) -> Optional[AppMetadata]:
    p = cache_dir / f"{app_id}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return AppMetadata(**data)
