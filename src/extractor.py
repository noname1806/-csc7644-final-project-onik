"""Privacy policy text extraction from a URL."""
from __future__ import annotations

import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

MAX_CHARS = 10_000
TIMEOUT = 15


class PolicyFetchError(Exception):
    pass


def fetch_policy_html(url: str) -> str:
    if not url:
        raise PolicyFetchError("No privacy policy URL provided.")
    try:
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise PolicyFetchError(f"Failed to fetch policy: {e}") from e
    return resp.text


def clean_policy_html(html: str, max_chars: int = MAX_CHARS) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe", "form"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = text.strip()

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[...truncated...]"
    return text


def extract_policy_text(url: Optional[str], max_chars: int = MAX_CHARS) -> str:
    if not url:
        raise PolicyFetchError("No privacy policy URL on file.")
    html = fetch_policy_html(url)
    return clean_policy_html(html, max_chars=max_chars)
