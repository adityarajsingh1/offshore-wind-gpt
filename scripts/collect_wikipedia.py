#!/usr/bin/env python3
"""
Download the plain text of every article in data/topics.txt from Wikipedia.

    python scripts/collect_wikipedia.py

Saves one .txt per article into data/raw/wikipedia/. Already downloaded
articles are skipped, so it's safe to re-run after adding topics.
Wikipedia text is CC BY-SA, see the README for attribution.
"""

import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
TOPICS = ROOT / "data" / "topics.txt"
OUT = ROOT / "data" / "raw" / "wikipedia"

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "offshore-wind-gpt/0.1 (hobby project; github.com/adityarajsingh1/offshore-wind-gpt)"}


def read_topics(path=TOPICS):
    topics = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            topics.append(line)
    return topics


def safe_name(title):
    return re.sub(r"[^\w\-]+", "_", title).strip("_") + ".txt"


def fetch(title):
    """Plain text of one article, following redirects. None if it doesn't exist."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
        "format": "json",
    }
    resp = requests.get(API, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    topics = read_topics()
    print(f"{len(topics)} topics")
    missing = []
    for title in topics:
        target = OUT / safe_name(title)
        if target.exists():
            continue
        try:
            text = fetch(title)
        except requests.RequestException as e:
            print(f"  ! {title}: {e}")
            missing.append(title)
            continue
        if not text:
            print(f"  ? no article called '{title}'")
            missing.append(title)
            continue
        target.write_text(text, encoding="utf-8")
        print(f"  ok {title} ({len(text):,} chars)")
        time.sleep(0.5)  # be nice to wikipedia

    if missing:
        print(f"\ncouldn't get {len(missing)} topics: {', '.join(missing)}")
    total = sum(f.stat().st_size for f in OUT.glob("*.txt"))
    print(f"\ncorpus is now {total / 1e6:.1f} MB in {OUT}")


if __name__ == "__main__":
    sys.exit(main())
