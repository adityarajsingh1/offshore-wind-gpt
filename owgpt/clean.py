"""
Cleaning raw text before it goes into the tokenizer.

Wikipedia plain text still has section headings like "== History ==" and
a tail of "References", "External links" etc. that are useless for us.
"""

import re
import unicodedata

# everything after one of these headings is dropped
JUNK_SECTIONS = {"see also", "references", "external links", "further reading", "notes", "sources", "bibliography"}

HEADING = re.compile(r"^(=+)\s*(.*?)\s*\1\s*$")


def clean_wikipedia(text):
    lines = []
    for line in text.splitlines():
        m = HEADING.match(line.strip())
        if m:
            title = m.group(2)
            if title.lower() in JUNK_SECTIONS:
                break
            # keep the heading as a normal line, it's useful context
            lines.append("")
            lines.append(title)
            continue
        lines.append(line)
    return clean_text("\n".join(lines))


def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ")
    # collapse runs of spaces and blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()
