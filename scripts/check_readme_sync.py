"""Check bilingual README structure and shared technical content without dependencies.

This cannot judge semantic translation quality; that requires manual review.
"""

from collections import Counter
from pathlib import Path
import re
import sys


def visible(text):
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def fences(text):
    blocks = re.findall(r"^```([^\n]*)\n(.*?)^```\s*$", text, flags=re.M | re.S)
    return [
        (kind, re.sub(r"\[[^\]\n]*\]", "[]", body) if kind == "mermaid" else body)
        for kind, body in blocks
    ]


def table_shapes(text):
    groups = re.findall(r"(?:^\|.*\|\s*\n)+", text, flags=re.M)
    return [tuple(line.count("|") - 1 for line in group.splitlines() if line.strip()) for group in groups]


def links(text):
    targets = re.findall(r"!?\[[^\]\n]*\]\(([^)]+)\)", text)
    return Counter(target for target in targets if target not in {"README.md", "README.zh-CN.md"})


def parity_errors(english, chinese):
    en, zh = visible(english), visible(chinese)
    checks = {
        "heading levels and order": lambda t: re.findall(r"^(#+) ", t, flags=re.M),
        "explicit anchors": lambda t: re.findall(r'<a id="([^"]+)"></a>', t),
        "command blocks and diagram structure": fences,
        "table dimensions": table_shapes,
        "link targets": links,
        "inline code": lambda t: Counter(re.findall(r"(?<!`)`([^`\n]+)`(?!`)", t)),
        "numerical values": lambda t: Counter(re.findall(r"\d+(?:[.,]\d+)*", t)),
    }
    return [name for name, extract in checks.items() if extract(en) != extract(zh)]


def local_link_errors(text, root):
    errors = []
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', visible(text)))
    for target in re.findall(r"!?\[[^\]\n]*\]\(([^)]+)\)", visible(text)):
        if "://" in target or target.startswith("mailto:"):
            continue
        if target.startswith("#"):
            if target[1:] not in anchors:
                errors.append(f"missing anchor: {target}")
        elif not (root / target.split("#", 1)[0]).exists():
            errors.append(f"missing local target: {target}")
    return errors


def main():
    root = Path(__file__).resolve().parents[1]
    english = (root / "README.md").read_text(encoding="utf-8")
    chinese = (root / "README.zh-CN.md").read_text(encoding="utf-8")
    errors = [f"Bilingual mismatch: {item}" for item in parity_errors(english, chinese)]
    for name, text in [("README.md", english), ("README.zh-CN.md", chinese)]:
        errors.extend(f"{name}: {item}" for item in local_link_errors(text, root))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("README parity and local links passed. Review semantic translation equivalence manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
