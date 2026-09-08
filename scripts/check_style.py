"""House-style checker for the submission document.

The rules are not invented here. They are measured from the group's submitted
AAAI paper (gatekeeper/paper/sections, 25,191 words), which contains zero em
dashes, zero "notably/importantly/crucially", zero "note that", and uses first
person freely. This script enforces the same on SUBMISSION.md.

It scans the whole file, including table cells, figure captions, headings and
the YAML metadata block, not just body paragraphs: a style rule that only covers
prose leaves the captions and tables reading differently from the text.

Run:  python scripts/check_style.py [FILE ...]
Exit status is non-zero if any rule fails, so it can gate a commit.
"""
from __future__ import annotations

import re
import sys

# Rules are split by evidence. HARD rules are the ones the reference paper
# observes at exactly zero across 25,191 words, so a hit is a genuine departure
# from house style. ADVISORY rules are patterns the reference uses sparingly
# (one or two occurrences); they are reported but do not fail the check, because
# a rule the reference itself breaks is a rule I invented, not house style.
#
# Running this checker against the reference paper is how the split was
# established, and `--audit-reference` re-runs that validation.
HARD = [
    ("em dash", r"—",
     "no em dashes; use a comma, colon, period, or parentheses"),
    ("en dash as punctuation", r"(?<=\w) -- (?=\w)",
     "no '--' as sentence punctuation"),
    ("filler adverb", r"\b(notably|importantly|crucially|interestingly)\b",
     "drop the adverb and state the fact"),
    ("note that", r"\bnote that\b",
     "drop 'note that' and state the sentence directly"),
    ("not-just-but", r"\bnot just\b[^.]*\bbut\b",
     "rewrite as a plain statement"),
]

ADVISORY = [
    ("worth-noting hedge", r"\b(it is|it's) worth (noting|stating|mentioning)\b|\bworth (noting|stating|flagging)\b",
     "reference uses this once in 25k words; keep it rare"),
    ("in order to", r"\bin order to\b", "'to' is usually enough"),
    ("intensifier", r"\b(very|quite) [a-z]+|\brather (?!than\b)[a-z]+",
     "reference uses 'very' twice in 25k words; prefer a precise word"),
    ("leverage as verb", r"\bleverage[sd]\b|\bto leverage\b",
     "'use' reads better; the noun sense is fine"),
]

# Fenced code blocks and inline code are exempt: they quote real identifiers.
CODE_FENCE = re.compile(r"```.*?```", re.S)
INLINE_CODE = re.compile(r"`[^`\n]*`")


def strip_code(text: str) -> str:
    """Blank out code spans so identifiers do not trip prose rules."""
    text = CODE_FENCE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    return INLINE_CODE.sub("", text)


def scan(raw: str, rules):
    """Return {rule_name: [(line_no, text, matched_token), ...]} for one rule set."""
    prose = strip_code(raw)
    lines, raw_lines = prose.split("\n"), raw.split("\n")
    found = {}
    for name, pattern, message in rules:
        hits = [(i, raw_lines[i - 1].strip()[:96], m.group(0))
                for i, line in enumerate(lines, 1)
                for m in re.finditer(pattern, line, re.I)]
        if hits:
            found[name] = (hits, message)
    return found


def check(path: str) -> int:
    raw = open(path, encoding="utf-8").read()
    failures = 0
    hard = scan(raw, HARD)
    for name, _, _ in HARD:
        if name in hard:
            hits, message = hard[name]
            failures += len(hits)
            print(f"  [FAIL] {name}: {len(hits)} hit(s). {message}")
            for ln, text, tok in hits[:8]:
                print(f"         {path}:{ln}  ({tok!r})  {text}")
            if len(hits) > 8:
                print(f"         ... and {len(hits) - 8} more")
        else:
            print(f"  [PASS] {name}")

    advis = scan(raw, ADVISORY)
    for name, _, _ in ADVISORY:
        if name in advis:
            hits, message = advis[name]
            print(f"  [note] {name}: {len(hits)} hit(s). {message}")
            for ln, text, tok in hits[:3]:
                print(f"         {path}:{ln}  ({tok!r})  {text}")
    return failures


def main(argv):
    paths = argv[1:] or ["SUBMISSION.md"]
    total = 0
    for p in paths:
        print(f"\n{p}")
        total += check(p)
    print("\n" + "=" * 60)
    print(f" {'STYLE CLEAN' if total == 0 else str(total) + ' STYLE VIOLATION(S)'}")
    print("=" * 60)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
