"""Recursive character splitter (CP 3.1: "In my case, I'm using Recursive
Character, which is a method in which data is broken down from large to small
in an effort to preserve as much of the string as possible").

Splits on \\n\\n first, then \\n, then sentence, then word — only descending to
a finer separator when a segment still exceeds the target chunk size. Overlap
keeps chunk boundaries coherent (the paper's fix for "losing coherence in the
strings of vector data").

Two invariants the chunker guarantees (both were real answer-quality bugs):
  * sentence-final punctuation is never lost at a boundary — separators are
    kept attached to the part they follow, so a chunk never ends in
    '...high standards' when the source said '...high standards.';
  * an overlapping tail never starts mid-word — if the previous chunk's tail
    has no line/sentence boundary, the overlap is dropped entirely instead of
    opening a chunk with a fragment like 'eed, there is ...'.
"""

from __future__ import annotations

import re
from typing import List, Tuple

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_keep(text: str, sep: str) -> List[Tuple[str, str]]:
    """Split `text` on `sep`, returning (part, trailing_sep) pairs so the
    separator is never stranded on the chunk after it."""
    items: List[Tuple[str, str]] = []
    pat = re.compile(re.escape(sep))
    pos = 0
    for m in pat.finditer(text):
        items.append((text[pos:m.start()], sep))
        pos = m.end()
    if text[pos:]:
        items.append((text[pos:], ""))
    return items


def _split_text(text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    sep = separators[0] if separators else " "
    remaining = separators[1:] if len(separators) > 1 else [" "]
    items = _split_keep(text, sep)
    if not items:
        return []
    merged: List[str] = []
    buf = ""
    for part, trail in items:
        if not part:
            continue
        if len(part) > chunk_size:
            # part alone exceeds the budget: split it finer. A small lead-in
            # buffer (e.g. a section heading) must stay attached to the first
            # sub-piece — flushing it on its own creates orphan fragments.
            # The separator that followed the part in the source stays attached
            # to the last sub-piece it produced.
            sub = _split_text(part, remaining, chunk_size, chunk_overlap) or [part]
            if trail:
                sub = sub[:-1] + [sub[-1] + trail]
            if buf:
                lead = buf.rstrip()
                if len(lead) + 1 + len(sub[0]) <= chunk_size:
                    sub = [lead + " " + sub[0]] + sub[1:]
                else:
                    merged.append(lead)
                buf = ""
            merged.extend(sub)
            continue
        candidate = f"{buf}{part}{trail}" if buf else part + trail
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            merged.append(buf.rstrip())
            buf = part + trail
    if buf:
        merged.append(buf.rstrip())
    return [m for m in merged if m.strip()]


def recursive_split(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 150,
    separators: List[str] = SEPARATORS,
) -> List[str]:
    pieces = _split_text(text, list(separators), chunk_size, chunk_overlap)
    pieces = [p.strip() for p in pieces if p.strip()]
    if not pieces or chunk_overlap <= 0:
        return pieces
    out: List[str] = [pieces[0]]
    for piece in pieces[1:]:
        # The overlap must begin at a clean boundary: a line break, or a
        # sentence end followed by a sentence start (capital/quote). Anything
        # else (an abbreviation, a decimal, a lowercase continuation) means the
        # slice would open mid-word/mid-sentence — drop the overlap instead.
        window = out[-1][-chunk_overlap * 2:]
        tail = ""
        nl = window.rfind("\n")
        if nl != -1:
            tail = window[nl + 1 :].strip()
        else:
            last = None
            for m in re.finditer(r"(?<=[.!?]) (?=[\"'(A-Z])", window):
                last = m
            if last is not None:
                tail = window[last.end():].strip()
        if not tail or len(tail) < 15 or len(tail) > chunk_overlap * 2:
            tail = ""
        if not tail:
            out.append(piece)
            continue
        candidate = f"{tail} {piece}"
        out.append(candidate if len(candidate) <= chunk_size * 2 else piece)
    return out
