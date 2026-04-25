from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff
import re


TOKEN_PATTERN = re.compile(r"\w+|\s+|[^\w\s]", re.UNICODE)
CONTEXT_TOKENS = 6


@dataclass(frozen=True)
class ChangeBlock:
    index: int
    operation: str
    before: str
    after: str
    before_context: str
    after_context: str


def build_unified_diff(before: str, after: str, *, fromfile: str = "before", tofile: str = "after") -> str:
    diff = unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=fromfile,
        tofile=tofile,
        lineterm="",
    )
    rendered = "".join(diff)
    if rendered:
        return rendered if rendered.endswith("\n") else rendered + "\n"
    return ""


def build_change_blocks(before: str, after: str) -> list[ChangeBlock]:
    before_tokens = TOKEN_PATTERN.findall(before)
    after_tokens = TOKEN_PATTERN.findall(after)
    matcher = SequenceMatcher(a=before_tokens, b=after_tokens, autojunk=False)

    blocks: list[ChangeBlock] = []
    for index, (opcode, i1, i2, j1, j2) in enumerate(matcher.get_opcodes(), start=1):
        if opcode == "equal":
            continue

        before_text = "".join(before_tokens[i1:i2]).strip()
        after_text = "".join(after_tokens[j1:j2]).strip()
        before_context = "".join(before_tokens[max(0, i1 - CONTEXT_TOKENS) : min(len(before_tokens), i2 + CONTEXT_TOKENS)]).strip()
        after_context = "".join(after_tokens[max(0, j1 - CONTEXT_TOKENS) : min(len(after_tokens), j2 + CONTEXT_TOKENS)]).strip()

        blocks.append(
            ChangeBlock(
                index=index,
                operation=opcode,
                before=before_text,
                after=after_text,
                before_context=before_context,
                after_context=after_context,
            )
        )

    return blocks


def render_change_report(before: str, after: str, *, title: str) -> str:
    blocks = build_change_blocks(before, after)
    report = [
        f"# {title}",
        "",
        f"- changed: {'yes' if before != after else 'no'}",
        f"- before_chars: {len(before)}",
        f"- after_chars: {len(after)}",
        f"- change_blocks: {len(blocks)}",
        "",
    ]

    if not blocks:
        report.append("No textual changes.")
        report.append("")
        return "\n".join(report)

    report.append("## Blocks")
    report.append("")
    for block in blocks:
        report.append(f"### {block.index}. {block.operation}")
        report.append("")
        report.append(f"- before: `{block.before}`")
        report.append(f"- after: `{block.after}`")
        report.append(f"- before_context: `{block.before_context}`")
        report.append(f"- after_context: `{block.after_context}`")
        report.append("")

    return "\n".join(report)
