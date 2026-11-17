"""The plutobt command lines in the README parse with the real CLI parser (nothing runs)."""
from __future__ import annotations

import re
import shlex
from pathlib import Path

from plutobt.cli import _parser

README = Path(__file__).resolve().parent.parent / "README.md"
SECTIONS = ("Run without code", "Buying data from Databento")


def _section(text: str, title: str) -> str:
    m = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert m, f"README has no '## {title}' section"
    return m.group(1)


def _command_lines(body: str) -> list[str]:
    lines = []
    for block in re.findall(r"```[a-z]*\n(.*?)```", body, re.S):
        lines += [ln.strip() for ln in block.splitlines() if ln.strip().startswith("plutobt ")]
    return lines


def test_readme_cli_examples_parse():
    text = README.read_text(encoding="utf-8")
    seen = []
    for title in SECTIONS:
        lines = _command_lines(_section(text, title))
        assert lines, f"no plutobt command lines in '{title}'"
        seen += lines
    for line in seen:
        assert "&&" not in line and not line.endswith("\\"), line
        argv = shlex.split(line)[1:]
        _parser().parse_args(argv)
    assert any(ln.startswith("plutobt run ") for ln in seen)
    assert any(ln.startswith("plutobt quote ") for ln in seen)
    assert any(ln.startswith("plutobt fetch ") for ln in seen)


def test_readme_has_the_required_sentences():
    text = README.read_text(encoding="utf-8")
    for s in ("pip install plutobt[databento]",
              "Data you buy is yours under your Databento terms; do not share it.",
              "Bought data is kept in ~/.plutobt/data on your computer. Never commit or "
              "share that folder.",
              "representative public retail costs, not a live account"):
        assert s in text, s
    for title in SECTIONS:
        assert chr(0x2014) not in _section(text, title)
