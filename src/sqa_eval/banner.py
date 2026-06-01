from __future__ import annotations

import os
import sys

_ASCII = r"""
  _____  ____                 ________      __     _
 / ____|/ __ \     /\        |  ____\ \    / /\   | |
| (___ | |  | |   /  \ ______| |__   \ \  / /  \  | |
 \___ \| |  | |  / /\ \______|  __|   \ \/ / /\ \ | |
 ____) | |__| | / ____ \     | |____   \  / ____ \| |____
|_____/ \___\_\/_/    \_\    |______|   \/_/    \_\______|
"""


def _ansi(code: str) -> str:
    return f"\x1b[{code}m"


RESET = _ansi("0")
BOLD = _ansi("1")
DIM = _ansi("2")
LIGHT_GRAY = _ansi("90")


def print_banner(model_name: str = "") -> None:
    if os.getenv("NO_BANNER", "").lower() in {"1", "true", "yes"}:
        return
    if not sys.stderr.isatty():
        return

    title = "SQA-EVAL"
    tagline = "speech quality assessment toolkit"

    content_lines = [ln.rstrip("\n") for ln in _ASCII.strip("\n").splitlines()]
    content_lines += ["", title, tagline]

    inner_w = max(len(ln) for ln in content_lines)
    tl, tr, bl, br = "\u250c", "\u2510", "\u2514", "\u2518"
    h, v = "\u2500", "\u2502"

    top = f"{tl}{h * (inner_w + 2)}{tr}"
    bot = f"{bl}{h * (inner_w + 2)}{br}"

    framed = [top]
    for ln in content_lines:
        framed.append(f"{v} {ln.ljust(inner_w)} {v}")
    framed.append(bot)

    styled: list[str] = []
    for i, ln in enumerate(framed):
        if i == 0 or i == len(framed) - 1:
            styled.append(f"{LIGHT_GRAY}{ln}{RESET}")
            continue

        ci = i - 1
        raw = content_lines[ci]
        if raw == title:
            styled_ln = ln.replace(raw, f"{BOLD}{LIGHT_GRAY}{raw}{RESET}{LIGHT_GRAY}", 1)
            styled.append(f"{LIGHT_GRAY}{styled_ln}{RESET}")
        elif raw == tagline:
            styled_ln = ln.replace(raw, f"{DIM}{LIGHT_GRAY}{raw}{RESET}{LIGHT_GRAY}", 1)
            styled.append(f"{LIGHT_GRAY}{styled_ln}{RESET}")
        else:
            styled.append(f"{LIGHT_GRAY}{ln}{RESET}")

    output = "\n".join(styled) + "\n"
    if model_name:
        output += f"  {BOLD}Model:{RESET} {model_name}\n"
    print(output, file=sys.stderr, flush=True)
