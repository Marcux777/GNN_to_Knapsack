#!/usr/bin/env python3
"""Guard script to ensure .codex/ stays intact."""

import os
import subprocess
import sys

REQUIRED = [
    ".codex/config.yaml",
    ".codex/system.md",
    ".codex/project.md",
    ".codex/style.md",
    ".codex/tasks.md",
    ".codex/runbook.md",
    ".codex/tools.md",
    ".codex/risks.md",
    ".codex/eval.md",
]


def git_diff_cached() -> str:
    try:
        return subprocess.check_output(
            ["git", "diff", "--cached", "--name-status"], text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        print("git diff --cached failed", exc, file=sys.stderr)
        return ""


def main() -> int:
    missing = [path for path in REQUIRED if not os.path.exists(path)]
    if missing:
        print("Faltando artefatos do .codex:", ", ".join(missing))
        return 1

    diff_output = git_diff_cached()
    if not diff_output:
        return 0

    violations: list[tuple[str, str]] = []
    for line in diff_output.splitlines():
        try:
            status, path = line.split("\t", 1)
        except ValueError:
            continue
        if path.startswith(".codex/") and status in {"D"}:
            violations.append((status, path))

    if violations:
        print("Tentativa de remover arquivos de .codex sem aprovação:")
        for status, path in violations:
            print(f"  {status}\t{path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
