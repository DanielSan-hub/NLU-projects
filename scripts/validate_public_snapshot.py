#!/usr/bin/env python3
"""Fast, dependency-free validation for the public repository snapshot."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARTS = ("LM/partA", "LM/partB", "NLU/partA", "NLU/partB")
REQUIRED_MODULES = ("main.py", "model.py", "functions.py", "utils.py", "README.md")
REQUIRED_FLAGS = ("--mode", "--device", "--seed")
REQUIRED_ARTIFACTS = (
    "PROJECT_CARD.md",
    "results/master_results.csv",
    "results/summary.md",
    "reports/REPORT_NOTES.md",
    "reports/figures/project_best_summary.svg",
)
FORBIDDEN_EXTENSIONS = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}


def check(condition: bool, message: str, errors: list[str]) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        errors.append(message)


def main() -> int:
    errors: list[str] = []

    for relative_part in PARTS:
        part = ROOT / relative_part
        check(part.is_dir(), f"{relative_part} exists", errors)

        for filename in REQUIRED_MODULES:
            path = part / filename
            check(path.is_file(), f"{relative_part}/{filename} exists", errors)

        entrypoint = part / "main.py"
        if entrypoint.is_file():
            source = entrypoint.read_text(encoding="utf-8")
            for flag in REQUIRED_FLAGS:
                check(flag in source, f"{relative_part}/main.py exposes {flag}", errors)

    python_files = sorted(
        path
        for folder in (ROOT / "LM", ROOT / "NLU", ROOT / "scripts")
        for path in folder.rglob("*.py")
    )
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f"invalid Python source: {path.relative_to(ROOT)} ({exc})")
    check(not any("invalid Python source" in error for error in errors),
          f"{len(python_files)} Python files parse successfully", errors)

    notebooks = list((ROOT / "LM").rglob("*.ipynb")) + list((ROOT / "NLU").rglob("*.ipynb"))
    check(not notebooks, "project submission directories contain no notebooks", errors)

    for relative_path in REQUIRED_ARTIFACTS:
        check((ROOT / relative_path).is_file(), f"{relative_path} exists", errors)

    forbidden_files = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_EXTENSIONS
    ]
    check(not forbidden_files, "no heavyweight checkpoint/model artifacts are tracked", errors)

    if errors:
        print(f"\nPublic snapshot validation failed with {len(errors)} error(s).")
        return 1
    print("\nPublic snapshot validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
