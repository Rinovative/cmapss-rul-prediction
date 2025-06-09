# src/poetry_lint.py

import subprocess
import sys


def main():
    cmds = [
        ["isort", ".", "--profile", "black"],
        ["black", "src", "--line-length", "150"],
        [
            "flake8",
            "--select=E,F",
            "--ignore=E203",
            "--exclude=.venv,__pycache__,.pytest_cache",
            "src",
            "--max-line-length=150",
        ],
        # ["basedpyright", "--level", "error"],
        ["complexipy", ".", "--max-complexity", "25", "--details", "normal", "--sort", "asc"],
    ]
    for cmd in cmds:
        print(f"\n>>> {' '.join(cmd)}")
        result = subprocess.run(["poetry", "run"] + cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)
