from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[2]

SKIP_DIRS = {
    ".git",
    ".agents",
    ".codex",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "secrets",
}
SKIP_PARTS = {
    ("analysis", "private"),
    ("apex-llm-poc", "analysis", "private"),
}
SKIP_FILENAMES = {
    ".env",
    ".gitignore",
    "secret_hygiene.py",
    "test_secret_hygiene.py",
}
TEXT_SUFFIXES = {
    ".bat",
    ".cfg",
    ".csv",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
PATTERNS = {
    "credential_json_private_key": re.compile(r'"private_key"\s*:'),
    "credential_json_client_email": re.compile(r'"client_email"\s*:'),
    "credential_json_type": re.compile(r'"type"\s*:\s*"service_account"'),
    "pem_private_key": re.compile(r"BEGIN PRIVATE KEY"),
    "specific_local_key_filename": re.compile(r"gcp-key\.json", re.IGNORECASE),
    "credential_env_points_to_secrets": re.compile(
        r"GOOGLE_APPLICATION_CREDENTIALS\s*=\s*.*secrets", re.IGNORECASE
    ),
}


@dataclass(frozen=True)
class Finding:
    path: Path
    pattern: str
    line_no: int

    def public_message(self, root: Path) -> str:
        try:
            rel = self.path.relative_to(root)
        except ValueError:
            rel = self.path
        return f"{rel}:{self.line_no} matched {self.pattern}"


def _has_part_sequence(parts: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    seq_len = len(sequence)
    return any(parts[i : i + seq_len] == sequence for i in range(0, len(parts) - seq_len + 1))


def should_skip(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    parts = rel.parts
    if any(part in SKIP_DIRS for part in parts[:-1]):
        return True
    if any(_has_part_sequence(parts, sequence) for sequence in SKIP_PARTS):
        return True
    if path.name in SKIP_FILENAMES:
        return True
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return True
    return False


def scan_workspace(root: Path = DEFAULT_ROOT) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    for path in root.rglob("*"):
        if not path.is_file() or should_skip(path, root):
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(Finding(path=path, pattern=name, line_no=line_no))
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    findings = scan_workspace(root)
    if findings:
        print("Secret hygiene scan failed. Matched file/pattern locations only; matched text is withheld.")
        for finding in findings:
            print(finding.public_message(root))
        return 1
    print(f"Secret hygiene scan passed under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
