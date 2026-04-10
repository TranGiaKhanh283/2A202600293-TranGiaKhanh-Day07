from __future__ import annotations

import re
import unicodedata
import argparse
from pathlib import Path


NOISE_PATTERNS = [
    r"(?im)^toggle the table of contents.*$",
    r"(?im)^contents\s*$",
    r"(?im)^from wikipedia.*$",
]


def clean_text(text: str) -> str:
    # Normalize Unicode to avoid inconsistent tokenization/chunk boundaries.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue

        # Remove heavy citation-only tails often seen in scraped wiki text.
        line = re.sub(r"\s*\[\s*\d+\s*\]\s*", " ", line)
        line = re.sub(r"\s+", " ", line).strip()
        kept.append(line)

    out = "\n".join(kept)
    for pattern in NOISE_PATTERNS:
        out = re.sub(pattern, "", out)

    # Collapse excessive blank lines.
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    return out


def clean_markdown_files(data_dir: Path, output_dir: Path, pattern: str = "*.md") -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_files = sorted(data_dir.glob(pattern))
    if not md_files:
        print(f"No markdown files found in: {data_dir}")
        return 1

    for src in md_files:
        raw = src.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw)
        dst = output_dir / src.name
        dst.write_text(cleaned, encoding="utf-8")
        print(f"Cleaned: {src.name} -> {dst}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean markdown files for chunking.")
    parser.add_argument(
        "--pattern",
        default="*.md",
        help="Glob pattern under data/ to select markdown files (default: *.md).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    output_dir = data_dir / "cleaned"
    return clean_markdown_files(data_dir, output_dir, pattern=args.pattern)


if __name__ == "__main__":
    raise SystemExit(main())

