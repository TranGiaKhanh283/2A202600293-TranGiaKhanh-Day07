from __future__ import annotations

from datetime import date
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re


class WikiTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._skip_stack: list[str] = []
        self._capture_depth = 0
        self._in_content_root = False
        self.title = ""
        self._in_title = False
        self.lines: list[str] = []
        self._buffer = ""
        self.source_url = ""
        self._active_tag: str | None = None
        self._active_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k: (v or "") for k, v in attrs}
        class_name = attrs_dict.get("class", "")
        tag_id = attrs_dict.get("id", "")
        rel = attrs_dict.get("rel", "")

        if tag == "meta" and attrs_dict.get("property") == "og:url":
            self.source_url = attrs_dict.get("content", "") or self.source_url
        if tag == "link" and "canonical" in rel and not self.source_url:
            self.source_url = attrs_dict.get("href", "") or self.source_url

        if tag in {"script", "style", "noscript"}:
            self._skip_stack.append(tag)
            return

        if tag == "title":
            self._in_title = True

        if tag == "div" and ("mw-parser-output" in class_name or tag_id == "mw-content-text"):
            self._capture_depth += 1
            self._in_content_root = True
            return

        if self._in_content_root and tag == "div":
            self._capture_depth += 1

        if self._in_content_root and tag in {"p", "h2", "h3"}:
            if self._active_tag is None:
                self._active_tag = tag
                self._active_depth = 1
                self._flush_buffer()
                if tag in {"h2", "h3"}:
                    self.lines.append("\n## ")
            else:
                self._active_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._skip_stack and tag == self._skip_stack[-1]:
            self._skip_stack.pop()
            return

        if tag == "title":
            self._in_title = False

        if self._active_tag == tag:
            self._active_depth -= 1
            if self._active_depth == 0:
                self._flush_buffer()
                self.lines.append("\n")
                self._active_tag = None

        if self._in_content_root and tag == "div" and self._capture_depth > 0:
            self._capture_depth -= 1
            if self._capture_depth == 0:
                self._in_content_root = False

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        if self._in_title:
            self.title += data
            return
        if self._in_content_root and self._active_tag in {"p", "h2", "h3"}:
            clean = unescape(data).replace("\xa0", " ")
            if clean.strip():
                self._buffer += clean.strip() + " "

    def _flush_buffer(self) -> None:
        text = " ".join(self._buffer.split())
        if text:
            self.lines.append(text)
        self._buffer = ""

    def get_markdown_body(self) -> str:
        self._flush_buffer()
        raw = "".join(self.lines)
        raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
        # Remove frequent navigation noise that sometimes leaks in snapshots.
        raw = re.sub(r"(?im)^contents\s*$", "", raw)
        raw = re.sub(r"(?im)^from wikipedia.*$", "", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
        return raw


def convert_html_file(html_path: Path) -> Path:
    parser = WikiTextExtractor()
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    parser.feed(html)

    title = parser.title.strip() or html_path.stem.replace("-", " ")
    source = parser.source_url.strip()
    body = parser.get_markdown_body()
    if not body:
        body = "No article text extracted from this HTML file."

    front = [
        f"# {title}",
        "",
        f"Source: {source or html_path.name}",
        f"Retrieved: {date.today().isoformat()}",
        f"Domain: ai_knowledge_wikipedia",
        f"Language: en",
        "",
        "---",
        "",
    ]

    md_path = html_path.with_suffix(".md")
    md_path.write_text("\n".join(front) + body + "\n", encoding="utf-8")
    return md_path


def main() -> int:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    html_files = sorted(data_dir.glob("*.html"))
    if not html_files:
        print("No HTML files found in data/.")
        return 1

    for html_file in html_files:
        md_file = convert_html_file(html_file)
        print(f"Converted: {html_file.name} -> {md_file.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

