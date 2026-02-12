#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from pathlib import Path
import fitz  # PyMuPDF

logger = logging.getLogger("local_pdf_extract")


# -------------------------
# PDF extraction
# -------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract raw text from digital PDFs using PyMuPDF.
    Output is plain text (Gemini-compatible).
    """
    doc = fitz.open(pdf_path)
    chunks = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text and text.strip():
            chunks.append(text.strip())

    return "\n\n".join(chunks).strip()


# -------------------------
# Path helpers
# -------------------------
def pdf_path_to_md_path(pdf_relative: str) -> str:
    md_path = pdf_relative.replace("/pdf/", "/md/")
    if md_path.endswith(".pdf"):
        md_path = md_path[:-4] + ".md"
    return md_path


def resolve_pdf_path(base_path: Path, pdf_relative: str) -> Path | None:
    """
    Try multiple strategies to locate the PDF.
    """
    # 1) default (same logic as Gemini)
    p1 = base_path / pdf_relative
    if p1.exists():
        return p1

    # 2) common nested-dataset case (base/base/pdf/...)
    p2 = base_path / base_path.name / pdf_relative
    if p2.exists():
        return p2

    # 3) last resort: search by filename
    name = Path(pdf_relative).name
    matches = list(base_path.rglob(name))
    if len(matches) == 1:
        return matches[0]

    return None


# -------------------------
# Markdown
# -------------------------
def build_markdown(text: str, metadata: dict) -> str:
    md = []

    md.append("---")
    for k in [
        "title", "type", "number", "year",
        "subject", "author", "presentation_date",
        "url", "house"
    ]:
        v = metadata.get(k)
        if not v:
            continue

        if isinstance(v, list):
            if len(v) == 1:
                md.append(f"{k}: {v[0]}")
            else:
                md.append(f"{k}:")
                for i in v:
                    md.append(f"  - {i}")
        else:
            md.append(f"{k}: {v}")

    md.append("---\n")
    md.append(f"# {metadata.get('title','')}\n")
    md.append(text)

    return "\n".join(md)


# -------------------------
# Document processing
# -------------------------
def process_document(doc: dict, base_path: Path, force: bool) -> tuple[str, str]:
    pdf_rel  = None

    if doc.get("pdf_files"):
        pdf_rel = doc["pdf_files"][0]
    elif doc.get("file_urls"):
        pdf_rel = doc["file_urls"][0]

    if not pdf_rel:
        return "skipped", "no pdf path field (pdf_files/file_urls)"

    md_relative = pdf_path_to_md_path(pdf_rel )
    md_path = base_path / md_relative

    if md_path.exists() and not force:
        return "skipped", "markdown already exists"

    pdf_path = resolve_pdf_path(base_path, pdf_rel )
    if not pdf_path:
        return "failed", f"PDF not found (tried multiple paths): {pdf_rel }"

    logger.info(f"PDF  → {pdf_path}")
    logger.info(f"MD   → {md_path}")

    text = extract_text_from_pdf(pdf_path)
    if not text:
        return "failed", "empty PDF extraction"

    metadata = {
        "title": doc.get("title"),
        "type": doc.get("type"),
        "number": doc.get("number"),
        "year": doc.get("year"),
        "subject": doc.get("subject"),
        "author": doc.get("author"),
        "presentation_date": doc.get("presentation_date"),
        "url": doc.get("url"),
        "house": doc.get("house"),
    }

    md_content = build_markdown(text, metadata)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_content, encoding="utf-8")

    return "ok", md_path.name


# -------------------------
# JSON processing
# -------------------------
def process_json(json_file: Path, force: bool, debug: bool):
    base_path = json_file.parent
    data = json.loads(json_file.read_text(encoding="utf-8"))

    stats = {"ok": 0, "skipped": 0, "failed": 0}

    logger.info(f"\nProcessing {json_file.name}")
    logger.info(f"Base path: {base_path}")

    for i, doc in enumerate(data, 1):
        status, msg = process_document(doc, base_path, force)
        stats[status] += 1

        if debug:
            logger.info(f"[{i:04}] {status.upper():7} {doc.get('title','')} → {msg}")

    return stats


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Local PDF → Markdown extractor (Gemini-compatible output)"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("json_file", nargs="?", type=Path)
    group.add_argument("--folder", type=Path)

    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.debug else logging.WARNING,
        format="%(levelname)s - %(message)s"
    )

    json_files = []

    if args.folder:
        if not args.folder.exists():
            print("Folder not found", file=sys.stderr)
            sys.exit(1)
        json_files = sorted(args.folder.glob("*.json"))
        if not json_files:
            print("No JSON files found in folder", file=sys.stderr)
            sys.exit(1)
    else:
        if not args.json_file.exists():
            print("JSON file not found", file=sys.stderr)
            sys.exit(1)
        json_files = [args.json_file]

    overall = {"ok": 0, "skipped": 0, "failed": 0}

    for jf in json_files:
        stats = process_json(jf, args.force, args.debug)
        for k in overall:
            overall[k] += stats[k]

    print("\nSummary")
    for k, v in overall.items():
        print(f"{k:7}: {v}")


if __name__ == "__main__":
    main()
