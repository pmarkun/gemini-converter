#!/usr/bin/env python3
"""
CLI tool to process JSON files containing legislative document metadata
and generate Markdown files from PDFs using Gemini Vision API.

Usage:
    python gemini_extract.py <json_file> [--force] [--max-documents N] [--debug] [--workers N]
    python gemini_extract.py --folder <directory> [--force] [--max-documents N] [--debug] [--workers N]

Environment:
    .env file with GEMINI_API_KEY variable

Rate Limits (Gemini 2.0 Flash Lite):
    - 4,000 requests per minute (RPM)
    - 4,000,000 tokens per minute (TPM)

Worker Configuration:
    - Default: 20 workers (conservative, ~40-50 docs/min)
    - Aggressive: 50 workers (~100-120 docs/min)
    - Maximum safe: 60 workers (well below 4,000 RPM limit)

    With ~3-5s per document, even 60 workers = ~60 req/min, which is only
    1.5% of the 4,000 RPM limit. The bottleneck is API latency, not rate limits.
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from threading import Lock

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

# Logger will be configured after parsing arguments
logger = logging.getLogger(__name__)

# Thread-safe lock for progress tracking
progress_lock = Lock()


def get_gemini_client():
    """Get Gemini client with API key from .env file."""
    # Load .env file
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file", file=sys.stderr)
        print("Please create a .env file with: GEMINI_API_KEY=your-key-here", file=sys.stderr)
        sys.exit(1)

    return genai.Client(api_key=api_key)


def extract_text_with_gemini(pdf_path: Path, client: genai.Client, debug: bool = False) -> str:
    """
    Extract text from PDF using Gemini Vision API.

    Args:
        pdf_path: Path to PDF file
        client: Gemini client instance
        debug: Enable debug logging

    Returns:
        Extracted text content in the format:
        Ementa
        [ementa]
        Texto da Lei
        [texto]
        Justificativa
        [justificativa]
    """
    if debug:
        logger.info(f"    Uploading PDF to Gemini...")

    # Read PDF file
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()

    # Create content with PDF file and new prompt
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    data=pdf_data,
                    mime_type="application/pdf"
                ),
                types.Part.from_text(
                    text="""Extraia a integra do texto e a justificativa no formato:
Texto da Lei
[texto da Lei]

Justificativa
[justificativa]"""
                ),
            ],
        ),
    ]

    if debug:
        logger.info(f"    Requesting extraction from Gemini...")

    # Configure generation
    generate_content_config = types.GenerateContentConfig(
        temperature=0.1,  # Low temperature for consistent extraction
        max_output_tokens=8192,  # Reasonable limit for most documents
    )

    # Make the API call using gemini-2.0-flash-lite
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=contents,
            config=generate_content_config,
        )

        # Extract text from response
        if response and response.text:
            text = response.text.strip()
            if debug:
                logger.info(f"    Extraction successful ({len(text)} chars)")
            return text
        else:
            raise Exception("Empty response from Gemini API")

    except Exception as e:
        raise Exception(f"Gemini API error: {str(e)}")


def load_json(json_path: Path, debug: bool = False) -> list[dict]:
    """Load and parse the JSON file containing document metadata."""
    if debug:
        logger.info(f"Loading JSON file: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("Error: JSON file must contain a list of documents", file=sys.stderr)
            sys.exit(1)

        if debug:
            logger.info(f"Loaded {len(data)} documents from JSON")
        return data

    except FileNotFoundError:
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}", file=sys.stderr)
        sys.exit(1)


def generate_pdf_path_from_metadata(doc: dict) -> str:
    """
    Generate PDF path from document metadata for alternative format.
    
    Example:
        type: "PL", number: 7, year: 2020
        -> "pdf/pl-7-2020.pdf"
    """
    doc_type = doc.get("type", "").lower()
    doc_number = doc.get("number", "")
    doc_year = doc.get("year", "")
    
    return f"pdf/{doc_type}-{doc_number}-{doc_year}.pdf"


def pdf_path_to_md_path(pdf_path: str) -> str:
    """
    Convert PDF path to markdown path by replacing /pdf/ with /md/ and .pdf with .md

    Example:
        "ce-fortaleza/pdf/2024/projeto-de-lei-ordinária-255-2024.pdf"
        -> "ce-fortaleza/md/2024/projeto-de-lei-ordinária-255-2024.md"
    """
    md_path = pdf_path.replace("/pdf/", "/md/")
    if md_path.endswith(".pdf"):
        md_path = md_path[:-4] + ".md"
    return md_path


def should_process_document(doc: dict, force: bool, base_path: Path, alt_format: bool = False) -> tuple[bool, Optional[str]]:
    """
    Determine if a document should be processed.

    Returns:
        Tuple of (should_process, reason)
    """
    # Get PDF path based on format
    if alt_format:
        # Alternative format: generate PDF path from metadata
        pdf_relative = generate_pdf_path_from_metadata(doc)
    else:
        # Original format: use pdf_files field
        if not doc.get("pdf_files"):
            return False, "No PDF files specified"
        pdf_relative = doc["pdf_files"][0]

    # Generate markdown path from PDF path
    md_relative = pdf_path_to_md_path(pdf_relative)
    md_path = base_path / md_relative

    # If force flag is set, always process
    if force:
        return True, "Force flag set"

    # Skip if markdown file already exists
    if md_path.exists():
        return False, f"Markdown file already exists: {md_path}"

    return True, "New document"


def build_markdown_with_text(text: str, metadata: dict) -> str:
    """
    Build markdown document with metadata frontmatter and text content.

    Args:
        text: Extracted text content
        metadata: Document metadata dictionary

    Returns:
        Formatted markdown string
    """
    markdown_parts = []

    # Add metadata header if provided
    if metadata:
        markdown_parts.append("---")

        # Add each metadata field that exists
        field_mapping = [
            ("title", "title"),
            ("type", "type"),
            ("number", "number"),
            ("year", "year"),
            ("subject", "subject"),
            ("author", "author"),
            ("presentation_date", "presentation_date"),
            ("url", "url"),
            ("house", "house"),
        ]

        for key, md_key in field_mapping:
            if key in metadata and metadata[key]:
                value = metadata[key]
                # Handle lists (like authors)
                if isinstance(value, list):
                    if len(value) == 1:
                        markdown_parts.append(f"{md_key}: {value[0]}")
                    else:
                        markdown_parts.append(f"{md_key}:")
                        for item in value:
                            markdown_parts.append(f"  - {item}")
                else:
                    markdown_parts.append(f"{md_key}: {value}")

        markdown_parts.append("---")
        markdown_parts.append("")  # Empty line after frontmatter

    # Add the main title if available
    if metadata and "title" in metadata:
        markdown_parts.append(f"# {metadata['title']}")
        markdown_parts.append("")

    # Add the extracted text
    markdown_parts.append(text)

    return "\n".join(markdown_parts)


def save_markdown(content: str, output_path: Path):
    """Save markdown content to file, creating directories if needed."""
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the markdown file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def show_text_preview(text: str, max_chars: int = 300):
    """Show a preview of extracted text for debugging."""
    if not text:
        logger.info("    [No text extracted]")
        return

    preview = text[:max_chars].replace('\n', ' ')
    if len(text) > max_chars:
        preview += "..."
    logger.info(f"    Preview: {preview}")


def process_document(doc: dict, base_path: Path, client: genai.Client, debug: bool = False, alt_format: bool = False) -> tuple[bool, str]:
    """
    Process a single document: extract text with Gemini and save as markdown.

    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Get PDF path based on format
        if alt_format:
            # Alternative format: generate PDF path from metadata
            pdf_relative = generate_pdf_path_from_metadata(doc)
        else:
            # Original format: use pdf_files field
            pdf_relative = doc["pdf_files"][0]
        
        pdf_path = base_path / pdf_relative

        # Generate markdown output path from PDF path
        md_relative = pdf_path_to_md_path(pdf_relative)
        md_path = base_path / md_relative

        if debug:
            logger.info(f"  PDF: {pdf_path.name}")

        # Check if PDF exists
        if not pdf_path.exists():
            error_msg = f"PDF file not found: {pdf_path}"
            if debug:
                logger.error(f"    {error_msg}")
            return False, error_msg

        # Extract text using Gemini
        text = extract_text_with_gemini(pdf_path, client, debug=debug)

        if debug:
            show_text_preview(text)

        # Extract metadata for the markdown header
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

        # Build markdown
        markdown_content = build_markdown_with_text(text, metadata)

        # Save the markdown file
        save_markdown(markdown_content, md_path)

        if debug:
            logger.info(f"    ✓ Saved to {md_path.name}")
        return True, ""

    except Exception as e:
        error_msg = str(e)
        if debug:
            logger.error(f"    ✗ Error: {error_msg}")
        return False, error_msg


def process_document_wrapper(args):
    """
    Wrapper function for multithread processing.

    Args:
        args: Tuple of (doc, base_path, client, doc_index, debug, alt_format)

    Returns:
        Tuple of (doc_index, success, error_message, doc_title)
    """
    doc, base_path, client, doc_index, debug, alt_format = args

    success, error_msg = process_document(doc, base_path, client, debug=debug, alt_format=alt_format)
    title = doc.get("title", "Unknown")

    return (doc_index, success, error_msg, title)


def process_json_file(json_file: Path, client: genai.Client, force: bool, max_documents: Optional[int], workers: int, debug: bool, alt_format: bool = False) -> tuple[int, int, int, int]:
    """
    Process a single JSON file.

    Returns:
        Tuple of (total, processed, skipped, failed)
    """
    # Get base path (directory containing the JSON file)
    base_path = json_file.parent

    # Load documents from JSON
    documents = load_json(json_file, debug=debug)

    # Apply max-documents limit if specified
    if max_documents:
        documents = documents[:max_documents]
        if debug:
            logger.info(f"Limited to processing {len(documents)} documents")

    # Filter documents that need processing
    docs_to_process = []
    skipped = 0

    for doc in documents:
        should_process, reason = should_process_document(doc, force, base_path, alt_format=alt_format)
        if should_process:
            docs_to_process.append(doc)
        else:
            skipped += 1
            if debug:
                logger.info(f"Skipping {doc.get('title', 'Unknown')}: {reason}")

    total = len(documents)
    to_process = len(docs_to_process)

    if to_process == 0:
        return (total, 0, skipped, 0)

    # Process documents with multithread
    processed = 0
    failed = 0
    failed_docs = []

    if debug:
        print(f"\n{'='*60}")
        print(f"Processing {json_file.name} with Gemini Vision API ({to_process} documents)")
        print(f"Workers: {workers}")
        if alt_format:
            print("Format: Alternative (auto-generated PDF paths)")
        print('='*60 + '\n')

        # Debug mode: detailed logging (still parallel but with logs)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_doc = {}
            for i, doc in enumerate(docs_to_process):
                future = executor.submit(
                    process_document_wrapper,
                    (doc, base_path, client, i, True, alt_format)
                )
                future_to_doc[future] = doc

            # Process results as they complete
            for future in as_completed(future_to_doc):
                doc_index, success, error_msg, title = future.result()

                logger.info(f"[{doc_index + 1}/{to_process}] {title}")

                if success:
                    processed += 1
                    logger.info("    ✓ Success")
                else:
                    failed += 1
                    failed_docs.append((title, error_msg))
                    logger.error(f"    ✗ Failed: {error_msg}")

                logger.info("")
    else:
        # Normal mode: progress bar with parallel processing
        format_msg = " (alt format)" if alt_format else ""
        print(f"\nProcessing {json_file.name}{format_msg} with Gemini Vision API ({to_process} documents, {workers} workers)...")

        with tqdm(total=to_process, unit="doc", ncols=80, desc=json_file.stem) as pbar:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                future_to_doc = {}
                for i, doc in enumerate(docs_to_process):
                    future = executor.submit(
                        process_document_wrapper,
                        (doc, base_path, client, i, False, alt_format)
                    )
                    future_to_doc[future] = doc

                # Process results as they complete
                for future in as_completed(future_to_doc):
                    doc_index, success, error_msg, title = future.result()

                    if success:
                        processed += 1
                        pbar.set_postfix_str(f"OK:{processed} Fail:{failed}")
                    else:
                        failed += 1
                        failed_docs.append((title, error_msg))
                        pbar.set_postfix_str(f"OK:{processed} Fail:{failed}")

                    pbar.update(1)

    # Print summary for this file
    print("\n" + "=" * 60)
    print(f"Processing complete: {json_file.name}")
    print(f"  Total documents: {total}")
    print(f"  Processed:       {processed}")
    print(f"  Skipped:         {skipped}")
    print(f"  Failed:          {failed}")
    print("=" * 60)

    # Print failed documents if any
    if failed_docs:
        print("\nFailed documents:")
        # Show first 3 errors with details (even without debug)
        for i, (title, error) in enumerate(failed_docs):
            if i < 3 or debug:
                print(f"  - {title}")
                print(f"    Error: {error}")
            else:
                print(f"  - {title}")
        
        # Show summary if there are many failures
        if len(failed_docs) > 3 and not debug:
            print(f"\n  ... and {len(failed_docs) - 3} more failures")
            print("  (Use --debug to see all error details)")

    return (total, processed, skipped, failed)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Convert legislative PDFs to Markdown using Gemini Vision API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all documents in a single JSON file
  python gemini_extract.py ce-fortaleza-2024.json

  # Process all JSON files in a directory
  python gemini_extract.py --folder data/

  # Process with force flag to overwrite existing markdown files
  python gemini_extract.py ce-fortaleza-2024.json --force

  # Process only the first 5 documents per file (for testing)
  python gemini_extract.py --folder data/ --max-documents 5

  # Enable debug mode for detailed logging
  python gemini_extract.py ce-fortaleza-2024.json --debug

  # Use more parallel workers for faster processing
  python gemini_extract.py ce-fortaleza-2024.json --workers 40
  
  # Use alternative JSON format (auto-generate PDF paths)
  python gemini_extract.py sp-sao-paulo.json --alt

Environment:
  Create a .env file with: GEMINI_API_KEY=your-key-here
        """
    )

    # Create mutually exclusive group for json_file and folder
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "json_file",
        type=Path,
        nargs="?",
        help="JSON file containing document metadata"
    )
    input_group.add_argument(
        "--folder",
        type=Path,
        help="Folder containing JSON files to process"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess documents even if markdown files already exist"
    )

    parser.add_argument(
        "--max-documents",
        type=int,
        metavar="N",
        help="Maximum number of documents to process per file (useful for testing)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        metavar="N",
        help="Number of parallel workers (default: 20, max ~60 for rate limits)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logging"
    )
    
    parser.add_argument(
        "--alt",
        action="store_true",
        help="Use alternative JSON format (auto-generate PDF paths from metadata)"
    )

    args = parser.parse_args()

    # Configure logging based on debug flag
    if args.debug:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        # In non-debug mode, suppress most logging
        logging.basicConfig(level=logging.WARNING)

    # Get list of JSON files to process
    json_files = []

    if args.folder:
        # Process all JSON files in the folder
        if not args.folder.exists():
            print(f"Error: Folder not found: {args.folder}", file=sys.stderr)
            sys.exit(1)
        if not args.folder.is_dir():
            print(f"Error: Not a directory: {args.folder}", file=sys.stderr)
            sys.exit(1)

        json_files = sorted(args.folder.glob("*.json"))
        if not json_files:
            print(f"Error: No JSON files found in {args.folder}", file=sys.stderr)
            sys.exit(1)

        print(f"\nFound {len(json_files)} JSON files in {args.folder}")
        for jf in json_files:
            print(f"  - {jf.name}")
    else:
        # Process single JSON file
        if not args.json_file.exists():
            print(f"Error: JSON file not found: {args.json_file}", file=sys.stderr)
            sys.exit(1)
        json_files = [args.json_file]

    # Initialize Gemini client
    client = get_gemini_client()

    # Process all JSON files
    overall_stats = {
        "total": 0,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
    }

    for json_file in json_files:
        total, processed, skipped, failed = process_json_file(
            json_file, client, args.force, args.max_documents, args.workers, args.debug, alt_format=args.alt
        )
        overall_stats["total"] += total
        overall_stats["processed"] += processed
        overall_stats["skipped"] += skipped
        overall_stats["failed"] += failed

    # Print overall summary if processing multiple files
    if len(json_files) > 1:
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print(f"  Files processed: {len(json_files)}")
        print(f"  Total documents: {overall_stats['total']}")
        print(f"  Processed:       {overall_stats['processed']}")
        print(f"  Skipped:         {overall_stats['skipped']}")
        print(f"  Failed:          {overall_stats['failed']}")
        print("=" * 60)

    # Exit with error code if any failed
    if overall_stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
