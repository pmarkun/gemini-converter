#!/usr/bin/env python3
"""
CLI tool to process JSON files containing legislative document metadata
and generate Markdown files from PDFs using OCR.

Usage:
    python main.py <json_file> [--force] [--max-documents N] [--debug]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.pdftomd.converter import (
    try_direct_extraction,
    pdf_to_text_ocr,
    clean_text,
    save_markdown,
)

# Logger will be configured after parsing arguments
logger = logging.getLogger(__name__)


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


def should_process_document(doc: dict, force: bool, base_path: Path) -> tuple[bool, Optional[str]]:
    """
    Determine if a document should be processed.
    
    Returns:
        Tuple of (should_process, reason)
    """
    # Check if document has required fields
    if not doc.get("pdf_files"):
        return False, "No PDF files specified"
    
    # Generate markdown path from PDF path
    pdf_relative = doc["pdf_files"][0]
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


def show_text_preview(text: str, max_chars: int = 300):
    """Show a preview of extracted text for debugging."""
    if not text:
        logger.info("    [No text extracted]")
        return
    
    preview = text[:max_chars].replace('\n', ' ')
    if len(text) > max_chars:
        preview += "..."
    logger.info(f"    Preview: {preview}")


def process_document_pass1(doc: dict, base_path: Path, debug: bool = False) -> tuple[str, Optional[str]]:
    """
    Pass 1: Try direct extraction, check quality, save if good.
    
    Returns:
        Tuple of (status, error_message)
        Status can be: "success", "needs_ocr", "error", "pdf_not_found"
    """
    try:
        # Get PDF path (use first PDF if multiple)
        pdf_relative = doc["pdf_files"][0]
        pdf_path = base_path / pdf_relative
        
        # Generate markdown output path from PDF path
        md_relative = pdf_path_to_md_path(pdf_relative)
        md_path = base_path / md_relative
        
        if debug:
            logger.info(f"  PDF: {pdf_path.name}")
        
        # Check if PDF exists
        if not pdf_path.exists():
            if debug:
                logger.error(f"    PDF file not found")
            return "pdf_not_found", f"PDF file not found: {pdf_path}"
        
        # Try direct extraction with quality check
        text, needs_ocr = try_direct_extraction(pdf_path)
        
        if needs_ocr:
            if debug:
                logger.info(f"    Direct extraction failed quality check - marking for OCR")
            return "needs_ocr", None
        
        # Text is good quality, build and save markdown
        if debug:
            logger.info(f"    Direct extraction successful ({len(text)} chars)")
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
        return "success", None
        
    except Exception as e:
        error_msg = str(e)
        if debug:
            logger.error(f"    ✗ Error: {error_msg}")
        return "error", error_msg


def process_document_pass2(doc: dict, base_path: Path, debug: bool = False) -> tuple[bool, str]:
    """
    Pass 2: Use OCR to extract text and save.
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Get PDF path (use first PDF if multiple)
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
        
        # Use OCR to extract text
        text = pdf_to_text_ocr(pdf_path)
        text = clean_text(text)
        
        if debug:
            logger.info(f"    OCR successful ({len(text)} chars)")
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
            logger.error(f"    ✗ OCR failed: {error_msg}")
        return False, error_msg


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Convert legislative PDFs to Markdown using OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all documents in the JSON file
  python main.py ce-fortaleza-2024.json
  
  # Process with force flag to overwrite existing markdown files
  python main.py ce-fortaleza-2024.json --force
  
  # Process only the first 5 documents (for testing)
  python main.py ce-fortaleza-2024.json --max-documents 5
  
  # Enable debug mode for detailed logging
  python main.py ce-fortaleza-2024.json --debug
        """
    )
    
    parser.add_argument(
        "json_file",
        type=Path,
        help="JSON file containing document metadata"
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
        help="Maximum number of documents to process (useful for testing)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging based on debug flag
    if args.debug:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        # Also configure doctr logger
        logging.getLogger("doctr").setLevel(logging.INFO)
    else:
        # In non-debug mode, suppress most logging
        logging.basicConfig(level=logging.WARNING)
        # Suppress doctr download progress bars
        logging.getLogger("doctr").setLevel(logging.WARNING)
    
    # Validate JSON file exists
    if not args.json_file.exists():
        print(f"Error: JSON file not found: {args.json_file}", file=sys.stderr)
        sys.exit(1)
    
    # Get base path (directory containing the JSON file)
    base_path = args.json_file.parent
    
    # Load documents from JSON
    documents = load_json(args.json_file, debug=args.debug)
    
    # Apply max-documents limit if specified
    if args.max_documents:
        documents = documents[:args.max_documents]
        if args.debug:
            logger.info(f"Limited to processing {len(documents)} documents")
    
    # Two-pass processing strategy
    # Pass 1: Try direct extraction on all documents, save good ones, track failed
    # Pass 2: OCR only the documents that failed quality check in pass 1
    
    total = len(documents)
    processed = 0
    skipped = 0
    needs_ocr = 0
    failed = 0
    failed_docs = []
    ocr_docs = []  # Documents that need OCR in pass 2
    
    # ============================================================================
    # PASS 1: Direct extraction with quality checks
    # ============================================================================
    
    if args.debug:
        logger.info(f"\n{'='*60}")
        logger.info(f"PASS 1: Direct text extraction")
        logger.info(f"{'='*60}\n")
        
        # Debug mode: detailed logging
        for i, doc in enumerate(documents, 1):
            logger.info(f"[{i}/{total}] {doc.get('title', 'Unknown')}")
            
            # Check if we should process this document
            should_process, reason = should_process_document(doc, args.force, base_path)
            
            if not should_process:
                logger.info(f"  Skipping: {reason}")
                skipped += 1
                logger.info("")
                continue
            
            # Try direct extraction (Pass 1)
            status, error_msg = process_document_pass1(doc, base_path, debug=True)
            
            if status == "success":
                processed += 1
            elif status == "needs_ocr":
                needs_ocr += 1
                ocr_docs.append(doc)
            else:  # error or pdf_not_found
                failed += 1
                failed_docs.append((doc.get("title", "Unknown"), error_msg))
            
            logger.info("")  # Empty line for readability
        
        logger.info(f"Pass 1 complete: {processed} direct, {needs_ocr} need OCR, {skipped} skipped, {failed} failed\n")
    else:
        # Normal mode: progress bar
        print(f"\nPass 1: Direct text extraction ({total} documents)...")
        
        with tqdm(total=total, unit="doc", ncols=80, desc="Pass 1") as pbar:
            for doc in documents:
                # Check if we should process this document
                should_process, reason = should_process_document(doc, args.force, base_path)
                
                if not should_process:
                    skipped += 1
                    pbar.set_postfix_str(f"OK:{processed} OCR:{needs_ocr} Skip:{skipped}")
                    pbar.update(1)
                    continue
                
                # Try direct extraction (Pass 1)
                status, error_msg = process_document_pass1(doc, base_path, debug=False)
                
                if status == "success":
                    processed += 1
                    pbar.set_postfix_str(f"OK:{processed} OCR:{needs_ocr} Skip:{skipped}")
                elif status == "needs_ocr":
                    needs_ocr += 1
                    ocr_docs.append(doc)
                    pbar.set_postfix_str(f"OK:{processed} OCR:{needs_ocr} Skip:{skipped}")
                else:  # error or pdf_not_found
                    failed += 1
                    failed_docs.append((doc.get("title", "Unknown"), error_msg))
                    pbar.set_postfix_str(f"OK:{processed} OCR:{needs_ocr} Skip:{skipped} Fail:{failed}")
                
                pbar.update(1)
    
    # ============================================================================
    # PASS 2: OCR for documents that failed quality check
    # ============================================================================
    
    if ocr_docs:
        if args.debug:
            logger.info(f"\n{'='*60}")
            logger.info(f"PASS 2: OCR processing ({len(ocr_docs)} documents)")
            logger.info(f"{'='*60}\n")
            
            # Debug mode: detailed logging
            for i, doc in enumerate(ocr_docs, 1):
                logger.info(f"[{i}/{len(ocr_docs)}] {doc.get('title', 'Unknown')}")
                
                # Process with OCR (Pass 2)
                success, error_msg = process_document_pass2(doc, base_path, debug=True)
                
                if success:
                    processed += 1
                else:
                    failed += 1
                    failed_docs.append((doc.get("title", "Unknown"), error_msg))
                
                logger.info("")  # Empty line for readability
        else:
            # Normal mode: progress bar
            print(f"\nPass 2: OCR processing ({len(ocr_docs)} documents)...")
            
            with tqdm(total=len(ocr_docs), unit="doc", ncols=80, desc="Pass 2") as pbar:
                for doc in ocr_docs:
                    # Process with OCR (Pass 2)
                    success, error_msg = process_document_pass2(doc, base_path, debug=False)
                    
                    if success:
                        processed += 1
                        pbar.set_postfix_str(f"OK:{processed-needs_ocr+pbar.n} Fail:{failed}")
                    else:
                        failed += 1
                        failed_docs.append((doc.get("title", "Unknown"), error_msg))
                        pbar.set_postfix_str(f"OK:{processed-needs_ocr+pbar.n} Fail:{failed}")
                    
                    pbar.update(1)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"  Total documents: {total}")
    print(f"  Processed:       {processed}")
    print(f"  Skipped:         {skipped}")
    print(f"  Failed:          {failed}")
    print("=" * 60)
    
    # Print failed documents if any
    if failed_docs:
        print("\nFailed documents:")
        for title, error in failed_docs:
            print(f"  - {title}")
            if args.debug:
                print(f"    Error: {error}")
    
    # Exit with error code if any failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
