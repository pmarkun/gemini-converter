#!/usr/bin/env python3
"""
Script to export JSON data with full markdown text content

Usage:
    python export_json.py <input_json> [--output OUTPUT] [--compact]
"""

import argparse
import json
import tarfile
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


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


def read_markdown_file(md_path: Path) -> Optional[str]:
    """
    Read markdown file and return its content.
    Returns None if file doesn't exist or can't be read.
    """
    try:
        if not md_path.exists():
            return None
        
        with open(md_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Failed to read {md_path}: {e}", file=sys.stderr)
        return None


def extract_full_text_from_markdown(markdown_content: str) -> str:
    """
    Extract the full text from markdown, skipping the frontmatter and title.
    
    Args:
        markdown_content: Full markdown content including frontmatter
        
    Returns:
        Extracted text content without frontmatter and title
    """
    if not markdown_content:
        return ""
    
    lines = markdown_content.split('\n')
    
    # Skip frontmatter (between --- ... ---)
    in_frontmatter = False
    frontmatter_count = 0
    content_start = 0
    
    for i, line in enumerate(lines):
        if line.strip() == '---':
            frontmatter_count += 1
            if frontmatter_count == 2:
                content_start = i + 1
                break
    
    # Skip empty lines and title (# ...)
    for i in range(content_start, len(lines)):
        line = lines[i].strip()
        if line and not line.startswith('#'):
            content_start = i
            break
    
    # Join the remaining lines
    full_text = '\n'.join(lines[content_start:]).strip()
    return full_text


def convert_document(doc: dict, base_path: Path) -> Optional[dict]:
    """
    Convert a document from input format to output format.
    
    Args:
        doc: Input document dictionary
        base_path: Base path for resolving relative paths
        
    Returns:
        Converted document dictionary or None if markdown not found
    """
    # Get markdown path
    if not doc.get("pdf_files"):
        print(f"Warning: Document '{doc.get('title', 'Unknown')}' has no pdf_files", file=sys.stderr)
        return None
    
    pdf_relative = doc["pdf_files"][0]
    md_relative = pdf_path_to_md_path(pdf_relative)
    md_path = base_path / md_relative
    
    # Read markdown content
    markdown_content = read_markdown_file(md_path)
    if markdown_content is None:
        print(f"Warning: Markdown not found for '{doc.get('title', 'Unknown')}': {md_path}", file=sys.stderr)
        return None
    
    # Extract full text (without frontmatter and title)
    full_text = extract_full_text_from_markdown(markdown_content)
    
    # Build output document
    output_doc = {
        "title": doc.get("title", ""),
        "house": doc.get("house", ""),
        "type": doc.get("type", ""),
        "number": int(doc.get("number", 0)) if doc.get("number") else 0,
        "presentation_date": doc.get("presentation_date", ""),
        "year": int(doc.get("year", 0)) if doc.get("year") else 0,
        "author": doc.get("author", []) if isinstance(doc.get("author"), list) else [doc.get("author", "")],
        "subject": doc.get("subject", ""),
        "full_text": full_text,
        "length": len(full_text),
        "url": doc.get("url", ""),
        "scraped_at": doc.get("scraped_at", ""),
        "metadata": {
            "pdf_files": doc.get("pdf_files", []),
            "file_urls": doc.get("file_urls", []),
            "project_url": doc.get("project_url", ""),
            "uuid": doc.get("uuid", ""),
            "status": doc.get("status", []),
        }
    }
    
    return output_doc


def export_json(input_path: Path, output_path: Path, compact: bool = False):
    """
    Export JSON data in the new format and optionally create tar.gz.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path for output file (without extension)
        compact: Whether to create a tar.gz archive
    """
    # Load input JSON
    print(f"Loading {input_path}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except Exception as e:
        print(f"Error: Failed to load JSON: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not isinstance(input_data, list):
        print("Error: Input JSON must be a list of documents", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(input_data)} documents")
    
    # Get base path for resolving markdown files
    base_path = input_path.parent
    
    # Convert documents
    print("Converting documents...")
    output_items = []
    skipped = 0
    
    for i, doc in enumerate(input_data, 1):
        converted = convert_document(doc, base_path)
        if converted:
            output_items.append(converted)
        else:
            skipped += 1
        
        # Progress indicator
        if i % 10 == 0:
            print(f"  Processed {i}/{len(input_data)} documents...")
    
    print(f"✓ Converted {len(output_items)} documents")
    if skipped > 0:
        print(f"⚠ Skipped {skipped} documents (markdown not found)")
    
    # Build output structure
    output_data = {
        "items": output_items,
        "export_info": {
            "exported_at": datetime.now().isoformat(),
            "total_items": len(output_items),
            "source_file": input_path.name,
        }
    }
    
    # Ensure data/ directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save JSON to data/ directory
    json_output_path = data_dir / output_path.with_suffix('.json').name
    print(f"\nSaving JSON to {json_output_path}...")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    json_size_mb = json_output_path.stat().st_size / (1024 * 1024)
    print(f"✓ JSON saved ({json_size_mb:.2f} MB)")
    
    # Create tar.gz if --compact flag is used
    if compact:
        tar_output_path = data_dir / output_path.with_suffix('.tar.gz').name
        print(f"\nCompressing to {tar_output_path}...")
        
        with tarfile.open(tar_output_path, 'w:gz') as tar:
            tar.add(json_output_path, arcname=json_output_path.name)
        
        tar_size_mb = tar_output_path.stat().st_size / (1024 * 1024)
        compression_ratio = (1 - tar_size_mb / json_size_mb) * 100
        
        print(f"✓ Compressed to {tar_size_mb:.2f} MB (saved {compression_ratio:.1f}%)")
        print(f"\nFiles created:")
        print(f"  - JSON: {json_output_path}")
        print(f"  - Compressed: {tar_output_path}")
    else:
        print(f"\nFile created:")
        print(f"  - JSON: {json_output_path}")
    
    print("\n✓ Export complete!")
    
    # Show statistics
    print("\nStatistics:")
    print(f"  Total documents: {len(output_items)}")
    if output_items:
        total_length = sum(item['length'] for item in output_items)
        avg_length = total_length / len(output_items)
        print(f"  Total text length: {total_length:,} characters")
        print(f"  Average text length: {avg_length:.0f} characters")
        print(f"  Shortest: {min(item['length'] for item in output_items):,} characters")
        print(f"  Longest: {max(item['length'] for item in output_items):,} characters")


def main():
    parser = argparse.ArgumentParser(
        description="Export JSON data with full markdown text (saved to data/ directory)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to data/ directory (JSON only)
  python export_json.py ce-fortaleza-2024.json
  
  # Export with custom output name
  python export_json.py ce-fortaleza-2024.json --output fortaleza-export
  
  # Export and create tar.gz archive
  python export_json.py ce-fortaleza-2024.json --compact
        """
    )
    
    parser.add_argument(
        "input_json",
        type=Path,
        help="Input JSON file containing document metadata"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file name (without extension). Default: <input>-export"
    )
    
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Create a tar.gz archive in addition to JSON"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input_json.exists():
        print(f"Error: Input file not found: {args.input_json}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default: input filename + "-export"
        output_path = Path(f"{args.input_json.stem}-export")
    
    # Export
    export_json(args.input_json, output_path, compact=args.compact)


if __name__ == "__main__":
    main()
