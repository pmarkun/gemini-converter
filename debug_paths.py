#!/usr/bin/env python3
"""
Debug script to check PDF path generation for alternative format.
"""

import json
import sys
from pathlib import Path


def generate_pdf_path_from_metadata(doc: dict) -> str:
    """Generate PDF path from document metadata for alternative format."""
    doc_type = doc.get("type", "").lower()
    doc_number = doc.get("number", "")
    doc_year = doc.get("year", "")
    
    return f"pdf/{doc_type}-{doc_number}-{doc_year}.pdf"


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_paths.py <json_file>")
        sys.exit(1)
    
    json_file = Path(sys.argv[1])
    base_path = json_file.parent
    
    if not json_file.exists():
        print(f"Error: JSON file not found: {json_file}")
        sys.exit(1)
    
    # Load JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    
    print(f"Checking first 10 documents from {json_file.name}:")
    print("=" * 80)
    
    for i, doc in enumerate(docs[:10]):
        title = doc.get("title", "Unknown")
        doc_type = doc.get("type", "")
        doc_number = doc.get("number", "")
        doc_year = doc.get("year", "")
        
        # Generate path
        pdf_path = generate_pdf_path_from_metadata(doc)
        full_path = base_path / pdf_path
        
        exists = "✓ EXISTS" if full_path.exists() else "✗ NOT FOUND"
        
        print(f"\n{i+1}. {title}")
        print(f"   Type: {doc_type!r}, Number: {doc_number!r}, Year: {doc_year!r}")
        print(f"   Generated path: {pdf_path}")
        print(f"   Full path: {full_path}")
        print(f"   Status: {exists}")
    
    # Check if pdf_files field exists
    has_pdf_files = any("pdf_files" in doc for doc in docs[:10])
    if has_pdf_files:
        print("\n" + "=" * 80)
        print("WARNING: Some documents have 'pdf_files' field!")
        print("You may not need --alt flag for this file.")
        print("\nExample with pdf_files:")
        for doc in docs[:3]:
            if "pdf_files" in doc:
                print(f"  {doc.get('title')}: {doc['pdf_files']}")
                break


if __name__ == "__main__":
    main()
