#!/usr/bin/env python3
"""
Script to update JSON file: remove md_files field since it's now auto-generated from pdf_files.
"""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python update_json.py <json_file>")
        sys.exit(1)
    
    json_file = Path(sys.argv[1])
    
    if not json_file.exists():
        print(f"Error: File not found: {json_file}")
        sys.exit(1)
    
    # Load JSON
    print(f"Loading {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} documents")
    
    # Remove md_files field from all documents
    removed_count = 0
    for doc in data:
        if "md_files" in doc:
            del doc["md_files"]
            removed_count += 1
    
    print(f"Removed md_files field from {removed_count} documents")
    
    # Create backup
    backup_file = json_file.with_suffix('.json.bak')
    print(f"Creating backup: {backup_file}")
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Save updated JSON
    print(f"Saving updated JSON to {json_file}")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("✓ Done!")
    print(f"\nThe md_files field has been removed.")
    print("Markdown paths are now auto-generated from PDF paths:")
    print("  ce-fortaleza/pdf/2024/file.pdf → ce-fortaleza/md/2024/file.md")


if __name__ == "__main__":
    main()
