#!/usr/bin/env python3
"""Test script to measure OCR processing time and analyze parallelization benefits."""

import time
from pathlib import Path
from src.pdftomd.converter import get_model
from doctr.io import DocumentFile

# Test with one document
json_path = Path("ce-fortaleza-2024.json")
import json
with open(json_path) as f:
    docs = json.load(f)

# Get first document with PDF
doc = docs[0]
pdf_path = json_path.parent / doc["pdf_files"][0]

print(f"Testing with: {pdf_path.name}")
print(f"PDF exists: {pdf_path.exists()}")
print()

# Load model once
print("Loading model...")
start = time.time()
model = get_model()
load_time = time.time() - start
print(f"Model load time: {load_time:.2f}s")
print()

# Test processing time for one document
print("Processing document...")
start = time.time()
doc_file = DocumentFile.from_pdf(str(pdf_path))
result = model(doc_file)
process_time = time.time() - start
print(f"Processing time: {process_time:.2f}s")
print(f"Number of pages: {len(doc_file)}")
print(f"Time per page: {process_time/len(doc_file):.2f}s")
print()

# Estimate for 243 documents (assuming average 3 pages each)
total_docs = 243
avg_pages = 3
estimated_time = total_docs * avg_pages * (process_time/len(doc_file))
print(f"Estimated time for {total_docs} documents:")
print(f"  Sequential: {estimated_time/60:.1f} minutes ({estimated_time/3600:.1f} hours)")
print(f"  With 4 processes: {estimated_time/4/60:.1f} minutes")
print(f"  With 8 processes: {estimated_time/8/60:.1f} minutes")
