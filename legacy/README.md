# Legacy OCR Implementation

This directory contains the original local OCR implementation using doctr and PyMuPDF.

## Contents

- **main.py**: Local OCR extraction using doctr (PyTorch)
- **src/pdftomd/converter.py**: Core OCR conversion logic
- **test_performance.py**: Performance testing script
- **update_json.py**: Script to remove md_files field from JSON

## Dependencies (Legacy)

The legacy implementation required:
- python-doctr[torch]>=1.0.0 (~3GB with PyTorch/CUDA)
- pymupdf>=1.24.0
- ftfy>=6.0.0
- tqdm>=4.67.1

## Why Moved to Legacy

The project now uses Gemini Vision API (gemini_extract.py) as the primary extraction method because:
- Simpler setup (no 3GB+ PyTorch dependency)
- Better structured output (ementa, texto, justificativa)
- Easier to maintain
- Cloud-based processing

## Using Legacy OCR

If you need the local OCR implementation:

1. **Install legacy dependencies**:
   ```bash
   uv add "python-doctr[torch]>=1.0.0" "pymupdf>=1.24.0" "ftfy>=6.0.0"
   ```

2. **Run local OCR**:
   ```bash
   uv run python legacy/main.py ce-fortaleza-2024.json
   ```

See the main README.md for full documentation of the legacy OCR features.
