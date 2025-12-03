# Agent Decision Log

This document records important architectural and technical decisions made during the development of the PDF to Markdown conversion tool.

## Project Overview

**Purpose**: Convert legislative PDF documents to Markdown format using OCR, processing metadata from JSON files.

**Tech Stack**:
- Python 3.13
- doctr (Document Text Recognition) for OCR
- PyTorch backend for doctr
- uv for package management

## Key Decisions

### 1. OCR Library Selection: doctr

**Decision**: Use `python-doctr` library for PDF OCR processing.

**Rationale**:
- Modern, actively maintained OCR library
- Built on top of deep learning models (superior accuracy to traditional OCR)
- Native PDF support (no need for PDF to image conversion)
- Good Python API with clean document structure (pages → blocks → lines → words)
- Supports both PyTorch and TensorFlow backends

**Trade-offs**:
- Large dependency footprint (~3GB+ with PyTorch/CUDA)
- Requires significant download time for initial setup
- Heavier than alternatives like pytesseract, but better accuracy

**Alternatives Considered**:
- `pytesseract`: Lighter weight but lower accuracy, requires separate PDF→image conversion
- `pymupdf` + OCR: Would need additional OCR engine integration
- Commercial APIs: Would require internet connectivity and API costs

### 2. Modular Architecture

**Decision**: Separate conversion logic (`src/pdftomd/converter.py`) from CLI logic (`main.py`).

**Rationale**:
- Allows the conversion functions to be reused in other contexts (web services, batch jobs, etc.)
- Easier to test conversion logic independently
- Clean separation of concerns
- CLI can be extended without touching core conversion logic

**Implementation**:
- `converter.py`: Pure functions for PDF processing and markdown generation
- `main.py`: CLI argument parsing, file system operations, workflow orchestration

### 3. Lazy Model Loading

**Decision**: Load the doctr model only when first needed, not at module import time.

**Rationale**:
- Improves startup time for CLI help/validation
- Reduces memory footprint for operations that don't need OCR
- Model loading has significant overhead (neural network initialization)
- Model is cached after first load for subsequent uses

**Implementation**:
```python
_model = None

def get_model():
    global _model
    if _model is None:
        from doctr.models import ocr_predictor
        _model = ocr_predictor(pretrained=True)
    return _model
```

### 4. Markdown Frontmatter Format

**Decision**: Use YAML-style frontmatter for metadata at the top of each markdown file.

**Format**:
```markdown
---
title: Project Title
type: Projeto de Lei Ordinária
number: 1
year: 2024
subject: Description...
author: Author Name
presentation_date: 2024-01-02
url: https://...
house: Câmara Municipal de Fortaleza
---

# Project Title

[OCR extracted content]
```

**Rationale**:
- Standard format used by static site generators (Jekyll, Hugo, etc.)
- Easy to parse programmatically
- Human-readable
- Preserves all important metadata from source JSON
- Supports both single values and lists (for multiple authors)

### 5. Skip Logic with --force Flag

**Decision**: By default, skip documents where markdown file already exists; add `--force` flag to override.

**Rationale**:
- OCR is computationally expensive (can take seconds per page)
- Prevents accidental reprocessing of large document sets
- Allows resuming interrupted batch processing
- `--force` flag provides explicit control when reprocessing is desired

**Implementation**:
- Check if target markdown file exists before processing
- Log skip reason for user visibility
- Track skipped count in summary statistics

### 6. Testing-Friendly CLI Design

**Decision**: Add `--max-documents N` parameter to limit processing during testing.

**Rationale**:
- Enables quick validation without processing entire dataset
- Useful for development and debugging
- Allows progressive testing (5 docs → 50 docs → all docs)
- No need to manually edit JSON or create test fixtures

**Usage**:
```bash
# Test with first 5 documents
python main.py ce-fortaleza-2024.json --max-documents 5
```

### 7. Path Resolution Strategy

**Decision**: Auto-generate markdown paths from PDF paths by replacing `/pdf/` with `/md/` and `.pdf` with `.md`.

**Rationale**:
- Eliminates redundant `md_files` field in JSON (DRY principle)
- Ensures consistent directory structure (pdf and md mirror each other)
- Simplifies JSON maintenance (only one path to manage)
- Easy to understand: markdown path is derived from PDF path
- Makes it impossible for paths to get out of sync

**Implementation**:
```python
def pdf_path_to_md_path(pdf_path: str) -> str:
    md_path = pdf_path.replace("/pdf/", "/md/")
    if md_path.endswith(".pdf"):
        md_path = md_path[:-4] + ".md"
    return md_path

# Example:
# "ce-fortaleza/pdf/2024/projeto-de-lei-ordinária-255-2024.pdf"
# → "ce-fortaleza/md/2024/projeto-de-lei-ordinária-255-2024.md"

base_path = args.json_file.parent
pdf_path = base_path / doc["pdf_files"][0]
md_path = base_path / pdf_path_to_md_path(doc["pdf_files"][0])
```

**Migration**: The `md_files` field was removed from the JSON using `update_json.py` script.

### 8. Error Handling Strategy

**Decision**: Continue processing remaining documents when one fails; report failures in summary.

**Rationale**:
- Individual PDF issues shouldn't halt entire batch
- User gets maximum value from each run
- Clear error logging helps identify problematic files
- Non-zero exit code if any failures (useful for CI/CD)

**Implementation**:
- Try-except around each document processing
- Track success/skip/failure counts
- Log detailed error messages
- Exit with code 1 if any failures

### 9. Text Structure Preservation

**Decision**: Preserve document structure from doctr with blocks separated by double newlines, pages separated by horizontal rules.

**Format**:
- Lines within a block: single newline
- Blocks within a page: double newline (`\n\n`)
- Pages: separated by `\n\n---\n\n`

**Rationale**:
- Maintains reading flow while preserving structure
- Horizontal rules provide visual page breaks
- Balance between preservation and readability
- Works well with markdown rendering

### 10. Logging Strategy

**Decision**: Use Python's `logging` module with INFO level, timestamp, and clear formatting.

**Rationale**:
- Standard Python logging is flexible and familiar
- INFO level provides progress visibility without debug noise
- Timestamps help with performance analysis
- Structured format (timestamp - level - message) is scannable
- Easy to redirect to file or adjust verbosity later

## Future Considerations

### Implemented Enhancements

1. **Performance Optimization**:
   - ✅ Smart extraction strategy (direct + OCR fallback)
   - ✅ GPU acceleration for OCR (when needed)
   - ✅ Text quality detection to avoid unnecessary OCR

### Potential Future Enhancements

1. **Quality Improvements**:
   - Confidence scores for OCR results
   - Post-processing to fix common OCR errors
   - Table detection and markdown table formatting
   - Image extraction and embedding

3. **Flexibility**:
   - Support for other document formats (DOCX, images)
   - Pluggable OCR backends (Tesseract, cloud services)
   - Configuration file for model selection and parameters

4. **Robustness**:
   - Retry logic for transient failures
   - Checkpointing for very large batches
   - Validation of output quality

### Known Limitations

1. **Dependency Size**: Full installation is 3+ GB due to PyTorch/CUDA
2. **Processing Speed**: OCR is CPU/GPU intensive; large batches take time
3. **Accuracy**: OCR quality depends on PDF quality (scans vs. digital PDFs)
4. **Layout**: Complex layouts (multi-column, tables) may not preserve perfectly

### 11. Smart Text Extraction Strategy

**Decision**: Use direct text extraction first, with OCR as fallback based on quality detection.

**Rationale**:
- Most PDFs have embedded text (92% in our dataset)
- Direct extraction is ~9x faster than OCR (0.2s vs 1.8s per document)
- Some PDFs are scanned images and require OCR
- Quality detection prevents using gibberish from corrupted text layers

**Implementation**:
```python
def pdf_to_text(pdf_path):
    # 1. Try direct extraction (PyMuPDF)
    text = extract_text_from_pdf(pdf_path)
    
    # 2. Check if text is valid
    if is_text_rubbish(text):
        # 3. Fall back to OCR (doctr)
        text = pdf_to_text_ocr(pdf_path)
    
    # 4. Clean the text
    return clean_text(text)
```

**Quality Detection Heuristics**:
- Minimum word count threshold
- Ratio of alphabetic to special characters
- Detection of common Portuguese words
- Single-character word ratio (gibberish indicator)
- Special character density

**Text Cleaning Heuristics**:
- Fix hyphenation at line breaks
- Remove page numbers
- Normalize whitespace
- Remove excessive blank lines
- Fix common OCR errors (rn→m, etc.)

**Performance Impact**:
- 92% of documents: 0.2s per document (direct extraction)
- 8% of documents: 1.8s per document (OCR)
- Average: ~0.3s per document (vs ~1.8s with OCR-only approach)
- **6x overall speedup** compared to always using OCR

### 12. Single-Threaded GPU Processing

**Decision**: Use single-threaded sequential processing with GPU acceleration for OCR.

**Rationale**:
- PyTorch is not thread-safe, causing crashes with threading
- Multiprocessing with GPU causes contention and is slower than sequential
- Direct text extraction is so fast that parallelization overhead isn't worth it
- OCR (8% of documents) benefits from GPU but doesn't benefit from parallelization
- Simpler code without complex worker management

**Performance Data**:
- Sequential GPU: 100 documents in 21 seconds
- Average throughput: 4.7 documents/second
- Simple, reliable, no crashes

**Rejected Alternatives**:
- Threading: Crashes due to PyTorch thread-safety issues
- Multiprocessing: Slower due to GPU contention and process spawning overhead
- TensorFlow backend: Would require full dependency change for marginal benefit

### 13. Auto-Generated Markdown Paths

**Decision**: Remove `md_files` field from JSON and auto-generate markdown paths from PDF paths.

**Rationale**:
- **DRY principle**: Eliminates redundant data (one source of truth)
- **Consistency**: Ensures parallel directory structure (`/pdf/` ↔ `/md/`)
- **Maintainability**: Only need to manage PDF paths in JSON
- **Simplicity**: Clear, predictable transformation rule
- **Prevents errors**: Impossible for paths to get out of sync

**Implementation**:
```python
def pdf_path_to_md_path(pdf_path: str) -> str:
    """
    ce-fortaleza/pdf/2024/file.pdf → ce-fortaleza/md/2024/file.md
    """
    md_path = pdf_path.replace("/pdf/", "/md/")
    if md_path.endswith(".pdf"):
        md_path = md_path[:-4] + ".md"
    return md_path
```

**Benefits**:
- Simpler JSON structure (15 fields → 14 fields)
- No possibility of typos in markdown paths
- Easier to understand directory organization
- Less data to maintain in JSON files

**Migration**:
- Created `update_json.py` script to remove `md_files` from existing JSON
- Script creates backup before modifying
- 243 documents updated successfully

### 14. Export Script Output Organization

**Decision**: Save exported files to `data/` directory; create tar.gz only with `--compact` flag.

**Rationale**:
- **Organization**: Keeps generated export files separate from source data
- **Flexibility**: Users can choose whether they want compression
- **Default behavior**: Most users just want the JSON for processing
- **Disk space**: Uncompressed JSON is easier to work with; compression optional for distribution

**Implementation**:
```python
# Default: JSON only in data/
python export_json.py input.json
# Output: data/input-export.json

# With --compact: JSON + tar.gz in data/
python export_json.py input.json --compact
# Output: data/input-export.json
#         data/input-export.tar.gz
```

**Benefits**:
- Clean project structure (all exports in one place)
- Optional compression (75% size reduction when needed)
- Easy to gitignore `data/` directory
- No clutter in project root

**Changes Made**:
- Renamed `--no-compress` flag to `--compact` (more intuitive)
- Changed default to no compression (compression is opt-in)
- All output files saved to `data/` directory
- Auto-creates `data/` directory if it doesn't exist

## Development Timeline

- **Session 1**: Project initialization, dependency setup, architecture planning
- **Session 2**: Implementation of converter module and CLI, documentation
- **Session 3**: GPU acceleration experiments, parallel processing attempts
- **Session 4**: Smart extraction strategy implementation (direct + OCR fallback)
- **Session 5**: Gemini Vision API integration with multithreading support
- **Session 6**: Rate limit optimization (20-60 workers), auto-generated markdown paths, export script implementation
- **Session 7**: Export script refinement (data/ directory, --compact flag)

- **Session 1**: Project initialization, dependency setup, architecture planning
- **Session 2**: Implementation of converter module and CLI, documentation
- **Session 3**: GPU acceleration experiments, parallel processing attempts
- **Session 4**: Smart extraction strategy implementation (direct + OCR fallback)
- **Session 5**: Gemini Vision API integration with multithreading support
- **Session 6**: Rate limit optimization (20-60 workers), auto-generated markdown paths

## References

- [doctr Documentation](https://github.com/mindee/doctr)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [YAML Frontmatter Spec](https://jekyllrb.com/docs/front-matter/)
