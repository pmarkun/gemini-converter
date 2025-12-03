"""
PDF to Markdown converter with smart text extraction.

This module provides functions to extract text from PDF files using:
1. Direct text extraction (fast, for PDFs with embedded text)
2. OCR fallback (slower, for scanned PDFs or when direct extraction fails)
3. Text quality detection to choose the best method
4. Text cleaning heuristics for better output
"""

from pathlib import Path
from typing import Optional, Tuple
import logging
import re

# Lazy imports to avoid loading heavy dependencies at module import time
_model = None

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str | Path) -> Tuple[str, int]:
    """
    Extract text directly from PDF (without OCR).
    
    Uses block-based extraction with proper sorting to preserve reading order.
    Blocks are sorted by vertical position (y) first, then horizontal (x).
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (extracted_text, page_count)
    """
    import fitz  # PyMuPDF
    
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    
    page_count = len(doc)
    extracted_pages = []
    
    for page_num in range(page_count):
        page = doc[page_num]
        
        # Get text blocks with position information
        # Format: (x0, y0, x1, y1, "text", block_type, block_no)
        blocks = page.get_text("blocks")
        
        # Sort blocks by position: y-coordinate first (top to bottom), then x (left to right)
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
        
        # Extract text from blocks
        page_text = "\n".join(block[4].strip() for block in blocks if block[4].strip())
        
        if page_text:  # Only add non-empty pages
            extracted_pages.append(page_text)
    
    doc.close()
    
    # Join pages with page separator
    full_text = "\n\n---\n\n".join(extracted_pages)
    return full_text, page_count


def is_text_rubbish(text: str, min_words: int = 50) -> bool:
    """
    Detect if extracted text is rubbish/gibberish from corrupted PDF text layers.
    
    Strategy: Be strict - only pass documents with near-perfect text quality.
    OCR is fast enough (~2s/doc) that we prefer false positives (send good docs to OCR)
    over false negatives (accept corrupted text).
    
    Heuristics:
    1. Too short (less than min_words)
    2. ANY mixed case within words (OCR corruption indicator)
    3. Unusual characters, curly braces, or special symbols
    4. Low ratio of valid Portuguese words
    5. Misspelled common words (strong indicator of corruption)
    6. Excessive non-ASCII or special characters
    
    Args:
        text: Text to analyze
        min_words: Minimum number of words expected
        
    Returns:
        True if text appears to be rubbish, False otherwise
    """
    if not text or len(text.strip()) < 50:
        return True
    
    # Count words (sequences of letters)
    words = re.findall(r'\b[a-zA-ZÀ-ÿ]+\b', text)
    
    # Check 1: Too few words
    if len(words) < min_words:
        logger.debug(f"Rubbish check: Only {len(words)} words found (minimum: {min_words})")
        return True
    
    # Check 2: ANY mixed case within words is suspicious
    # Examples: "PRoJEro", "LEr", "oRDrNÁRrA", "FoRTALEZA", "MUNICIPAT"
    mixed_case_words = 0
    for word in words:
        if len(word) >= 4:  # Check words of 4+ chars
            # Count transitions between upper and lower case
            transitions = sum(1 for i in range(len(word)-1) 
                            if word[i].isupper() != word[i+1].isupper())
            if transitions >= 2:  # 2+ transitions = suspicious
                mixed_case_words += 1
    
    mixed_case_ratio = mixed_case_words / len(words) if words else 0
    if mixed_case_ratio > 0.05:  # More than 5% words with mixed case
        logger.debug(f"Rubbish check: {mixed_case_ratio:.1%} words with mixed case (threshold: 5%)")
        return True
    
    # Check 3: Unusual characters - expanded to catch more corruption
    # Examples: "00{}Í/", "RAIMI]I\IDO", "fi'l", "Ü"
    unusual_char_pattern = r'[\\|\[\]§{}]'  # Added curly braces
    all_tokens = re.findall(r'\S+', text)
    tokens_with_unusual = sum(1 for t in all_tokens if re.search(unusual_char_pattern, t))
    unusual_ratio = tokens_with_unusual / len(all_tokens) if all_tokens else 0
    if unusual_ratio > 0.002:  # More than 0.2% tokens (tightened from 0.5%)
        logger.debug(f"Rubbish check: {unusual_ratio:.2%} tokens with unusual chars (threshold: 0.2%)")
        return True
    
    # Check 4: Detect common misspellings in legislative documents
    # These specific words should be spelled correctly if PDF text is clean
    misspelling_patterns = [
        (r'\bPRoJE[rR]o\b', 'PROJETO'),  # PRoJEro, PRoJERo
        (r'\bLE[rR]\b', 'LEI'),  # LEr
        (r'\boRDrNÁRrA\b', 'ORDINÁRIA'),  # oRDrNÁRrA
        (r'\bMUNICIPAT\b', 'MUNICIPAL'),  # MUNICIPAT
        (r'\bFORTATEZA\b', 'FORTALEZA'),  # FORTATEZA
        (r'\bCÂMARAÂ\b', 'CÂMARA'),  # CÂMARÂ
        (r'\bMUMCIPAL\b', 'MUNICIPAL'),  # MUMCIPAL
        (r'\blnstitui\b', 'Institui'),  # lnstitui (lowercase L)
        (r'\blnclui\b', 'Inclui'),  # lnclui (lowercase L)
    ]
    
    misspelling_count = 0
    for pattern, correct in misspelling_patterns:
        if re.search(pattern, text):
            misspelling_count += 1
            logger.debug(f"Rubbish check: Found misspelling pattern '{pattern}' (should be '{correct}')")
    
    if misspelling_count > 0:  # ANY misspelling is a red flag
        return True
    
    # Check 5: Look for correctly spelled Portuguese words
    # We need substantial words to be spelled correctly
    valid_portuguese_words = [
        # Common verbs (conjugated forms) - must be exact match
        'fica', 'situada', 'entra', 'vigor', 'publicação', 'aprova', 'denomina',
        # Legislative terms - case insensitive
        'câmara', 'municipal', 'fortaleza', 'projeto', 'lei', 'decreto', 'artigo',
        'legislativo', 'ordinária', 'vereador', 'gabinete', 'departamento',
        # Common nouns
        'cidade', 'estado', 'rua', 'bairro', 'data', 'disposições', 'contrário',
        # Adjectives
        'presente', 'seguinte', 'anterior', 'nacional', 'federal', 'estadual'
    ]
    
    text_lower = text.lower()
    valid_word_matches = sum(1 for word in valid_portuguese_words if word in text_lower)
    
    # Require more valid words (tightened from 3 to 5)
    if valid_word_matches < 5:
        logger.debug(f"Rubbish check: Only {valid_word_matches} valid Portuguese words found (need 5)")
        return True
    
    # Check 6: Ratio of properly formatted words in a sample
    # Sample medium-length words and check if they look reasonable
    medium_words = [w for w in words if 5 <= len(w) <= 12]
    if medium_words:
        sample_size = min(30, len(medium_words))
        sample = medium_words[:sample_size]
        
        # Count words that have reasonable letter patterns
        reasonable_words = 0
        for word in sample:
            lower_count = sum(1 for c in word if c.islower())
            upper_count = sum(1 for c in word if c.isupper())
            total = len(word)
            
            # Word is reasonable if it's mostly one case (tightened to 80%)
            if lower_count > total * 0.8 or upper_count > total * 0.8:
                reasonable_words += 1
            # Or if it's title case (first upper, rest lower)
            elif word[0].isupper() and lower_count > total * 0.8:
                reasonable_words += 1
        
        reasonable_ratio = reasonable_words / sample_size
        if reasonable_ratio < 0.7:  # Less than 70% reasonable (tightened from 50%)
            logger.debug(f"Rubbish check: Only {reasonable_ratio:.1%} words properly formatted (need 70%)")
            return True
    
    # Check 7: Excessive special characters or encoding artifacts
    # Count non-alphanumeric, non-whitespace, non-punctuation characters
    special_chars = re.findall(r'[^\w\s.,;:!?()"\'-áàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ]', text)
    special_ratio = len(special_chars) / len(text) if text else 0
    if special_ratio > 0.02:  # More than 2% special characters
        logger.debug(f"Rubbish check: {special_ratio:.1%} special characters (threshold: 2%)")
        return True
    
    return False


def clean_text(text: str) -> str:
    """
    Clean extracted text using heuristics and Unicode normalization.
    
    Operations:
    1. Fix Unicode encoding issues with ftfy
    2. Fix hyphenation at line breaks
    3. Normalize whitespace (multiple spaces/tabs/newlines)
    4. Remove page numbers in common formats
    5. Remove excessive blank lines
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return text
    
    # 1. Fix Unicode encoding issues (mojibake, encoding errors, etc.)
    try:
        from ftfy import fix_text
        text = fix_text(text)
    except ImportError:
        logger.warning("ftfy not installed, skipping Unicode fixes")
    
    # 2. Fix hyphenation at line breaks (word- \n word -> wordword)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # 3. Normalize whitespace
    # - Collapse multiple spaces/tabs to single space
    text = re.sub(r'[ \t]+', ' ', text)
    # - Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # 4. Remove common page number patterns
    # Examples: "Página 1", "Page 1", "1/10", standalone numbers at line start
    text = re.sub(r'(?i)^página\s+\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^page\s+\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^\d+/\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    
    # 5. Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def get_model():
    """
    Lazy load and return the doctr OCR model configured for Brazilian Portuguese.
    
    Uses fast_base detection architecture and crnn_vgg16_bn recognition architecture.
    The model is cached after first load.
    
    Returns:
        Loaded doctr OCR model optimized for Portuguese text, running on GPU if available
    """
    global _model
    
    if _model is None:
        import torch
        from doctr.models import ocr_predictor, recognition
        
        # Detect device (GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading doctr OCR model (lang=portuguese, device={device})...")
        
        # Create OCR predictor with default multilingual model
        # Note: Portuguese-specific vocab can sometimes perform worse than the default
        _model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
            assume_straight_pages=True,
            preserve_aspect_ratio=True,
            detect_orientation=False,
            detect_language=False,
        )
        
        # Move model to GPU if available
        if device.type == "cuda":
            _model.det_predictor.model = _model.det_predictor.model.to(device)
            _model.reco_predictor.model = _model.reco_predictor.model.to(device)
            logger.info(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")
        
        logger.info("Model loaded successfully")
    return _model


def pdf_to_text_ocr(pdf_path: str | Path) -> str:
    """
    Extract text from a PDF file using doctr OCR (slow fallback method).
    
    This returns raw OCR text without cleaning. Use clean_text() separately if needed.
    
    Args:
        pdf_path: Path to the PDF file to process
        
    Returns:
        Extracted text as a string (not cleaned)
        
    Raises:
        Exception: If OCR processing fails
    """
    logger.info(f"Using OCR for: {pdf_path.name}")
    
    try:
        from doctr.io import DocumentFile
        
        # Load the PDF
        doc = DocumentFile.from_pdf(str(pdf_path))
        
        # Get the OCR model and run prediction
        model = get_model()
        result = model(doc)
        
        # Extract text from the result
        # The result has a hierarchical structure: pages -> blocks -> lines -> words
        extracted_text = []
        
        for page in result.pages:
            page_text = []
            for block in page.blocks:
                block_text = []
                for line in block.lines:
                    # Join words in a line with spaces
                    line_text = " ".join(word.value for word in line.words)
                    block_text.append(line_text)
                # Join lines in a block with newlines
                if block_text:
                    page_text.append("\n".join(block_text))
            
            # Join blocks with double newlines for better readability
            if page_text:
                extracted_text.append("\n\n".join(page_text))
        
        # Join pages with page breaks
        full_text = "\n\n---\n\n".join(extracted_text)
        
        logger.info(f"OCR extracted {len(full_text)} characters from {pdf_path.name}")
        return full_text
        
    except Exception as e:
        logger.error(f"OCR failed for {pdf_path.name}: {e}")
        raise


def try_direct_extraction(pdf_path: str | Path) -> Tuple[Optional[str], bool]:
    """
    Try direct text extraction and check quality.
    
    This is for the first pass of two-pass processing:
    1. Extract text directly (fast)
    2. Check quality
    3. Clean if good quality
    
    Args:
        pdf_path: Path to the PDF file to process
        
    Returns:
        Tuple of (extracted_text or None, needs_ocr)
        - If quality is good: (cleaned_text, False)
        - If quality is bad: (None, True)
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        # Step 1: Try direct text extraction (fast)
        text, page_count = extract_text_from_pdf(pdf_path)
        logger.debug(f"Direct extraction: {len(text)} chars from {page_count} pages")
        
        # Step 2: Check if extracted text is valid
        if is_text_rubbish(text):
            logger.debug(f"Direct extraction failed quality check for {pdf_path.name}")
            return None, True
        
        # Step 3: Clean the text
        text = clean_text(text)
        logger.debug(f"Direct extraction successful for {pdf_path.name} ({len(text)} characters)")
        return text, False
        
    except Exception as e:
        logger.error(f"Failed direct extraction for {pdf_path.name}: {e}")
        # If direct extraction fails completely, mark for OCR
        return None, True


def pdf_to_text(pdf_path: str | Path) -> str:
    """
    Extract text from a PDF file using the best available method.
    
    Strategy:
    1. Try direct text extraction (fast)
    2. Check if extracted text is valid/readable
    3. If text is rubbish, fall back to OCR (slow but accurate)
    4. Clean the extracted text
    
    Args:
        pdf_path: Path to the PDF file to process
        
    Returns:
        Extracted and cleaned text as a string
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If both extraction methods fail
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"Processing PDF: {pdf_path.name}")
    
    try:
        # Step 1: Try direct text extraction (fast)
        text, needs_ocr = try_direct_extraction(pdf_path)
        
        # Step 2: Fall back to OCR if needed
        if needs_ocr:
            logger.info(f"Direct extraction failed quality check, using OCR fallback")
            text = pdf_to_text_ocr(pdf_path)
            text = clean_text(text)
        else:
            logger.info(f"Direct extraction successful ({len(text)} characters)")
        
        logger.info(f"Successfully processed {pdf_path.name} ({len(text)} characters)")
        return text
        
    except Exception as e:
        logger.error(f"Failed to process {pdf_path.name}: {e}")
        raise


def pdf_to_markdown(pdf_path: str | Path, metadata: Optional[dict] = None) -> str:
    """
    Convert a PDF to Markdown format with optional metadata header.
    
    Args:
        pdf_path: Path to the PDF file to process
        metadata: Optional dictionary with document metadata to include in header
                 Expected keys: title, type, number, year, subject, author, 
                 presentation_date, url, house
        
    Returns:
        Formatted Markdown string with metadata and extracted text
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If OCR processing fails
    """
    # Extract text from PDF
    text = pdf_to_text(pdf_path)
    
    # Build markdown document
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


def save_markdown(content: str, output_path: str | Path) -> None:
    """
    Save markdown content to a file, creating directories as needed.
    
    Args:
        content: Markdown content to save
        output_path: Path where the markdown file should be saved
    """
    output_path = Path(output_path)
    
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the file
    output_path.write_text(content, encoding="utf-8")
    logger.info(f"Saved markdown to: {output_path}")
