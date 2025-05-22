"""
PDF Parser for APEGA.
Extracts text, tables, and structure from PDF documents using PyMuPDF and PyMuPDF4LLM.
"""

import fitz  # PyMuPDF
import pymupdf4llm # New library for robust PDF to Markdown conversion
import re
from typing import List, Dict, Any, Tuple, Optional
import os
from loguru import logger

from src.models.data_models import ParsedDocument, StructuredTable, DocumentType


class PdfParser:
    """
    Extracts text, tables, and structural metadata from PDF documents using PyMuPDF4LLM.
    """
    
    def __init__(self, use_ocr: bool = False, ocr_language: str = 'eng'): # OCR params might be less relevant now
        """
        Initialize the PDF parser.
        
        Args:
            use_ocr: Whether to use OCR (Note: PyMuPDF4LLM handles underlying text extraction)
            ocr_language: Language for OCR (Note: PyMuPDF4LLM handles underlying text extraction)
        """
        self.use_ocr = use_ocr
        self.ocr_language = ocr_language
        # Note: With PyMuPDF4LLM, direct OCR control here might be less critical
        # as it handles text extraction comprehensively.

    def parse_pdf(self, pdf_path: str, document_id: str, temp_dir: Optional[str] = None) -> ParsedDocument:
        """
        Parse a PDF document using PyMuPDF4LLM to extract content as Markdown.

        Args:
            pdf_path: Path to the PDF file.
            document_id: A unique identifier for the document.
            temp_dir: Optional temporary directory for intermediate files (less used with PyMuPDF4LLM's direct conversion).

        Returns:
            A ParsedDocument object containing the Markdown content and basic metadata.
        """
        logger.info(f"Starting PDF parsing with PyMuPDF4LLM for: {pdf_path}")
        try:
            # PyMuPDF4LLM's to_markdown function is the core of the new parsing
            # It converts the entire PDF to a single Markdown string.
            # page_chunks=True can be useful if we later want to process page by page, 
            # but for now, a single markdown string is the target.
            markdown_output = pymupdf4llm.to_markdown(pdf_path, page_chunks=False) # Keep as single markdown string for now
            
            doc = fitz.open(pdf_path) # Still open with fitz for metadata
            metadata = self._extract_metadata(doc, pdf_path)
            doc.close()

            # With PyMuPDF4LLM, tables are part of the markdown_output.
            # The StructuredTable model might need to be re-evaluated or populated differently
            # if we want to keep it. For now, tables will be an empty list in ParsedDocument.
            tables: List[StructuredTable] = [] 
            
            # TOC extraction might still be useful for document structure, but the primary
            # content is now the markdown string.
            toc = self._extract_toc(fitz.open(pdf_path)) # Use a new fitz instance for TOC
            
            logger.success(f"Successfully parsed PDF with PyMuPDF4LLM: {pdf_path}")
            return ParsedDocument(
                document_id=document_id,
                source_path=pdf_path,
                document_type=DocumentType.PDF,
                text_content=markdown_output, # Main content is the full Markdown
                raw_text_content=markdown_output, # For consistency, can be the same or more raw if needed
                tables=tables, # Empty for now, as tables are in Markdown
                images=[], # Image extraction not implemented with this PyMuPDF4LLM flow yet
                metadata=metadata,
                table_of_contents=toc,
                page_count=len(fitz.open(pdf_path)) # Get page count from a new fitz instance
            )

        except Exception as e:
            logger.error(f"PyMuPDF4LLM failed to parse PDF {pdf_path}: {e}")
            # Return a minimal ParsedDocument on failure to prevent downstream crashes
            # Ensure all fields have defaults or are Optional in ParsedDocument
            return ParsedDocument(
                document_id=document_id,
                source_path=pdf_path,
                document_type=DocumentType.PDF,
                text_content="", # Default to empty string
                raw_text_content="",
                tables=[],
                images=[],
                metadata={"title": os.path.basename(pdf_path), "error": str(e)},
                table_of_contents=[],
                page_count=0
            )

    def _extract_metadata(self, pdf_document: fitz.Document, pdf_path: str) -> Dict[str, Any]:
        """
        Extract metadata from the PDF document.
        Args:
            pdf_document: PyMuPDF document object
            pdf_path: Path to the PDF file
        Returns:
            Dictionary of metadata
        """
        metadata = pdf_document.metadata
        # Ensure basic metadata is present, provide defaults if not
        return {
            "title": metadata.get("title") or os.path.basename(pdf_path),
            "author": metadata.get("author") or "Unknown",
            "subject": metadata.get("subject") or "Unknown",
            "producer": metadata.get("producer") or "Unknown",
            "creationDate": metadata.get("creationDate") or "Unknown",
            "modDate": metadata.get("modDate") or "Unknown",
            "page_count": pdf_document.page_count,
            "encryption": metadata.get("encryption") or "None"
        }

    def _extract_toc(self, pdf_document: fitz.Document) -> List[Dict[str, Any]]:
        """
        Extract the table of contents (TOC) from the PDF document.
        
        Args:
            pdf_document: PyMuPDF document object
            
        Returns:
            List of TOC entries with title, page number, and level
        """
        toc = []
        raw_toc = [] # Initialize raw_toc
        try:
            # get_toc() returns a list of lists: [lvl, title, page, pos]
            # where pos is a Point. For simplicity, we mostly care about lvl, title, page.
            raw_toc = pdf_document.get_toc(simple=True) 
        except Exception as e:
            logger.warning(f"Could not extract TOC for PDF '{pdf_document.metadata.get('title', 'unknown PDF')}' (path: {pdf_document.name}): {e}")
            return toc # Return empty toc if extraction fails

        if not raw_toc:
            logger.info(f"No TOC found or extracted for PDF '{pdf_document.metadata.get('title', 'unknown PDF')}' (path: {pdf_document.name})")
            return toc

        for item in raw_toc:
            try:
                # Defensive access to item elements
                level = item[0] if len(item) > 0 else 0
                title = str(item[1]) if len(item) > 1 and item[1] is not None else "Untitled Section"
                page_num = int(item[2]) if len(item) > 2 and item[2] is not None else 0
                # PyMuPDF's get_toc page numbers are 1-based. 
                # No adjustment needed if downstream expects 1-based.

                toc.append({
                    "level": level,
                    "title": title.strip(),
                    "page": page_num
                })
            except IndexError as ie:
                logger.warning(f"Skipping malformed TOC item (IndexError: {ie}) in '{pdf_document.metadata.get('title', 'unknown PDF')}': {item}")
            except (TypeError, ValueError) as te:
                logger.warning(f"Skipping malformed TOC item (ConversionError: {te}) in '{pdf_document.metadata.get('title', 'unknown PDF')}': {item}")    
            except Exception as ex:
                logger.warning(f"Skipping malformed TOC item (Unexpected Error: {ex}) in '{pdf_document.metadata.get('title', 'unknown PDF')}': {item}")
        
        pdf_document.close() # Close the document after TOC extraction
        return toc

    # Methods like _extract_tables_from_page, _process_page_text, _visual_debug, etc. 
    # from the old parser are no longer directly used as PyMuPDF4LLM handles this internally.
    # They can be removed or kept if there's a future need for hybrid approaches.

    # Example of a method that might be removed or adapted:
    # def _extract_images_from_page(self, page: fitz.Page, page_num: int, temp_dir: str) -> List[Dict[str, Any]]:
    #     """Extracts images from a single page and saves them."""
    #     images_info = []
    #     # ... logic for image extraction ...
    #     return images_info
