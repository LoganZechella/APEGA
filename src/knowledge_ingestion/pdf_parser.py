"""
PDF Parser for APEGA.
Extracts text, tables, and structure from PDF documents using PyMuPDF (fitz).
"""

import fitz  # PyMuPDF
import re
from typing import List, Dict, Any, Tuple, Optional
import os
from loguru import logger
from collections import defaultdict

from src.models.data_models import ParsedDocument, StructuredTable, DocumentType


class PdfParser:
    """
    Extracts text, tables, and structural metadata from PDF documents using PyMuPDF.
    Preserves layout information as much as possible.
    """
    
    def __init__(self, use_ocr: bool = False, ocr_language: str = 'eng'):
        """
        Initialize the PDF parser.
        
        Args:
            use_ocr: Whether to use OCR for text extraction
            ocr_language: Language for OCR (only used if use_ocr is True)
        """
        self.use_ocr = use_ocr
        self.ocr_language = ocr_language
        
        # Only import pytesseract if OCR is enabled
        if self.use_ocr:
            try:
                import pytesseract
                self.pytesseract = pytesseract
            except ImportError:
                logger.warning("pytesseract not installed. OCR will not be available.")
                self.use_ocr = False
    
    def parse_pdf(self, file_path: str) -> ParsedDocument:
        """
        Parse a PDF file to extract text, tables, and structure.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            A ParsedDocument object containing the extracted content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        document_id = os.path.basename(file_path).replace('.pdf', '')
        
        try:
            # Open the PDF document
            pdf_document = fitz.open(file_path)
            
            # Extract document metadata
            metadata = self._extract_metadata(pdf_document)
            
            # Extract text content with layout preservation
            text_content, tables = self._extract_text_and_tables(pdf_document)
            
            # Create ParsedDocument
            parsed_doc = ParsedDocument(
                document_id=document_id,
                title=metadata.get('title', document_id),
                source_path=file_path,
                document_type=DocumentType.PDF,
                text_content=text_content,
                tables=tables,
                metadata=metadata
            )
            
            pdf_document.close()
            return parsed_doc
            
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            # Return a minimal document with error information
            return ParsedDocument(
                document_id=document_id,
                source_path=file_path,
                document_type=DocumentType.PDF,
                text_content="",
                metadata={"error": str(e)}
            )
    
    def _extract_metadata(self, pdf_document: fitz.Document) -> Dict[str, Any]:
        """
        Extract metadata from the PDF document.
        
        Args:
            pdf_document: PyMuPDF document object
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            'title': pdf_document.metadata.get('title', ''),
            'author': pdf_document.metadata.get('author', ''),
            'subject': pdf_document.metadata.get('subject', ''),
            'keywords': pdf_document.metadata.get('keywords', ''),
            'creator': pdf_document.metadata.get('creator', ''),
            'producer': pdf_document.metadata.get('producer', ''),
            'page_count': len(pdf_document),
            'toc': self._extract_toc(pdf_document),
        }
        return metadata
    
    def _extract_toc(self, pdf_document: fitz.Document) -> List[Dict[str, Any]]:
        """
        Extract the table of contents (TOC) from the PDF document.
        
        Args:
            pdf_document: PyMuPDF document object
            
        Returns:
            List of TOC entries with title, page number, and level
        """
        toc = []
        for item in pdf_document.get_toc():
            if len(item) >= 3:
                level, title, page = item[:3]
                toc.append({
                    'level': level,
                    'title': title,
                    'page': page
                })
        return toc
    
    def _extract_text_and_tables(self, pdf_document: fitz.Document) -> Tuple[str, List[StructuredTable]]:
        """
        Extract text and tables from the PDF document.
        
        Args:
            pdf_document: PyMuPDF document object
            
        Returns:
            Tuple of (text_content, tables)
        """
        full_text = []
        tables = []
        table_counter = 0
        
        for page_idx, page in enumerate(pdf_document):
            page_number = page_idx + 1
            
            # Extract text with layout preservation
            if self.use_ocr:
                # Use OCR for text extraction
                text = self._extract_text_with_ocr(page)
            else:
                # Use PyMuPDF's built-in text extraction
                text = page.get_text("text")
            
            # Add page information
            page_text = f"PAGE {page_number}\n{text}\n"
            full_text.append(page_text)
            
            # Extract tables
            page_tables = self._extract_tables_from_page(page, page_number, table_counter)
            tables.extend(page_tables)
            table_counter += len(page_tables)
        
        return "\n".join(full_text), tables
    
    def _extract_text_with_ocr(self, page: fitz.Page) -> str:
        """
        Extract text from a page using OCR.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Extracted text
        """
        if not hasattr(self, 'pytesseract'):
            logger.warning("OCR requested but pytesseract is not available.")
            return page.get_text("text")
        
        try:
            # Render page to an image (higher resolution for better OCR)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            
            # Convert to PIL Image
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            
            # Run OCR
            ocr_text = self.pytesseract.image_to_string(img, lang=self.ocr_language)
            return ocr_text
            
        except Exception as e:
            logger.error(f"OCR error: {e}. Falling back to standard text extraction.")
            return page.get_text("text")
    
    def _extract_tables_from_page(self, page: fitz.Page, page_number: int, table_counter: int) -> List[StructuredTable]:
        """
        Extract tables from a page.
        
        Args:
            page: PyMuPDF page object
            page_number: Page number
            table_counter: Starting counter for table IDs
            
        Returns:
            List of StructuredTable objects
        """
        tables = []
        
        # Use PyMuPDF's table detection
        tab = page.find_tables()
        
        if tab.tables:
            for i, table in enumerate(tab.tables):
                table_id = f"table_{page_number}_{i+1}"
                
                # Extract data from the table
                rows = []
                for row_cells in table.extract():
                    # Clean cell text
                    rows.append([self._clean_text(cell.replace("\n", " ").strip()) for cell in row_cells])
                
                # Only include non-empty tables
                if rows and any(cell for row in rows for cell in row):
                    # Try to determine if the first row is a header
                    caption = None
                    if len(rows) > 1:
                        first_row_text = " ".join(rows[0])
                        if re.search(r"table|figure|fig\.?|tab\.?", first_row_text.lower()):
                            caption = first_row_text
                    
                    structured_table = StructuredTable(
                        table_id=table_id,
                        data=rows,
                        caption=caption,
                        page_number=page_number
                    )
                    tables.append(structured_table)
        
        return tables
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean extracted text by removing excessive whitespace and fixing common issues.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Fix broken words at line breaks (e.g., "sam- ple" -> "sample")
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        
        return text.strip()
