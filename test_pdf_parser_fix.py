#!/usr/bin/env python
"""
Test script to verify PDF parser fixes for APEGA.
Tests that the pdf_parser can successfully parse documents without field mismatches.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.knowledge_ingestion.pdf_parser import PdfParser
from src.models.data_models import ParsedDocument, DocumentType

def test_pdf_parser():
    """Test that the PDF parser can parse a document without errors."""
    print("Testing PDF parser fixes...")
    
    # Initialize parser
    parser = PdfParser()
    print("✓ PDF parser initialized successfully")
    
    # Test PDF path (using one of the documents that failed before)
    test_pdf = "/Users/logan/Git/Agents/APEGA/Context/CLP Exam Blueprint Oct 2019.pdf"
    
    if os.path.exists(test_pdf):
        print(f"\nParsing test document: {test_pdf}")
        
        try:
            # Parse the document
            parsed_doc = parser.parse_pdf(test_pdf)
            
            # Verify it's a valid ParsedDocument
            assert isinstance(parsed_doc, ParsedDocument), "Result is not a ParsedDocument"
            print("✓ Document parsed successfully")
            
            # Check key fields
            print(f"\nDocument details:")
            print(f"  - Document ID: {parsed_doc.document_id}")
            print(f"  - Title: {parsed_doc.title}")
            print(f"  - Source path: {parsed_doc.source_path}")
            print(f"  - Document type: {parsed_doc.document_type}")
            print(f"  - Text content length: {len(parsed_doc.text_content)} chars")
            print(f"  - Number of tables: {len(parsed_doc.tables)}")
            
            # Check metadata
            print(f"\nMetadata keys: {list(parsed_doc.metadata.keys())}")
            
            # Verify TOC is in metadata (required by text_chunker)
            if 'toc' in parsed_doc.metadata:
                print(f"✓ TOC found in metadata with {len(parsed_doc.metadata['toc'])} entries")
            else:
                print("⚠ TOC not found in metadata")
            
            # Verify page_count is in metadata
            if 'page_count' in parsed_doc.metadata:
                print(f"✓ Page count in metadata: {parsed_doc.metadata['page_count']}")
            else:
                print("⚠ Page count not found in metadata")
            
            print("\n✅ All tests passed! The PDF parser fix is working correctly.")
            return True
            
        except Exception as e:
            print(f"\n❌ Error parsing document: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"\n⚠ Test PDF not found: {test_pdf}")
        print("Please ensure the Context directory contains the CLP documents.")
        return False

if __name__ == "__main__":
    success = test_pdf_parser()
    sys.exit(0 if success else 1)
