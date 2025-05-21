"""
Document Source Manager for APEGA.
Handles the management of document sources and tracking processed documents.
"""

import os
import json
from typing import List, Dict, Set, Optional
from datetime import datetime
import hashlib
from pathlib import Path
from loguru import logger

from src.models.data_models import DocumentType


class DocumentSourceManager:
    """
    Manages document sources and tracks which documents have been processed.
    Keeps track of document changes to determine which ones need reprocessing.
    """
    
    def __init__(self, source_paths: List[str], manifest_path: Optional[str] = None):
        """
        Initialize the DocumentSourceManager.
        
        Args:
            source_paths: List of file paths or directories to process
            manifest_path: Path to the manifest file tracking processed documents
        """
        self.source_paths = source_paths
        self.manifest_path = manifest_path or 'config/document_manifest.json'
        self.processed_docs = self._load_manifest()
    
    def _load_manifest(self) -> Dict[str, Dict]:
        """
        Load the document processing manifest.
        
        Returns:
            Dictionary mapping document paths to their processing metadata
        """
        if not os.path.exists(self.manifest_path):
            logger.info(f"No manifest found at {self.manifest_path}. Creating new manifest.")
            return {}
        
        try:
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading manifest: {e}. Starting with empty manifest.")
            return {}
    
    def _save_manifest(self) -> None:
        """Save the current document processing manifest."""
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        with open(self.manifest_path, 'w') as f:
            json.dump(self.processed_docs, f, indent=2)
    
    def _get_file_hash(self, file_path: str) -> str:
        """
        Calculate MD5 hash of a file to detect changes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MD5 hash of the file
        """
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except IOError as e:
            logger.error(f"Error reading file {file_path}: {e}")
            # Return a dummy hash for error cases
            return f"error-{datetime.now().isoformat()}"
    
    def _get_file_metadata(self, file_path: str) -> Dict:
        """
        Get file metadata including size, modified time, and hash.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of file metadata
        """
        stat = os.stat(file_path)
        return {
            'size': stat.st_size,
            'mtime': stat.st_mtime,
            'hash': self._get_file_hash(file_path),
            'last_processed': datetime.now().isoformat()
        }
    
    def _infer_document_type(self, file_path: str) -> DocumentType:
        """
        Infer the document type from the file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            The document type
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return DocumentType.PDF
        elif ext in ['.html', '.htm']:
            return DocumentType.HTML
        else:
            return DocumentType.TEXT
    
    def _get_all_documents(self) -> List[str]:
        """
        Get all document files from the provided source paths.
        
        Returns:
            List of document file paths
        """
        all_docs = []
        
        for source_path in self.source_paths:
            if os.path.isfile(source_path):
                all_docs.append(source_path)
            elif os.path.isdir(source_path):
                for root, _, files in os.walk(source_path):
                    for file in files:
                        # Only consider PDF files for now
                        if file.lower().endswith('.pdf'):
                            all_docs.append(os.path.join(root, file))
        
        return all_docs
    
    def get_documents_to_process(self) -> List[Dict]:
        """
        Identify new or updated documents that need processing.
        
        Returns:
            List of document metadata dictionaries for documents that need processing
        """
        all_docs = self._get_all_documents()
        to_process = []
        
        for doc_path in all_docs:
            doc_path = os.path.abspath(doc_path)
            current_metadata = self._get_file_metadata(doc_path)
            
            needs_processing = False
            if doc_path not in self.processed_docs:
                # New document
                needs_processing = True
            elif self.processed_docs[doc_path]['hash'] != current_metadata['hash']:
                # Document has changed
                needs_processing = True
            
            if needs_processing:
                document_type = self._infer_document_type(doc_path)
                to_process.append({
                    'document_id': Path(doc_path).stem,
                    'source_path': doc_path,
                    'document_type': document_type,
                    'metadata': current_metadata
                })
        
        return to_process
    
    def mark_document_processed(self, document_path: str, success: bool = True) -> None:
        """
        Mark a document as processed in the manifest.
        
        Args:
            document_path: Path to the document
            success: Whether the processing was successful
        """
        document_path = os.path.abspath(document_path)
        metadata = self._get_file_metadata(document_path)
        metadata['processing_success'] = success
        
        self.processed_docs[document_path] = metadata
        self._save_manifest()
    
    def clear_processing_history(self, document_path: Optional[str] = None) -> None:
        """
        Clear processing history for a specific document or all documents.
        
        Args:
            document_path: Path to the document, or None for all documents
        """
        if document_path:
            document_path = os.path.abspath(document_path)
            if document_path in self.processed_docs:
                del self.processed_docs[document_path]
        else:
            self.processed_docs = {}
        
        self._save_manifest()
    
    def get_document_metadata(self, document_path: str) -> Optional[Dict]:
        """
        Get the processing metadata for a document.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Processing metadata for the document, or None if not processed
        """
        document_path = os.path.abspath(document_path)
        return self.processed_docs.get(document_path)
