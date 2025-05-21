"""
Knowledge Ingestion and Vectorization Module for APEGA.
Combines all components for ingesting, parsing, chunking, and vectorizing documents.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import os

from src.knowledge_ingestion.document_source_manager import DocumentSourceManager
from src.knowledge_ingestion.pdf_parser import PdfParser
from src.knowledge_ingestion.text_chunker import TextChunker
from src.knowledge_ingestion.embedding_generator import EmbeddingGenerator
from src.knowledge_ingestion.vector_db_manager import VectorDBManager
from src.models.data_models import ParsedDocument, TextChunk, EmbeddedChunk, DocumentType


class KnowledgeIngestion:
    """
    Main module for knowledge ingestion and vectorization.
    Orchestrates the entire process from document source management to vector database storage.
    """
    
    def __init__(
        self,
        source_paths: Optional[List[str]] = None,
        openai_api_key: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        chunking_strategy: str = 'hybrid_hierarchical_semantic',
        max_chunk_size_tokens: int = 1024,
        chunk_overlap_tokens: int = 200,
        vector_dimensions: int = 3072,
        use_ocr: bool = False
    ):
        """
        Initialize the KnowledgeIngestion module.
        
        Args:
            source_paths: List of paths to source documents
            openai_api_key: OpenAI API key for embeddings
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key for cloud deployments
            collection_name: Name of the vector database collection
            chunking_strategy: Strategy for text chunking
            max_chunk_size_tokens: Maximum number of tokens per chunk
            chunk_overlap_tokens: Number of tokens to overlap between chunks
            vector_dimensions: Dimensionality of embeddings
            use_ocr: Whether to use OCR for text extraction from PDFs
        """
        # Use environment variables if parameters are not provided
        self.source_paths = source_paths or [os.getenv("SOURCE_DOCUMENTS_DIR", "")]
        
        # Initialize components
        self.document_manager = DocumentSourceManager(self.source_paths)
        self.pdf_parser = PdfParser(use_ocr=use_ocr)
        self.text_chunker = TextChunker(
            strategy=chunking_strategy,
            max_chunk_size_tokens=max_chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens
        )
        self.embedding_generator = EmbeddingGenerator(
            api_key=openai_api_key,
            dimensions=vector_dimensions
        )
        self.vector_db = VectorDBManager(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
            vector_dimensions=vector_dimensions
        )
    
    def process_documents(self) -> Dict[str, Any]:
        """
        Process all documents in the source paths.
        Performs document parsing, chunking, embedding generation, and vector storage.
        
        Returns:
            Processing statistics
        """
        stats = {
            "documents_found": 0,
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "vectors_stored": 0,
            "errors": []
        }
        
        # Get documents that need processing
        documents_to_process = self.document_manager.get_documents_to_process()
        stats["documents_found"] = len(documents_to_process)
        
        if not documents_to_process:
            logger.info("No new or updated documents to process")
            return stats
        
        logger.info(f"Found {len(documents_to_process)} documents to process")
        
        # Process each document
        for doc_info in documents_to_process:
            doc_path = doc_info["source_path"]
            doc_id = doc_info["document_id"]
            doc_type = doc_info["document_type"]
            
            logger.info(f"Processing document: {doc_path}")
            
            try:
                # Parse document
                if doc_type == DocumentType.PDF:
                    parsed_doc = self.pdf_parser.parse_pdf(doc_path)
                else:
                    # For non-PDF documents (not implemented yet)
                    logger.warning(f"Document type {doc_type} not supported yet")
                    self.document_manager.mark_document_processed(doc_path, success=False)
                    stats["errors"].append(f"Unsupported document type: {doc_type}")
                    continue
                
                # Chunk the parsed document
                chunks = self.text_chunker.chunk_document(parsed_doc)
                stats["chunks_created"] += len(chunks)
                
                logger.info(f"Created {len(chunks)} chunks for document {doc_id}")
                
                # Generate embeddings
                embedded_chunks = self.embedding_generator.generate_embeddings(chunks)
                valid_embeddings = sum(1 for chunk in embedded_chunks if chunk.embedding_vector is not None)
                stats["embeddings_generated"] += valid_embeddings
                
                logger.info(f"Generated {valid_embeddings} embeddings for document {doc_id}")
                
                # Store in vector database
                if valid_embeddings > 0:
                    # First delete any existing vectors for this document
                    self.vector_db.delete_by_document_id(doc_id)
                    
                    # Then store the new vectors
                    vectors_stored = self.vector_db.upsert_embeddings(embedded_chunks)
                    stats["vectors_stored"] += vectors_stored
                    
                    logger.info(f"Stored {vectors_stored} vectors for document {doc_id}")
                
                # Mark document as processed
                self.document_manager.mark_document_processed(doc_path, success=True)
                stats["documents_processed"] += 1
                
            except Exception as e:
                logger.error(f"Error processing document {doc_path}: {str(e)}")
                self.document_manager.mark_document_processed(doc_path, success=False)
                stats["errors"].append(f"Error processing {doc_path}: {str(e)}")
        
        logger.info(f"Finished processing {stats['documents_processed']} documents")
        return stats
    
    def process_single_document(self, document_path: str) -> Dict[str, Any]:
        """
        Process a single document by path.
        
        Args:
            document_path: Path to the document to process
            
        Returns:
            Processing statistics for this document
        """
        stats = {
            "document": document_path,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "vectors_stored": 0,
            "success": False,
            "error": None
        }
        
        logger.info(f"Processing single document: {document_path}")
        
        try:
            # Determine document type
            if document_path.lower().endswith('.pdf'):
                doc_type = DocumentType.PDF
                doc_id = os.path.basename(document_path).replace('.pdf', '')
            else:
                logger.warning(f"Document type not supported: {document_path}")
                stats["error"] = "Unsupported document type"
                return stats
            
            # Parse document
            if doc_type == DocumentType.PDF:
                parsed_doc = self.pdf_parser.parse_pdf(document_path)
            else:
                stats["error"] = f"Unsupported document type: {doc_type}"
                return stats
            
            # Chunk the parsed document
            chunks = self.text_chunker.chunk_document(parsed_doc)
            stats["chunks_created"] = len(chunks)
            
            logger.info(f"Created {len(chunks)} chunks for document {doc_id}")
            
            # Generate embeddings
            embedded_chunks = self.embedding_generator.generate_embeddings(chunks)
            valid_embeddings = sum(1 for chunk in embedded_chunks if chunk.embedding_vector is not None)
            stats["embeddings_generated"] = valid_embeddings
            
            logger.info(f"Generated {valid_embeddings} embeddings for document {doc_id}")
            
            # Store in vector database
            if valid_embeddings > 0:
                # First delete any existing vectors for this document
                self.vector_db.delete_by_document_id(doc_id)
                
                # Then store the new vectors
                vectors_stored = self.vector_db.upsert_embeddings(embedded_chunks)
                stats["vectors_stored"] = vectors_stored
                
                logger.info(f"Stored {vectors_stored} vectors for document {doc_id}")
            
            # Mark document as processed
            self.document_manager.mark_document_processed(document_path, success=True)
            stats["success"] = True
            
        except Exception as e:
            logger.error(f"Error processing document {document_path}: {str(e)}")
            stats["error"] = str(e)
            self.document_manager.mark_document_processed(document_path, success=False)
        
        return stats
