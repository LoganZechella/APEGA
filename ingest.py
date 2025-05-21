#!/usr/bin/env python
"""
Document Ingestor for APEGA.
Script for ingesting documents into the APEGA knowledge base.
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from loguru import logger

# Ensure src is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge_ingestion import KnowledgeIngestion

def setup_logging(log_level):
    """Configure logging"""
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level=log_level)
    logger.add("logs/ingest_{time}.log", rotation="10 MB", level=log_level)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="APEGA Document Ingestion Tool")
    parser.add_argument("--config", default="config/config.env", help="Path to configuration file")
    parser.add_argument("--document", help="Path to a specific document to ingest")
    parser.add_argument("--directory", help="Path to a directory of documents to ingest")
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Load the configuration file
    load_dotenv(args.config)
    
    # Load .env file for sensitive credentials if it exists
    if os.path.exists(".env"):
        load_dotenv(".env", override=True)
    
    # Setup logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    setup_logging(log_level)
    
    # Initialize knowledge ingestion
    try:
        source_paths = []
        
        if args.document:
            source_paths.append(args.document)
            logger.info(f"Ingesting document: {args.document}")
        elif args.directory:
            source_paths.append(args.directory)
            logger.info(f"Ingesting documents from directory: {args.directory}")
        else:
            # Default to documents directory from config
            default_dir = os.getenv("SOURCE_DOCUMENTS_DIR", "Context")
            source_paths.append(default_dir)
            logger.info(f"Ingesting documents from default directory: {default_dir}")
        
        knowledge_ingestion = KnowledgeIngestion(source_paths=source_paths)
        
        # Process documents
        result = knowledge_ingestion.process_documents()
        
        # Log results
        logger.info(f"Documents found: {result.get('documents_found', 0)}")
        logger.info(f"Documents processed: {result.get('documents_processed', 0)}")
        logger.info(f"Chunks created: {result.get('chunks_created', 0)}")
        logger.info(f"Embeddings generated: {result.get('embeddings_generated', 0)}")
        logger.info(f"Vectors stored: {result.get('vectors_stored', 0)}")
        
        # Check for errors
        if result.get("errors"):
            logger.warning(f"Completed with {len(result['errors'])} errors:")
            for error in result["errors"]:
                logger.warning(f"  - {error}")
        else:
            logger.info("Document ingestion completed successfully")
            
    except Exception as e:
        logger.exception(f"Error during document ingestion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()