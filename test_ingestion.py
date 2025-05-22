#!/usr/bin/env python
"""
Test script to verify that document ingestion is working.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment
load_dotenv("config/config.env")
if os.path.exists(".env"):
    load_dotenv(".env", override=True)

from src.utils.logging_utils import setup_logging
from src.knowledge_ingestion import KnowledgeIngestion

def test_ingestion():
    """Test basic document ingestion functionality."""
    
    # Set up logging
    setup_logging("INFO")
    
    print("🔄 Testing APEGA Document Ingestion...")
    
    # Initialize knowledge ingestion
    try:
        print("📁 Initializing KnowledgeIngestion...")
        knowledge_ingestion = KnowledgeIngestion(
            source_paths=["Context"],
            chunking_strategy="hierarchical",  # Use hierarchical to avoid semantic model dependency
            max_chunk_size_tokens=512,  # Smaller chunks for testing
            chunk_overlap_tokens=100
        )
        print("✅ KnowledgeIngestion initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize KnowledgeIngestion: {e}")
        return False
    
    # Test document processing
    try:
        print("📄 Processing documents...")
        result = knowledge_ingestion.process_documents()
        
        print(f"📊 Processing Results:")
        print(f"   Documents found: {result.get('documents_found', 0)}")
        print(f"   Documents processed: {result.get('documents_processed', 0)}")
        print(f"   Chunks created: {result.get('chunks_created', 0)}")
        print(f"   Embeddings generated: {result.get('embeddings_generated', 0)}")
        print(f"   Vectors stored: {result.get('vectors_stored', 0)}")
        
        if result.get('errors'):
            print(f"⚠️  Errors encountered:")
            for error in result['errors']:
                print(f"     {error}")
        
        # Check if we successfully processed anything
        if result.get('documents_processed', 0) > 0 and result.get('vectors_stored', 0) > 0:
            print("✅ Document ingestion completed successfully!")
            return True
        else:
            print("⚠️  Document ingestion completed but no documents were fully processed")
            return False
            
    except Exception as e:
        print(f"❌ Error during document processing: {e}")
        return False

def test_vector_db():
    """Test vector database connection."""
    try:
        print("🔍 Testing Qdrant connection...")
        from src.knowledge_ingestion.vector_db_manager import VectorDBManager
        
        vector_db = VectorDBManager()
        collection_info = vector_db.get_collection_info()
        
        print(f"✅ Connected to Qdrant successfully!")
        print(f"   Collection: {collection_info.get('name', 'unknown')}")
        print(f"   Vector size: {collection_info.get('vector_size', 'unknown')}")
        print(f"   Points count: {collection_info.get('points_count', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector database connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 APEGA Ingestion Test Starting...\n")
    
    # Test vector database first
    vector_db_ok = test_vector_db()
    print()
    
    if vector_db_ok:
        # Test full ingestion
        ingestion_ok = test_ingestion()
        print()
        
        if ingestion_ok:
            print("🎉 All tests passed! APEGA ingestion is working correctly.")
            sys.exit(0)
        else:
            print("❌ Ingestion test failed.")
            sys.exit(1)
    else:
        print("❌ Vector database test failed. Cannot proceed with ingestion test.")
        sys.exit(1)