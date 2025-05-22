#!/usr/bin/env python
"""
Diagnostic script for APEGA ingestion issues.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_environment():
    """Check environment setup."""
    print("🔍 Environment Check:")
    
    # Check if config file exists
    config_path = "config/config.env"
    if os.path.exists(config_path):
        print(f"✅ Config file found: {config_path}")
        load_dotenv(config_path)
    else:
        print(f"❌ Config file missing: {config_path}")
        return False
    
    # Check .env file
    if os.path.exists(".env"):
        print("✅ .env file found")
        load_dotenv(".env", override=True)
    else:
        print("ℹ️  No .env file (this is optional)")
    
    # Check key environment variables
    source_dir = os.getenv("SOURCE_DOCUMENTS_DIR", "Context")
    print(f"📁 Source directory: {source_dir}")
    
    if os.path.exists(source_dir):
        files = [f for f in os.listdir(source_dir) if f.endswith('.pdf')]
        print(f"✅ Found {len(files)} PDF files in source directory")
        for f in files[:3]:  # Show first 3 files
            print(f"   - {f}")
        if len(files) > 3:
            print(f"   ... and {len(files) - 3} more")
    else:
        print(f"❌ Source directory does not exist: {source_dir}")
        return False
    
    # Check chunking strategy
    chunking_strategy = os.getenv("CHUNKING_STRATEGY", "hierarchical")
    print(f"📝 Chunking strategy: {chunking_strategy}")
    
    # Check Qdrant configuration
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    print(f"🗄️  Qdrant URL: {qdrant_url}")
    if qdrant_api_key:
        print(f"🔑 Qdrant API key: {'*' * min(len(qdrant_api_key), 10)}...")
    else:
        print("🔑 No Qdrant API key (good for local instances)")
    
    return True

def check_dependencies():
    """Check if required dependencies are available."""
    print("\n📦 Dependency Check:")
    
    required_packages = [
        'qdrant_client',
        'openai', 
        'google.generativeai',
        'nltk',
        'fitz',  # PyMuPDF
        'loguru',
        'pydantic'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        return False
    
    return True

def check_qdrant_connection():
    """Check if we can connect to Qdrant."""
    print("\n🗄️  Qdrant Connection Check:")
    
    try:
        import qdrant_client
        
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")
        
        # Parse URL
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or url.replace("http://", "").replace("https://", "")
        port = parsed.port or 6333
        use_https = parsed.scheme == 'https'
        
        # Clear API key for local connections
        if api_key and ("|" in api_key or len(api_key) > 50):
            if "localhost" in url or "127.0.0.1" in url:
                api_key = None
        
        print(f"   Connecting to {host}:{port} (HTTPS: {use_https})")
        
        client = qdrant_client.QdrantClient(
            host=host,
            port=port,
            api_key=api_key,
            timeout=10.0,
            https=use_https,
            prefer_grpc=False
        )
        
        # Test connection
        collections = client.get_collections()
        print(f"✅ Connected! Found {len(collections.collections)} collections")
        
        # List collections
        for collection in collections.collections:
            print(f"   - {collection.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Ensure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        print("   2. Check if the URL is correct")
        print("   3. For local instances, remove any QDRANT_API_KEY")
        return False

def check_text_chunker():
    """Check if text chunker can initialize without internet."""
    print("\n📝 Text Chunker Check:")
    
    try:
        # Import here to catch initialization errors
        from src.knowledge_ingestion.text_chunker import TextChunker
        
        # Test with hierarchical strategy (no internet required)
        chunker = TextChunker(strategy="hierarchical")
        print("✅ Hierarchical chunker initialized successfully")
        
        # Test with semantic strategy (might require internet)
        try:
            semantic_chunker = TextChunker(strategy="hybrid_hierarchical_semantic")
            if semantic_chunker.semantic_available:
                print("✅ Semantic chunker available")
            else:
                print("ℹ️  Semantic chunker not available (will use fallback)")
        except Exception as e:
            print(f"ℹ️  Semantic chunker failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Text chunker initialization failed: {e}")
        return False

def main():
    """Run all diagnostic checks."""
    print("🔧 APEGA Ingestion Diagnostics\n")
    
    checks = [
        ("Environment", check_environment),
        ("Dependencies", check_dependencies), 
        ("Qdrant Connection", check_qdrant_connection),
        ("Text Chunker", check_text_chunker)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed with exception: {e}")
            results.append((name, False))
    
    print("\n📊 Summary:")
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! Ingestion should work correctly.")
    else:
        print("\n⚠️  Some checks failed. Please address the issues above.")

if __name__ == "__main__":
    main()