#!/usr/bin/env python
"""
Interactive RAG Testing Tool for APEGA.
Command-line interface for manual RAG testing and debugging.
"""

import os
import sys
import cmd
import json
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment
load_dotenv("config/config.env")
if os.path.exists(".env"):
    load_dotenv(".env", override=True)

from src.utils.logging_utils import setup_logging
from src.rag_engine import RAGEngine
from src.knowledge_ingestion.embedding_generator import EmbeddingGenerator
from src.knowledge_ingestion.vector_db_manager import VectorDBManager

class InteractiveRAGShell(cmd.Cmd):
    """Interactive shell for RAG testing and debugging."""
    
    intro = """
    🔍 APEGA Interactive RAG Testing Shell
    =====================================
    
    Available commands:
    - query <text>        : Test RAG with a query
    - search <text>       : Test hybrid search only
    - dense <text>        : Test dense search only
    - sparse <text>       : Test sparse search only
    - rerank <text>       : Test reranking
    - analyze <text>      : Test deep analysis
    - filters <filters>   : Set search filters (JSON format)
    - config             : Show current configuration
    - stats              : Show database statistics
    - help               : Show this help
    - quit               : Exit the shell
    
    Type 'help <command>' for detailed help on a command.
    """
    
    prompt = 'RAG> '
    
    def __init__(self):
        super().__init__()
        self.setup_rag_engine()
        self.current_filters = None
        self.last_results = None
    
    def setup_rag_engine(self):
        """Initialize RAG engine components."""
        try:
            print("Initializing RAG engine...")
            
            self.embedding_generator = EmbeddingGenerator()
            self.vector_db = VectorDBManager()
            self.rag_engine = RAGEngine(
                vector_db=self.vector_db,
                embedding_generator=self.embedding_generator,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            print("✅ RAG engine initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize RAG engine: {str(e)}")
            sys.exit(1)
    
    def do_query(self, query_text: str):
        """Test full RAG pipeline with a query."""
        if not query_text.strip():
            print("Usage: query <text>")
            return
        
        try:
            print(f"\n🔍 Testing RAG pipeline with: '{query_text}'")
            print("-" * 60)
            
            # Run full RAG pipeline
            retrieved_contexts, synthesized_knowledge = self.rag_engine.retrieve_and_analyze(
                query_text=query_text,
                query_details={"task": "interactive test"},
                filters=self.current_filters
            )
            
            # Display results
            print(f"📊 Retrieved {len(retrieved_contexts)} contexts")
            
            if retrieved_contexts:
                print(f"\n📋 Top Retrieved Contexts:")
                for i, ctx in enumerate(retrieved_contexts[:3], 1):
                    score = ctx.rerank_score or ctx.initial_score
                    print(f"\n{i}. Score: {score:.3f}")
                    print(f"   Document: {ctx.document_id}")
                    print(f"   Text: {ctx.text[:200]}...")
            
            if synthesized_knowledge.summary:
                print(f"\n🧠 Synthesized Knowledge:")
                print(f"Summary: {synthesized_knowledge.summary[:300]}...")
                
                if synthesized_knowledge.key_concepts:
                    print(f"\nKey Concepts ({len(synthesized_knowledge.key_concepts)}):")
                    for concept in synthesized_knowledge.key_concepts[:3]:
                        print(f"  - {concept.get('concept', 'N/A')}")
                
                if synthesized_knowledge.potential_exam_areas:
                    print(f"\nPotential Exam Areas ({len(synthesized_knowledge.potential_exam_areas)}):")
                    for area in synthesized_knowledge.potential_exam_areas[:3]:
                        print(f"  - {area}")
            
            # Store results for further analysis
            self.last_results = {
                "query": query_text,
                "retrieved_contexts": retrieved_contexts,
                "synthesized_knowledge": synthesized_knowledge
            }
            
        except Exception as e:
            print(f"❌ Query failed: {str(e)}")
    
    def do_search(self, query_text: str):
        """Test hybrid search only."""
        if not query_text.strip():
            print("Usage: search <text>")
            return
        
        try:
            print(f"\n🔍 Testing hybrid search with: '{query_text}'")
            print("-" * 60)
            
            results = self.rag_engine.hybrid_searcher.hybrid_search(
                query_text=query_text,
                top_k=10,
                filters=self.current_filters
            )
            
            print(f"📊 Retrieved {len(results)} contexts")
            
            for i, ctx in enumerate(results[:5], 1):
                print(f"\n{i}. Score: {ctx.initial_score:.3f}")
                print(f"   Document: {ctx.document_id}")
                print(f"   Text: {ctx.text[:150]}...")
            
        except Exception as e:
            print(f"❌ Search failed: {str(e)}")
    
    def do_dense(self, query_text: str):
        """Test dense vector search only."""
        if not query_text.strip():
            print("Usage: dense <text>")
            return
        
        try:
            print(f"\n🔍 Testing dense search with: '{query_text}'")
            print("-" * 60)
            
            results = self.rag_engine.hybrid_searcher._dense_search(
                query_text=query_text,
                top_k=10,
                filters=self.current_filters
            )
            
            print(f"📊 Retrieved {len(results)} contexts")
            
            for i, ctx in enumerate(results[:5], 1):
                print(f"\n{i}. Score: {ctx.initial_score:.3f}")
                print(f"   Document: {ctx.document_id}")
                print(f"   Text: {ctx.text[:150]}...")
            
        except Exception as e:
            print(f"❌ Dense search failed: {str(e)}")
    
    def do_sparse(self, query_text: str):
        """Test sparse keyword search only."""
        if not query_text.strip():
            print("Usage: sparse <text>")
            return
        
        try:
            print(f"\n🔍 Testing sparse search with: '{query_text}'")
            print("-" * 60)
            
            results = self.rag_engine.hybrid_searcher._sparse_search(
                query_text=query_text,
                top_k=10,
                filters=self.current_filters
            )
            
            print(f"📊 Retrieved {len(results)} contexts")
            
            for i, ctx in enumerate(results[:5], 1):
                print(f"\n{i}. Score: {ctx.initial_score:.3f}")
                print(f"   Document: {ctx.document_id}")
                print(f"   Text: {ctx.text[:150]}...")
            
        except Exception as e:
            print(f"❌ Sparse search failed: {str(e)}")
    
    def do_rerank(self, query_text: str):
        """Test reranking with previous search results."""
        if not query_text.strip():
            print("Usage: rerank <text>")
            return
        
        try:
            print(f"\n🔍 Testing reranking with: '{query_text}'")
            print("-" * 60)
            
            # First get hybrid search results
            hybrid_results = self.rag_engine.hybrid_searcher.hybrid_search(
                query_text=query_text,
                top_k=10,
                filters=self.current_filters
            )
            
            if not hybrid_results:
                print("❌ No search results to rerank")
                return
            
            # Then rerank them
            reranked_results = self.rag_engine.reranker.rerank_contexts(
                query_text=query_text,
                contexts=hybrid_results,
                top_n=5
            )
            
            print(f"📊 Reranked {len(reranked_results)} contexts")
            
            for i, ctx in enumerate(reranked_results, 1):
                print(f"\n{i}. Initial Score: {ctx.initial_score:.3f}, Rerank Score: {ctx.rerank_score:.3f}")
                print(f"   Document: {ctx.document_id}")
                print(f"   Text: {ctx.text[:150]}...")
            
        except Exception as e:
            print(f"❌ Reranking failed: {str(e)}")
    
    def do_analyze(self, query_text: str):
        """Test deep analysis with search results."""
        if not query_text.strip():
            print("Usage: analyze <text>")
            return
        
        try:
            print(f"\n🔍 Testing deep analysis with: '{query_text}'")
            print("-" * 60)
            
            # Get search and reranked results
            hybrid_results = self.rag_engine.hybrid_searcher.hybrid_search(
                query_text=query_text,
                top_k=10,
                filters=self.current_filters
            )
            
            if not hybrid_results:
                print("❌ No search results for analysis")
                return
            
            reranked_results = self.rag_engine.reranker.rerank_contexts(
                query_text=query_text,
                contexts=hybrid_results,
                top_n=5
            )
            
            # Run deep analysis
            synthesized = self.rag_engine.deep_analyzer.synthesize_knowledge(
                query_details={"task": "interactive analysis"},
                contexts=reranked_results
            )
            
            print(f"🧠 Deep Analysis Results:")
            print(f"Summary: {synthesized.summary[:500]}...")
            
            if synthesized.key_concepts:
                print(f"\nKey Concepts ({len(synthesized.key_concepts)}):")
                for concept in synthesized.key_concepts:
                    print(f"  - {concept.get('concept', 'N/A')}: {concept.get('description', '')[:100]}...")
            
            if synthesized.potential_exam_areas:
                print(f"\nPotential Exam Areas ({len(synthesized.potential_exam_areas)}):")
                for area in synthesized.potential_exam_areas:
                    print(f"  - {area}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
    
    def do_filters(self, filter_json: str):
        """Set search filters (JSON format)."""
        if not filter_json.strip():
            print("Current filters:", self.current_filters)
            print("Usage: filters <JSON>")
            print("Example: filters {\"clp_domain_id\": \"Domain_1\"}")
            print("Clear filters: filters null")
            return
        
        try:
            if filter_json.strip().lower() == "null":
                self.current_filters = None
                print("✅ Filters cleared")
            else:
                self.current_filters = json.loads(filter_json)
                print(f"✅ Filters set: {self.current_filters}")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {str(e)}")
    
    def do_config(self, line):
        """Show current configuration."""
        print(f"\n⚙️ RAG Engine Configuration:")
        print(f"   Vector DB URL: {os.getenv('QDRANT_URL', 'Not set')}")
        print(f"   Collection: {os.getenv('QDRANT_COLLECTION_NAME', 'Not set')}")
        print(f"   Vector Dimensions: {os.getenv('VECTOR_DIMENSIONS', 'Not set')}")
        print(f"   Top K Dense: {os.getenv('TOP_K_DENSE', 'Not set')}")
        print(f"   Top K Sparse: {os.getenv('TOP_K_SPARSE', 'Not set')}")
        print(f"   Top K Rerank: {os.getenv('TOP_K_RERANK', 'Not set')}")
        print(f"   Current Filters: {self.current_filters}")
    
    def do_stats(self, line):
        """Show database statistics."""
        try:
            collection_info = self.vector_db.get_collection_info()
            print(f"\n📊 Database Statistics:")
            print(f"   Collection Name: {collection_info.get('name', 'Unknown')}")
            print(f"   Vector Size: {collection_info.get('vector_size', 'Unknown')}")
            print(f"   Distance Metric: {collection_info.get('distance', 'Unknown')}")
            print(f"   Total Points: {collection_info.get('points_count', 0):,}")
            print(f"   Indexed Vectors: {collection_info.get('indexed_vectors_count', 'Unknown')}")
        except Exception as e:
            print(f"❌ Failed to get stats: {str(e)}")
    
    def do_last(self, line):
        """Show details from the last query results."""
        if not self.last_results:
            print("No previous query results available")
            return
        
        print(f"\n📋 Last Query Details:")
        print(f"Query: {self.last_results['query']}")
        print(f"Retrieved Contexts: {len(self.last_results['retrieved_contexts'])}")
        
        contexts = self.last_results['retrieved_contexts']
        if contexts:
            print(f"\nAll Retrieved Contexts:")
            for i, ctx in enumerate(contexts, 1):
                score = ctx.rerank_score or ctx.initial_score
                print(f"\n{i}. Score: {score:.3f}")
                print(f"   Document: {ctx.document_id}")
                print(f"   Domain: {ctx.metadata.get('clp_domain_id', 'Unknown')}")
                print(f"   Text: {ctx.text[:100]}...")
        
        knowledge = self.last_results['synthesized_knowledge']
        if knowledge.summary:
            print(f"\nFull Summary:")
            print(knowledge.summary)
            
            if knowledge.key_concepts:
                print(f"\nAll Key Concepts:")
                for concept in knowledge.key_concepts:
                    print(f"  - {concept.get('concept', 'N/A')}: {concept.get('description', '')}")
    
    def do_domains(self, line):
        """Show available CLP domains."""
        domains = [
            "Domain_1_Opportunity_Assessment_Development_Valuation",
            "Domain_2_Intellectual_Property_Protection",
            "Domain_3_Strategy_Management_Commercialization",
            "Domain_4_Negotiation_Agreement_Development",
            "Domain_5_Agreement_Management"
        ]
        
        print(f"\n📚 Available CLP Domains:")
        for i, domain in enumerate(domains, 1):
            domain_name = domain.replace("Domain_", "").replace("_", " ")
            print(f"   {i}. {domain_name}")
            print(f"      Filter: {{\"clp_domain_id\": \"{domain}\"}}")
    
    def do_examples(self, line):
        """Show example queries for testing."""
        examples = [
            "What factors should be considered when valuing intellectual property?",
            "How do you assess market opportunities for licensing?",
            "What are the key differences between patents and trademarks?",
            "How do you conduct a freedom-to-operate analysis?",
            "What are the essential elements of a licensing agreement?",
            "How do you monitor compliance with licensing agreements?"
        ]
        
        print(f"\n💡 Example Test Queries:")
        for i, query in enumerate(examples, 1):
            print(f"   {i}. {query}")
    
    def help_query(self):
        """Help for query command."""
        print("""
query <text> - Test the full RAG pipeline with a query
    
    This command runs the complete RAG pipeline:
    1. Hybrid search (dense + sparse)
    2. Reranking with LLM
    3. Deep analysis and knowledge synthesis
    
    Example: query What is patent valuation?
        """)
    
    def help_search(self):
        """Help for search command."""
        print("""
search <text> - Test hybrid search only
    
    This command tests only the hybrid search component,
    combining dense vector search and sparse keyword search.
    
    Example: search intellectual property licensing
        """)
    
    def help_filters(self):
        """Help for filters command."""
        print("""
filters <JSON> - Set search filters
    
    Set filters to constrain search results to specific criteria.
    Use JSON format for filter specification.
    
    Examples:
    filters {"clp_domain_id": "Domain_1"}
    filters {"document_type": "practice_test"}
    filters null  (to clear filters)
        """)
    
    def do_quit(self, line):
        """Exit the RAG shell."""
        print("👋 Goodbye!")
        return True
    
    def do_exit(self, line):
        """Exit the RAG shell."""
        return self.do_quit(line)
    
    def do_EOF(self, line):
        """Handle Ctrl+D."""
        print("\n👋 Goodbye!")
        return True

def main():
    """Main entry point for interactive RAG shell."""
    setup_logging("INFO")
    
    try:
        shell = InteractiveRAGShell()
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 