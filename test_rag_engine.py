#!/usr/bin/env python
"""
Comprehensive RAG Engine Testing Framework for APEGA.
Tests and validates the entire RAG pipeline with CLP-specific queries.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from loguru import logger
import numpy as np

# Load environment
load_dotenv("config/config.env")
if os.path.exists(".env"):
    load_dotenv(".env", override=True)

from src.utils.logging_utils import setup_logging
from src.rag_engine import RAGEngine
from src.knowledge_ingestion.embedding_generator import EmbeddingGenerator
from src.knowledge_ingestion.vector_db_manager import VectorDBManager
from tests.rag_test_queries import CLPTestQueries, TestQuery
from tests.rag_quality_metrics import RAGQualityMetrics, RetrievalQualityReport

class RAGTestFramework:
    """Comprehensive testing framework for APEGA RAG Engine."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the RAG testing framework."""
        self.verbose = verbose
        self.setup_logging()
        
        # Initialize RAG components
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDBManager()
        self.rag_engine = RAGEngine(
            vector_db=self.vector_db,
            embedding_generator=self.embedding_generator,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize quality metrics
        self.quality_metrics = RAGQualityMetrics()
        
        # Test results storage
        self.test_results = []
        self.summary_stats = {}
    
    def setup_logging(self):
        """Set up logging for the test framework."""
        log_level = "DEBUG" if self.verbose else "INFO"
        setup_logging(log_level)
        
        # Create logs directory for test results
        os.makedirs("logs/rag_tests", exist_ok=True)
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """
        Run the complete RAG test suite.
        
        Returns:
            Dictionary with comprehensive test results
        """
        logger.info("🚀 Starting Comprehensive RAG Test Suite")
        start_time = time.time()
        
        # 1. Verify RAG component initialization
        if not self._verify_rag_components():
            return {"status": "failed", "error": "RAG component initialization failed"}
        
        # 2. Test vector database connectivity
        if not self._test_vector_db_connectivity():
            return {"status": "failed", "error": "Vector database connectivity failed"}
        
        # 3. Run individual component tests
        component_results = self._test_individual_components()
        
        # 4. Run end-to-end RAG pipeline tests
        pipeline_results = self._test_rag_pipeline()
        
        # 5. Run CLP-specific query tests
        clp_results = self._test_clp_queries()
        
        # 6. Generate comprehensive report
        total_time = time.time() - start_time
        
        final_results = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "component_tests": component_results,
            "pipeline_tests": pipeline_results,
            "clp_query_tests": clp_results,
            "summary_statistics": self._calculate_summary_statistics(clp_results),
            "recommendations": self._generate_overall_recommendations(clp_results)
        }
        
        # Save results
        self._save_test_results(final_results)
        
        # Print summary
        self._print_test_summary(final_results)
        
        return final_results
    
    def _verify_rag_components(self) -> bool:
        """Verify all RAG components are properly initialized."""
        logger.info("🔍 Verifying RAG component initialization...")
        
        try:
            # Test embedding generator
            test_text = "test query for embedding generation"
            test_chunk = type('obj', (object,), {
                'chunk_id': 'test',
                'document_id': 'test',
                'text': test_text,
                'metadata': {}
            })()
            
            embeddings = self.embedding_generator.generate_embeddings([test_chunk])
            if not embeddings or embeddings[0].embedding_vector is None:
                logger.error("❌ Embedding generator test failed")
                return False
            
            logger.info("✅ Embedding generator working")
            
            # Test vector database
            collection_info = self.vector_db.get_collection_info()
            if not collection_info or collection_info.get("points_count", 0) == 0:
                logger.error("❌ Vector database has no data")
                return False
            
            logger.info(f"✅ Vector database connected ({collection_info.get('points_count', 0)} vectors)")
            
            # Test RAG engine initialization
            if not hasattr(self.rag_engine, 'hybrid_searcher') or not hasattr(self.rag_engine, 'reranker'):
                logger.error("❌ RAG engine components not properly initialized")
                return False
            
            logger.info("✅ RAG engine components initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Component verification failed: {str(e)}")
            return False
    
    def _test_vector_db_connectivity(self) -> bool:
        """Test vector database connectivity and basic search functionality."""
        logger.info("🔍 Testing vector database connectivity...")
        
        try:
            # Test basic dense search
            test_embedding = [0.1] * self.vector_db.vector_dimensions
            results = self.vector_db.dense_vector_search(
                query_embedding=test_embedding,
                top_k=5
            )
            
            if not results:
                logger.warning("⚠️ Dense search returned no results")
                return False
            
            logger.info(f"✅ Dense search working ({len(results)} results)")
            
            # Test keyword search if available
            if hasattr(self.vector_db, 'keyword_search'):
                keyword_results = self.vector_db.keyword_search(
                    query_text="intellectual property",
                    top_k=5
                )
                logger.info(f"✅ Keyword search working ({len(keyword_results)} results)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Vector database connectivity test failed: {str(e)}")
            return False
    
    def _test_individual_components(self) -> Dict[str, Any]:
        """Test individual RAG components."""
        logger.info("🔍 Testing individual RAG components...")
        
        results = {}
        test_query = "What factors should be considered when valuing intellectual property?"
        
        try:
            # Test HybridSearcher
            logger.info("Testing HybridSearcher...")
            hybrid_results = self.rag_engine.hybrid_searcher.hybrid_search(
                query_text=test_query,
                top_k=10
            )
            results["hybrid_searcher"] = {
                "status": "success" if hybrid_results else "failed",
                "results_count": len(hybrid_results),
                "avg_score": np.mean([r.initial_score for r in hybrid_results]) if hybrid_results else 0
            }
            
            # Test ReRanker
            if hybrid_results:
                logger.info("Testing ReRanker...")
                reranked_results = self.rag_engine.reranker.rerank_contexts(
                    query_text=test_query,
                    contexts=hybrid_results[:10],
                    top_n=5
                )
                results["reranker"] = {
                    "status": "success" if reranked_results else "failed",
                    "results_count": len(reranked_results),
                    "avg_rerank_score": np.mean([r.rerank_score for r in reranked_results if r.rerank_score]) if reranked_results else 0
                }
                
                # Test DeepAnalyzer
                if reranked_results:
                    logger.info("Testing DeepAnalyzer...")
                    synthesized = self.rag_engine.deep_analyzer.synthesize_knowledge(
                        query_details={"task": "test query"},
                        contexts=reranked_results
                    )
                    results["deep_analyzer"] = {
                        "status": "success" if synthesized.summary else "failed",
                        "summary_length": len(synthesized.summary) if synthesized.summary else 0,
                        "key_concepts_count": len(synthesized.key_concepts),
                        "exam_areas_count": len(synthesized.potential_exam_areas)
                    }
            
            logger.info("✅ Individual component testing completed")
            
        except Exception as e:
            logger.error(f"❌ Individual component testing failed: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def _test_rag_pipeline(self) -> Dict[str, Any]:
        """Test end-to-end RAG pipeline."""
        logger.info("🔍 Testing end-to-end RAG pipeline...")
        
        pipeline_tests = []
        test_queries = [
            "What factors should be considered when valuing intellectual property?",
            "How do you assess market opportunities for licensing?",
            "What are the key differences between patents and trademarks?"
        ]
        
        for query in test_queries:
            try:
                logger.info(f"Testing pipeline with query: '{query[:50]}...'")
                
                start_time = time.time()
                retrieved_contexts, synthesized_knowledge = self.rag_engine.retrieve_and_analyze(
                    query_text=query,
                    query_details={"task": "pipeline test"}
                )
                processing_time = time.time() - start_time
                
                test_result = {
                    "query": query,
                    "status": "success",
                    "processing_time": processing_time,
                    "retrieved_contexts": len(retrieved_contexts),
                    "synthesis_summary_length": len(synthesized_knowledge.summary) if synthesized_knowledge.summary else 0,
                    "key_concepts": len(synthesized_knowledge.key_concepts),
                    "potential_exam_areas": len(synthesized_knowledge.potential_exam_areas)
                }
                
                pipeline_tests.append(test_result)
                logger.info(f"✅ Pipeline test succeeded ({len(retrieved_contexts)} contexts, {processing_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"❌ Pipeline test failed for query '{query}': {str(e)}")
                pipeline_tests.append({
                    "query": query,
                    "status": "failed",
                    "error": str(e)
                })
        
        return {
            "total_tests": len(test_queries),
            "successful_tests": sum(1 for t in pipeline_tests if t.get("status") == "success"),
            "avg_processing_time": np.mean([t.get("processing_time", 0) for t in pipeline_tests if t.get("processing_time")]),
            "tests": pipeline_tests
        }
    
    def _test_clp_queries(self) -> Dict[str, Any]:
        """Test RAG with comprehensive CLP-specific queries."""
        logger.info("🔍 Testing CLP-specific queries...")
        
        test_queries = CLPTestQueries.get_all_test_queries()
        clp_results = []
        
        for test_query in test_queries:
            try:
                logger.info(f"Testing CLP query: '{test_query.query_text[:50]}...'")
                
                # Run RAG pipeline
                retrieved_contexts, synthesized_knowledge = self.rag_engine.retrieve_and_analyze(
                    query_text=test_query.query_text,
                    query_details={
                        "clp_domain_id": test_query.primary_domain.value,
                        "task": "CLP test query"
                    },
                    filters={"clp_domain_id": test_query.primary_domain.value} if test_query.primary_domain else None
                )
                
                # Evaluate quality
                quality_report = self.quality_metrics.evaluate_retrieval_quality(
                    test_query=test_query,
                    retrieved_contexts=retrieved_contexts,
                    synthesized_knowledge=synthesized_knowledge
                )
                
                clp_results.append({
                    "test_query": test_query,
                    "quality_report": quality_report,
                    "status": "success"
                })
                
                logger.info(f"✅ CLP query succeeded (Quality: {quality_report.overall_score:.2f})")
                
            except Exception as e:
                logger.error(f"❌ CLP query failed: {str(e)}")
                clp_results.append({
                    "test_query": test_query,
                    "error": str(e),
                    "status": "failed"
                })
        
        # Calculate CLP-specific statistics
        successful_tests = [r for r in clp_results if r.get("status") == "success"]
        quality_reports = [r["quality_report"] for r in successful_tests]
        quality_scores = [r.overall_score for r in quality_reports]
        
        return {
            "total_clp_queries": len(test_queries),
            "successful_queries": len(successful_tests),
            "avg_quality_score": np.mean(quality_scores) if quality_scores else 0,
            "quality_distribution": {
                "excellent": sum(1 for s in quality_scores if s >= 0.8),
                "good": sum(1 for s in quality_scores if 0.6 <= s < 0.8),
                "fair": sum(1 for s in quality_scores if 0.4 <= s < 0.6),
                "poor": sum(1 for s in quality_scores if s < 0.4)
            },
            "domain_performance": self._analyze_domain_performance(clp_results),
            "detailed_results": clp_results,
            "quality_reports": quality_reports
        }
    
    def _analyze_domain_performance(self, clp_results: List[Dict]) -> Dict[str, float]:
        """Analyze performance by CLP domain."""
        domain_scores = {}
        
        for result in clp_results:
            if result.get("status") == "success":
                domain = result["test_query"].primary_domain.value
                score = result["quality_report"].overall_score
                
                if domain not in domain_scores:
                    domain_scores[domain] = []
                domain_scores[domain].append(score)
        
        return {domain: np.mean(scores) for domain, scores in domain_scores.items()}
    
    def _calculate_summary_statistics(self, clp_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall summary statistics."""
        quality_reports = clp_results.get("quality_reports", [])
        
        if not quality_reports:
            return {
                "total_test_time": 0,
                "total_queries_tested": 0,
                "avg_response_time": 0,
                "overall_success_rate": 0,
                "avg_quality_score": 0
            }
        
        batch_stats = self.quality_metrics.calculate_batch_statistics(quality_reports)
        
        return {
            "total_queries_tested": clp_results.get("total_clp_queries", 0),
            "overall_success_rate": clp_results.get("successful_queries", 0) / clp_results.get("total_clp_queries", 1),
            "avg_quality_score": clp_results.get("avg_quality_score", 0),
            "batch_statistics": batch_stats
        }
    
    def _generate_overall_recommendations(self, clp_results: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations based on test results."""
        quality_reports = clp_results.get("quality_reports", [])
        
        if not quality_reports:
            return [
                "Complete comprehensive RAG testing to identify specific improvement areas",
                "Monitor retrieval quality metrics for ongoing optimization"
            ]
        
        improvement_areas = self.quality_metrics.identify_improvement_areas(quality_reports)
        
        recommendations = improvement_areas + [
            "Consider domain-specific tuning for underperforming CLP areas",
            "Implement automated RAG quality monitoring in production",
            "Regular evaluation of retrieval quality as knowledge base grows"
        ]
        
        return recommendations
    
    def _save_test_results(self, results: Dict[str, Any]):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"logs/rag_tests/rag_test_results_{timestamp}.json"
        
        try:
            # Convert non-serializable objects
            serializable_results = self._make_serializable(results)
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"💾 Test results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save test results: {str(e)}")
    
    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'value'):  # For Enums
            return obj.value
        else:
            return obj
    
    def _print_test_summary(self, results: Dict[str, Any]):
        """Print a comprehensive test summary."""
        print("\n" + "="*80)
        print("🎯 APEGA RAG ENGINE TEST SUMMARY")
        print("="*80)
        
        print(f"⏱️  Total Test Time: {results.get('total_time_seconds', 0):.2f} seconds")
        print(f"📊 Test Status: {results.get('status', 'unknown').upper()}")
        
        # Component test summary
        component_tests = results.get('component_tests', {})
        print(f"\n🔧 Component Tests:")
        for component, result in component_tests.items():
            if isinstance(result, dict) and 'status' in result:
                status = "✅" if result['status'] == 'success' else "❌"
                print(f"   {status} {component}: {result['status']}")
        
        # Pipeline test summary
        pipeline_tests = results.get('pipeline_tests', {})
        if pipeline_tests:
            success_rate = pipeline_tests.get('successful_tests', 0) / pipeline_tests.get('total_tests', 1) * 100
            print(f"\n🔄 Pipeline Tests:")
            print(f"   Success Rate: {success_rate:.1f}% ({pipeline_tests.get('successful_tests', 0)}/{pipeline_tests.get('total_tests', 0)})")
            print(f"   Avg Processing Time: {pipeline_tests.get('avg_processing_time', 0):.2f}s")
        
        # CLP query test summary
        clp_tests = results.get('clp_query_tests', {})
        if clp_tests:
            quality_score = clp_tests.get('avg_quality_score', 0)
            success_rate = clp_tests.get('successful_queries', 0) / clp_tests.get('total_clp_queries', 1) * 100
            print(f"\n📚 CLP Query Tests:")
            print(f"   Success Rate: {success_rate:.1f}% ({clp_tests.get('successful_queries', 0)}/{clp_tests.get('total_clp_queries', 0)})")
            print(f"   Avg Quality Score: {quality_score:.2f}/1.0")
            
            # Quality distribution
            qual_dist = clp_tests.get('quality_distribution', {})
            print(f"   Quality Distribution:")
            print(f"     Excellent (≥0.8): {qual_dist.get('excellent', 0)}")
            print(f"     Good (0.6-0.8): {qual_dist.get('good', 0)}")
            print(f"     Fair (0.4-0.6): {qual_dist.get('fair', 0)}")
            print(f"     Poor (<0.4): {qual_dist.get('poor', 0)}")
            
            # Domain performance
            domain_perf = clp_tests.get('domain_performance', {})
            if domain_perf:
                print(f"   Domain Performance:")
                for domain, score in domain_perf.items():
                    domain_name = domain.replace("Domain_", "").replace("_", " ")
                    print(f"     {domain_name}: {score:.2f}")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                print(f"   {i}. {rec}")
        
        print("\n" + "="*80)

def main():
    """Main entry point for RAG testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="APEGA RAG Engine Test Framework")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    args = parser.parse_args()
    
    # Initialize test framework
    test_framework = RAGTestFramework(verbose=args.verbose)
    
    # Run tests
    if args.quick:
        # Quick test implementation would go here
        print("Quick test mode not yet implemented")
    else:
        results = test_framework.run_comprehensive_test_suite()
        
        # Exit with appropriate code
        if results.get("status") == "completed":
            print("🎉 All RAG tests completed successfully!")
            sys.exit(0)
        else:
            print("❌ RAG tests failed.")
            sys.exit(1)

if __name__ == "__main__":
    main() 