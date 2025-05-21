"""
RAG Engine Module for APEGA.
Combines hybrid search, re-ranking, and deep analysis components.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from src.rag_engine.hybrid_searcher import HybridSearcher
from src.rag_engine.reranker import ReRanker
from src.rag_engine.deep_analyzer import DeepAnalyzer
from src.knowledge_ingestion.embedding_generator import EmbeddingGenerator
from src.knowledge_ingestion.vector_db_manager import VectorDBManager
from src.models.data_models import RetrievedContext, SynthesizedKnowledge


class RAGEngine:
    """
    Main RAG (Retrieval Augmented Generation) engine for APEGA.
    Orchestrates the process of retrieving, re-ranking, and analyzing context for question generation.
    """
    
    def __init__(
        self,
        vector_db: VectorDBManager,
        embedding_generator: EmbeddingGenerator,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        top_k_dense: int = 10,
        top_k_sparse: int = 10,
        top_k_rerank: int = 5
    ):
        """
        Initialize the RAG Engine.
        
        Args:
            vector_db: Vector database manager
            embedding_generator: Embedding generator
            openai_api_key: OpenAI API key
            google_api_key: Google API key
            top_k_dense: Number of results from dense search
            top_k_sparse: Number of results from sparse search
            top_k_rerank: Number of results after re-ranking
        """
        self.hybrid_searcher = HybridSearcher(
            vector_db=vector_db,
            embedding_generator=embedding_generator,
            top_k_dense=top_k_dense,
            top_k_sparse=top_k_sparse
        )
        
        self.reranker = ReRanker(
            api_key=openai_api_key
        )
        
        self.deep_analyzer = DeepAnalyzer(
            api_key=google_api_key
        )
        
        self.top_k_rerank = top_k_rerank
    
    def retrieve_and_analyze(
        self,
        query_text: str,
        query_details: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[RetrievedContext], SynthesizedKnowledge]:
        """
        Retrieve, re-rank, and synthesize knowledge for a query.
        
        Args:
            query_text: The query text
            query_details: Additional details about the query (e.g., CLP domain/task)
            filters: Optional filters for the search
            
        Returns:
            Tuple of (retrieved_contexts, synthesized_knowledge)
        """
        logger.info(f"Processing query: {query_text}")
        
        # 1. Retrieve initial contexts with hybrid search
        retrieved_contexts = self.hybrid_searcher.hybrid_search(
            query_text=query_text,
            filters=filters
        )
        
        if not retrieved_contexts:
            logger.warning("No contexts retrieved from hybrid search")
            empty_knowledge = SynthesizedKnowledge(
                summary="No relevant information found.",
                source_chunk_ids=[]
            )
            return [], empty_knowledge
        
        # 2. Re-rank contexts for improved relevance
        reranked_contexts = self.reranker.rerank_contexts(
            query_text=query_text,
            contexts=retrieved_contexts,
            top_n=self.top_k_rerank
        )
        
        if not reranked_contexts:
            logger.warning("No contexts after re-ranking")
            empty_knowledge = SynthesizedKnowledge(
                summary="No relevant information found after re-ranking.",
                source_chunk_ids=[]
            )
            return retrieved_contexts, empty_knowledge
        
        # 3. Perform deep analysis on re-ranked contexts
        synthesized_knowledge = self.deep_analyzer.synthesize_knowledge(
            query_details=query_details,
            contexts=reranked_contexts
        )
        
        logger.info("Completed retrieval and analysis process")
        return retrieved_contexts, synthesized_knowledge
