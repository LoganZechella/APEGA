"""
Hybrid Searcher for APEGA.
Combines dense vector search and sparse keyword search for improved retrieval.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from loguru import logger
import numpy as np
from rank_bm25 import BM25Okapi

from src.knowledge_ingestion.embedding_generator import EmbeddingGenerator
from src.knowledge_ingestion.vector_db_manager import VectorDBManager
from src.models.data_models import RetrievedContext, TextChunk


class HybridSearcher:
    """
    Performs hybrid search by combining dense vector search and sparse keyword search.
    Uses Reciprocal Rank Fusion to combine results from both search methods.
    """
    
    def __init__(
        self,
        vector_db: VectorDBManager,
        embedding_generator: EmbeddingGenerator,
        top_k_dense: int = 10,
        top_k_sparse: int = 10,
        rrf_k: int = 60  # Constant for Reciprocal Rank Fusion
    ):
        """
        Initialize the HybridSearcher.
        
        Args:
            vector_db: VectorDBManager instance for vector search
            embedding_generator: EmbeddingGenerator for query embedding
            top_k_dense: Number of results from dense search
            top_k_sparse: Number of results from sparse search
            rrf_k: Constant for Reciprocal Rank Fusion (typically 60)
        """
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.rrf_k = rrf_k
        
        # Initialize BM25 index if needed for offline search
        self.bm25_index = None
        self.bm25_corpus = None
        self.bm25_corpus_ids = None
    
    def hybrid_search(
        self, 
        query_text: str, 
        top_k: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedContext]:
        """
        Perform hybrid search using both dense vectors and sparse keyword matching.
        
        Args:
            query_text: Text query
            top_k: Number of results to return
            filters: Optional filters to apply to the search
            
        Returns:
            List of RetrievedContext objects with combined and ranked results
        """
        logger.info(f"Performing hybrid search for query: {query_text}")
        
        # Get dense search results
        dense_results = self._dense_search(query_text, self.top_k_dense, filters)
        
        # Get sparse search results
        sparse_results = self._sparse_search(query_text, self.top_k_sparse, filters)
        
        # Combine results using Reciprocal Rank Fusion
        combined_results = self._reciprocal_rank_fusion(
            dense_results, 
            sparse_results, 
            top_k
        )
        
        return combined_results
    
    def _dense_search(
        self, 
        query_text: str, 
        top_k: int, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedContext]:
        """
        Perform dense vector search.
        
        Args:
            query_text: Text query
            top_k: Number of results to return
            filters: Optional filters to apply to the search
            
        Returns:
            List of RetrievedContext objects from dense search
        """
        try:
            # Generate embedding for the query text
            query_chunk = TextChunk(
                chunk_id="query",
                document_id="query",
                text=query_text,
                metadata={}
            )
            embedded_query = self.embedding_generator.generate_embeddings([query_chunk])
            
            if not embedded_query or embedded_query[0].embedding_vector is None:
                logger.warning("Failed to generate query embedding for dense search")
                return []
            
            # Perform vector search
            query_embedding = embedded_query[0].embedding_vector
            dense_results = self.vector_db.dense_vector_search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters
            )
            
            logger.info(f"Dense search returned {len(dense_results)} results")
            return dense_results
            
        except Exception as e:
            logger.error(f"Error in dense search: {str(e)}")
            return []
    
    def _sparse_search(
        self, 
        query_text: str, 
        top_k: int, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedContext]:
        """
        Perform sparse keyword search.
        
        Args:
            query_text: Text query
            top_k: Number of results to return
            filters: Optional filters to apply to the search
            
        Returns:
            List of RetrievedContext objects from sparse search
        """
        try:
            # First approach: Use Qdrant's text search capability if available
            if hasattr(self.vector_db, 'keyword_search'):
                sparse_results = self.vector_db.keyword_search(
                    query_text=query_text,
                    top_k=top_k,
                    filters=filters
                )
                
                if sparse_results:
                    logger.info(f"Sparse search (database) returned {len(sparse_results)} results")
                    return sparse_results
            
            # Fallback: Use BM25 for sparse search (requires corpus indexing)
            if self.bm25_index is None:
                logger.warning("BM25 index not initialized and database search failed. No sparse search results.")
                return []
            
            # Tokenize query
            query_tokens = query_text.lower().split()
            
            # Search with BM25
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top-k document indices
            top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
            
            # Construct retrieved contexts
            sparse_results = []
            for idx in top_indices:
                if bm25_scores[idx] > 0:  # Only include non-zero scores
                    chunk_id = self.bm25_corpus_ids[idx]
                    text = self.bm25_corpus[idx]
                    
                    # For BM25 results, we don't have all metadata
                    # In a production system, you would look up or store this information
                    context = RetrievedContext(
                        chunk_id=chunk_id,
                        document_id="unknown",  # Would need to be extracted from chunk_id
                        text=text,
                        initial_score=float(bm25_scores[idx]),
                        metadata={}
                    )
                    sparse_results.append(context)
            
            logger.info(f"Sparse search (BM25) returned {len(sparse_results)} results")
            return sparse_results
            
        except Exception as e:
            logger.error(f"Error in sparse search: {str(e)}")
            return []
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievedContext],
        sparse_results: List[RetrievedContext],
        top_k: int
    ) -> List[RetrievedContext]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        Args:
            dense_results: Results from dense vector search
            sparse_results: Results from sparse keyword search
            top_k: Number of final results to return
            
        Returns:
            Combined and re-ranked list of RetrievedContext objects
        """
        # Create a dictionary to store RRF scores by chunk_id
        rrf_scores = {}
        seen_chunks = {}
        
        # Process dense search results
        for rank, result in enumerate(dense_results):
            chunk_id = result.chunk_id
            # RRF formula: 1 / (rank + k)
            rrf_score = 1.0 / (rank + 1 + self.rrf_k)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score
            seen_chunks[chunk_id] = result
        
        # Process sparse search results
        for rank, result in enumerate(sparse_results):
            chunk_id = result.chunk_id
            rrf_score = 1.0 / (rank + 1 + self.rrf_k)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + rrf_score
            if chunk_id not in seen_chunks:
                seen_chunks[chunk_id] = result
        
        # Sort by RRF score
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Take top-k results
        top_chunk_ids = sorted_chunk_ids[:top_k]
        
        # Create final result list
        combined_results = []
        for chunk_id in top_chunk_ids:
            result = seen_chunks[chunk_id]
            # Update with RRF score
            result.initial_score = rrf_scores[chunk_id]
            combined_results.append(result)
        
        logger.info(f"Reciprocal Rank Fusion returned {len(combined_results)} results")
        return combined_results
    
    def index_corpus_for_bm25(self, corpus: List[str], corpus_ids: List[str]) -> None:
        """
        Index a corpus for offline BM25 search.
        
        Args:
            corpus: List of document texts
            corpus_ids: List of corresponding document IDs
        """
        # Tokenize corpus
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        
        # Create BM25 index
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self.bm25_corpus = corpus
        self.bm25_corpus_ids = corpus_ids
        
        logger.info(f"Indexed {len(corpus)} documents for BM25 search")
