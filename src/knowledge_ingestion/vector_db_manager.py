"""
Vector Database Manager for APEGA.
Manages storage and retrieval of embeddings in Qdrant vector database.
"""

import os
from typing import List, Dict, Any, Optional, Union, Tuple
from loguru import logger
import qdrant_client
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
import time

from src.models.data_models import EmbeddedChunk, RetrievedContext


class VectorDBManager:
    """
    Manages storage and retrieval of embeddings in Qdrant vector database.
    Handles collection creation, indexing, and search operations.
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        vector_dimensions: int = 3072,
        distance_metric: str = "cosine"
    ):
        """
        Initialize the VectorDBManager.
        
        Args:
            url: Qdrant server URL (defaults to environment variable or localhost)
            api_key: Qdrant API key for cloud deployments (defaults to environment variable)
            collection_name: Name of the collection to use (defaults to environment variable)
            vector_dimensions: Dimensionality of embeddings
            distance_metric: Distance metric to use (cosine, euclid, or dot)
        """
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "clp_knowledge")
        self.vector_dimensions = vector_dimensions
        self.distance_metric = distance_metric
        
        # Initialize client
        self.client = qdrant_client.QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=60.0
        )
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self) -> None:
        """Ensure the specified collection exists, create it if not."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating new collection '{self.collection_name}'")
                
                # Create the collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_dimensions,
                        distance=self.distance_metric
                    )
                )
                
                # Create payload indexes for efficient filtering
                self._create_payload_indexes()
                
                logger.info(f"Collection '{self.collection_name}' created successfully")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise
    
    def _create_payload_indexes(self) -> None:
        """Create payload indexes for efficient filtering."""
        try:
            # Common fields to index
            index_fields = [
                ("document_id", "keyword"),
                ("chunk_type", "keyword"),
                ("clp_domain_id", "keyword"),
                ("clp_task_id", "keyword")
            ]
            
            for field_name, field_type in index_fields:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                
            logger.info(f"Created payload indexes for collection '{self.collection_name}'")
            
        except Exception as e:
            logger.warning(f"Error creating payload indexes: {str(e)}")
    
    def upsert_embeddings(self, embedded_chunks: List[EmbeddedChunk]) -> int:
        """
        Insert or update embeddings in the vector database.
        
        Args:
            embedded_chunks: List of EmbeddedChunk objects with generated embeddings
            
        Returns:
            Number of successfully upserted chunks
        """
        # Filter out chunks with missing embeddings
        valid_chunks = [chunk for chunk in embedded_chunks if chunk.embedding_vector is not None]
        
        if not valid_chunks:
            logger.warning("No valid chunks with embeddings to upsert")
            return 0
        
        try:
            # Prepare points for batch upsert
            points = []
            
            for chunk in valid_chunks:
                # Create a point ID from chunk_id
                point_id = chunk.chunk_id
                
                # Create payload with all metadata and additional fields
                payload = {
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "chunk_type": chunk.chunk_type,
                    **chunk.metadata
                }
                
                points.append(qmodels.PointStruct(
                    id=point_id,
                    vector=chunk.embedding_vector,
                    payload=payload
                ))
            
            # Batch upsert to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Successfully upserted {len(points)} points to collection '{self.collection_name}'")
            return len(points)
            
        except Exception as e:
            logger.error(f"Error upserting embeddings: {str(e)}")
            raise
    
    def delete_by_document_id(self, document_id: str) -> int:
        """
        Delete all points for a specific document_id.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            Number of deleted points
        """
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id",
                                match=qmodels.MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"Deleted {result.deleted} points for document_id '{document_id}'")
            return result.deleted
            
        except Exception as e:
            logger.error(f"Error deleting points for document_id '{document_id}': {str(e)}")
            raise
    
    def dense_vector_search(
        self, 
        query_embedding: List[float], 
        top_k: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedContext]:
        """
        Perform a dense vector search using the provided query embedding.
        
        Args:
            query_embedding: Vector embedding of the query
            top_k: Number of results to return
            filters: Optional filters to apply to the search
            
        Returns:
            List of RetrievedContext objects
        """
        try:
            # Prepare filter if provided
            filter_obj = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        # Handle list values (OR condition)
                        should_conditions = [
                            qmodels.FieldCondition(
                                key=key,
                                match=qmodels.MatchValue(value=v)
                            )
                            for v in value
                        ]
                        filter_conditions.append(qmodels.Filter(should=should_conditions))
                    else:
                        # Handle single values
                        filter_conditions.append(
                            qmodels.FieldCondition(
                                key=key,
                                match=qmodels.MatchValue(value=value)
                            )
                        )
                
                filter_obj = qmodels.Filter(must=filter_conditions)
            
            # Execute search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
                filter=filter_obj
            )
            
            # Convert to RetrievedContext objects
            results = []
            for hit in search_result:
                payload = hit.payload or {}
                
                # Extract text and metadata
                text = payload.get("text", "")
                document_id = payload.get("document_id", "")
                chunk_id = hit.id
                
                # Create metadata dictionary from payload
                metadata = {k: v for k, v in payload.items() if k not in ["text", "document_id"]}
                
                context = RetrievedContext(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=text,
                    initial_score=hit.score,
                    metadata=metadata
                )
                results.append(context)
            
            logger.info(f"Dense vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in dense vector search: {str(e)}")
            return []
    
    def keyword_search(
        self, 
        query_text: str, 
        top_k: int = 10, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedContext]:
        """
        Perform a keyword search using the provided query text.
        
        Args:
            query_text: Text query
            top_k: Number of results to return
            filters: Optional filters to apply to the search
            
        Returns:
            List of RetrievedContext objects
        """
        try:
            # Prepare filter conditions
            filter_conditions = []
            
            # Add filter conditions if provided
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        # Handle list values (OR condition)
                        should_conditions = [
                            qmodels.FieldCondition(
                                key=key,
                                match=qmodels.MatchValue(value=v)
                            )
                            for v in value
                        ]
                        filter_conditions.append(qmodels.Filter(should=should_conditions))
                    else:
                        # Handle single values
                        filter_conditions.append(
                            qmodels.FieldCondition(
                                key=key,
                                match=qmodels.MatchValue(value=value)
                            )
                        )
            
            # Create text search condition
            text_condition = qmodels.FieldCondition(
                key="text",
                match=qmodels.MatchText(text=query_text)
            )
            filter_conditions.append(text_condition)
            
            # Execute search
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qmodels.Filter(must=filter_conditions),
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )[0]  # scroll returns (points, next_page_offset)
            
            # Convert to RetrievedContext objects
            results = []
            for point in search_result:
                payload = point.payload or {}
                
                # Extract text and metadata
                text = payload.get("text", "")
                document_id = payload.get("document_id", "")
                chunk_id = point.id
                
                # Create metadata dictionary from payload
                metadata = {k: v for k, v in payload.items() if k not in ["text", "document_id"]}
                
                # Since keyword search doesn't return a score, use a default
                # In a real implementation, you might compute a relevance score (e.g., BM25)
                score = 0.5
                
                context = RetrievedContext(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=text,
                    initial_score=score,
                    metadata=metadata
                )
                results.append(context)
            
            logger.info(f"Keyword search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": info.name,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
                "payload_schema": info.payload_schema
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}
