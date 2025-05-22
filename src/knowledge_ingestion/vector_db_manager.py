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
        distance_metric: str = "cosine"  # Keep input as string for simplicity
    ):
        """
        Initialize the VectorDBManager.
        
        Args:
            url: Qdrant server URL (defaults to environment variable or localhost)
            api_key: Qdrant API key for cloud deployments (defaults to environment variable)
            collection_name: Name of the collection to use (defaults to environment variable)
            vector_dimensions: Dimensionality of embeddings
            distance_metric: Distance metric to use (cosine, euclid, or dot) - will be converted to enum
        """
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "clp_knowledge")
        self.vector_dimensions = vector_dimensions
        
        # Convert string distance_metric to qmodels.Distance enum
        try:
            self.distance_metric_enum = qmodels.Distance[distance_metric.upper()]
        except KeyError:
            logger.error(f"Invalid distance metric '{distance_metric}'. Defaulting to COSINE. Valid options are 'cosine', 'euclid', 'dot', 'manhattan'.")
            self.distance_metric_enum = qmodels.Distance.COSINE
        
        # For local Qdrant instances, clear the API key if it looks like a cloud key
        if self.api_key and ("|" in self.api_key or len(self.api_key) > 50):
            if "localhost" in self.url or "127.0.0.1" in self.url:
                logger.info("Detected local Qdrant URL with cloud API key - removing API key for local connection")
                self.api_key = None
        
        logger.info(f"Initializing QdrantClient with URL: '{self.url}'")
        if self.api_key:
            logger.info("Using API key for authentication")
        else:
            logger.info("No API key - connecting to local Qdrant instance")
        
        # Initialize client with parsed URL to support HTTP without SSL errors
        try:
            import urllib.parse
            parsed = urllib.parse.urlparse(self.url)
            host = parsed.hostname or self.url.replace("http://", "").replace("https://", "")
            port = parsed.port or 6333
            use_https = parsed.scheme == 'https'

            # Create client with improved error handling
            self.client = qdrant_client.QdrantClient(
                host=host,
                port=port,
                api_key=self.api_key,
                timeout=30.0,
                https=use_https,
                prefer_grpc=False  # Use HTTP instead of gRPC for better compatibility
            )
            
            # Test connection
            logger.info("Testing Qdrant connection...")
            collections = self.client.get_collections()
            logger.info(f"Successfully connected to Qdrant. Found {len(collections.collections)} existing collections.")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            logger.error("Please ensure:")
            logger.error("1. Qdrant is running (e.g., via Docker: docker run -p 6333:6333 qdrant/qdrant)")
            logger.error("2. The QDRANT_URL is correct")
            logger.error("3. For local instances, no API key should be set")
            raise
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self) -> None:
        """Ensure the specified collection exists, create it if not."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating new collection '{self.collection_name}' with {self.vector_dimensions} dimensions using {self.distance_metric_enum.value} distance")
                
                # Create the collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_dimensions,
                        distance=self.distance_metric_enum # Use the enum here
                    )
                )
                
                # Create payload indexes for efficient filtering
                self._create_payload_indexes()
                
                logger.info(f"Collection '{self.collection_name}' created successfully")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
                # Verify collection configuration
                collection_info = self.client.get_collection(self.collection_name)
                actual_size = collection_info.config.params.vectors.size
                actual_distance = collection_info.config.params.vectors.distance
                
                if actual_size != self.vector_dimensions:
                    logger.warning(f"Collection vector size ({actual_size}) doesn't match expected size ({self.vector_dimensions})")
                if actual_distance != self.distance_metric_enum:
                    logger.warning(f"Collection distance metric ({(actual_distance.value if hasattr(actual_distance, 'value') else actual_distance)}) doesn't match expected metric ({self.distance_metric_enum.value})")
                
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
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=field_type
                    )
                    logger.debug(f"Created index for field '{field_name}'")
                except Exception as e:
                    # Index might already exist, which is fine
                    logger.debug(f"Index creation for '{field_name}' returned: {str(e)}")
                
            logger.info(f"Payload indexes configured for collection '{self.collection_name}'")
            
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
        
        logger.info(f"Upserting {len(valid_chunks)} chunks to Qdrant...")
        
        try:
            # Prepare points for batch upsert
            points = []
            
            for chunk in valid_chunks:
                # Create a point ID from chunk_id (ensure it's a string)
                point_id = str(chunk.chunk_id)
                
                # Create payload with all metadata and additional fields
                payload = {
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "chunk_type": str(chunk.chunk_type),
                    **chunk.metadata
                }
                
                # Ensure all payload values are JSON serializable
                cleaned_payload = {}
                for key, value in payload.items():
                    if value is not None:
                        cleaned_payload[key] = value
                
                points.append(qmodels.PointStruct(
                    id=point_id,
                    vector=chunk.embedding_vector,
                    payload=cleaned_payload
                ))
            
            # Batch upsert to Qdrant with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Attempt {attempt + 1} to upsert {len(points)} points to collection '{self.collection_name}'")
                    result = self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    
                    logger.info(f"Successfully upserted {len(points)} points to collection '{self.collection_name}'. Result: {result}")
                    return len(points)
                    
                except UnexpectedResponse as ue:
                    logger.error(f"Qdrant UnexpectedResponse during upsert attempt {attempt + 1}/{max_retries}: {ue.status_code} - {ue.content}")
                    logger.error(f"Content of unexpected response: {ue.content.decode() if isinstance(ue.content, bytes) else ue.content}")
                    if attempt == max_retries - 1:
                        logger.error("Max retries reached for Qdrant UnexpectedResponse.")
                        raise
                    time.sleep(2 ** attempt) # Exponential backoff
                except qdrant_client.http.exceptions.ResponseHandlingException as rhe:
                    logger.error(f"Qdrant ResponseHandlingException during upsert attempt {attempt + 1}/{max_retries}: {rhe}")
                    if attempt == max_retries - 1:
                        logger.error("Max retries reached for Qdrant ResponseHandlingException.")
                        raise
                    time.sleep(2 ** attempt) # Exponential backoff
                except Exception as e:
                    logger.error(f"Generic error during upsert attempt {attempt + 1}/{max_retries}: {str(e)}", exc_info=True)
                    if attempt == max_retries - 1:
                        logger.error("Max retries reached for generic error during upsert.")
                        raise
                    time.sleep(2 ** attempt) # Exponential backoff for other errors too
            
        except Exception as e:
            # This outer except block catches errors from preparing points or if all retries failed.
            logger.error(f"Error preparing or upserting embeddings after all retries: {str(e)}", exc_info=True)
    
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
            
            deleted_count = getattr(result, 'deleted', 0)
            logger.info(f"Deleted {deleted_count} points for document_id '{document_id}'")
            return deleted_count
            
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
                chunk_id = str(hit.id)
                
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
                chunk_id = str(point.id)
                
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