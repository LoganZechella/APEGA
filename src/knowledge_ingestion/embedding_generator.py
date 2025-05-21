"""
Embedding Generator for APEGA.
Generates embeddings for text chunks using OpenAI's text-embedding-3-large model.
"""

import time
import random
from typing import List, Dict, Any, Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os

from openai import OpenAI, RateLimitError, APIError, APITimeoutError

from src.models.data_models import TextChunk, EmbeddedChunk


class EmbeddingGenerator:
    """
    Generates vector embeddings for text chunks using OpenAI's text-embedding-3-large model.
    Includes robust error handling and rate limit management.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-3-large",
        dimensions: int = 3072,
        max_retries: int = 5
    ):
        """
        Initialize the EmbeddingGenerator.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model_name: Name of the embedding model
            dimensions: Dimensionality of embeddings (native 3072 for text-embedding-3-large)
            max_retries: Maximum number of retry attempts for API calls
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it via parameter or OPENAI_API_KEY environment variable.")
            
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.dimensions = dimensions
        self.max_retries = max_retries
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        reraise=True
    )
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts using the OpenAI API.
        Uses tenacity for retries with exponential backoff.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
                dimensions=self.dimensions
            )
            
            # Extract embeddings from response
            embeddings = [data.embedding for data in response.data]
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_embeddings(self, text_chunks: List[TextChunk], batch_size: int = 100) -> List[EmbeddedChunk]:
        """
        Generate embeddings for a list of text chunks.
        Processes in batches to avoid hitting rate limits.
        
        Args:
            text_chunks: List of TextChunk objects to embed
            batch_size: Number of chunks to process in each batch
            
        Returns:
            List of EmbeddedChunk objects with generated embeddings
        """
        embedded_chunks = []
        
        # Process in batches
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i+batch_size]
            
            # Extract text content from chunks
            texts = [chunk.text for chunk in batch]
            
            try:
                # Get embeddings for the batch
                embeddings = self._get_embeddings_batch(texts)
                
                # Create EmbeddedChunk objects
                for chunk, embedding in zip(batch, embeddings):
                    embedded_chunk = EmbeddedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        text=chunk.text,
                        embedding_vector=embedding,
                        chunk_type=chunk.chunk_type,
                        metadata=chunk.metadata
                    )
                    embedded_chunks.append(embedded_chunk)
                
                logger.info(f"Successfully embedded batch {i//batch_size + 1} ({len(batch)} chunks)")
                
            except Exception as e:
                logger.error(f"Failed to embed batch {i//batch_size + 1}: {str(e)}")
                
                # Add chunks with None embeddings to indicate failure
                for chunk in batch:
                    embedded_chunk = EmbeddedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        text=chunk.text,
                        embedding_vector=None,  # Indicates embedding failure
                        chunk_type=chunk.chunk_type,
                        metadata=chunk.metadata
                    )
                    embedded_chunks.append(embedded_chunk)
        
        # Log summary
        successful_embeddings = sum(1 for chunk in embedded_chunks if chunk.embedding_vector is not None)
        logger.info(f"Generated embeddings for {successful_embeddings}/{len(embedded_chunks)} chunks")
        
        return embedded_chunks
