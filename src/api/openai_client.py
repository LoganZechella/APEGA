"""
OpenAI API Client for APEGA.
Provides utility functions for interacting with OpenAI's API.
"""

from typing import Dict, Any, List, Optional, Union
from loguru import logger
import os
from openai import OpenAI
from openai import OpenAIError, RateLimitError, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential


class OpenAIClient:
    """
    Client for interacting with OpenAI's API.
    Provides utility functions for various AI models.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        max_retries: int = 5
    ):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            organization: OpenAI organization ID (defaults to environment variable)
            max_retries: Maximum number of retry attempts for API calls
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it via parameter or OPENAI_API_KEY environment variable.")
            
        self.organization = organization or os.getenv("OPENAI_ORGANIZATION")
        self.max_retries = max_retries
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            organization=self.organization
        )
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True
    )
    def generate_embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-3-large",
        dimensions: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            model: Name of the embedding model
            dimensions: Output dimensionality (optional)
            
        Returns:
            List of embedding vectors
        """
        try:
            # Prepare request
            request = {
                "model": model,
                "input": texts
            }
            if dimensions:
                request["dimensions"] = dimensions
            
            # Call OpenAI API
            response = self.client.embeddings.create(**request)
            
            # Extract embeddings from response
            embeddings = [data.embedding for data in response.data]
            
            return embeddings
            
        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {str(e)}. Retrying with exponential backoff.")
            raise
        except APITimeoutError as e:
            logger.warning(f"API timeout: {str(e)}. Retrying with exponential backoff.")
            raise
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True
    )
    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "o4-mini",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a chat completion.
        
        Args:
            messages: List of message objects (role, content)
            model: Name of the model
            temperature: Temperature for generation
            max_tokens: Maximum number of tokens to generate
            response_format: Format for the response (e.g., {"type": "json_object"})
            tools: List of tool objects for function calling
            
        Returns:
            Chat completion response
        """
        try:
            # Prepare request
            request = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            
            if max_tokens:
                request["max_tokens"] = max_tokens
                
            if response_format:
                request["response_format"] = response_format
                
            if tools:
                request["tools"] = tools
            
            # Call OpenAI API
            response = self.client.chat.completions.create(**request)
            
            # Return the complete response object
            return response
            
        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {str(e)}. Retrying with exponential backoff.")
            raise
        except APITimeoutError as e:
            logger.warning(f"API timeout: {str(e)}. Retrying with exponential backoff.")
            raise
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise