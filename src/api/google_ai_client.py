"""
Google AI Client for APEGA.
Provides utility functions for interacting with Google's AI APIs.
"""

from typing import Dict, Any, List, Optional, Union
from loguru import logger
import os
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential


class GoogleAIClient:
    """
    Client for interacting with Google's AI APIs.
    Provides utility functions for Gemini models.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 5
    ):
        """
        Initialize the Google AI client.
        
        Args:
            api_key: Google API key (defaults to environment variable)
            max_retries: Maximum number of retry attempts for API calls
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set it via parameter or GOOGLE_API_KEY environment variable.")
            
        self.max_retries = max_retries
        
        # Initialize Google AI client
        genai.configure(api_key=self.api_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True
    )
    def generate_content(
        self,
        prompt: str,
        model: str = "gemini-2.5-pro-preview-05-06",
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        response_mime_type: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate content using a Gemini model.
        
        Args:
            prompt: The prompt text
            model: Name of the Gemini model
            temperature: Temperature for generation
            max_output_tokens: Maximum number of tokens to generate
            response_mime_type: MIME type for the response (e.g., "application/json")
            tools: List of tool objects for function calling
            
        Returns:
            Generation response
        """
        try:
            # Configure generation parameters
            generation_config = {}
            
            if temperature is not None:
                generation_config["temperature"] = temperature
                
            if max_output_tokens:
                generation_config["max_output_tokens"] = max_output_tokens
                
            if response_mime_type:
                generation_config["response_mime_type"] = response_mime_type
            
            # Create the model
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config
            )
            
            # Generate content
            if tools:
                response = model_instance.generate_content(
                    prompt,
                    tools=tools
                )
            else:
                response = model_instance.generate_content(prompt)
            
            # Return the response
            return response
            
        except Exception as e:
            logger.error(f"Google AI API error: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True
    )
    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemini-2.5-pro-preview-05-06",
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        response_mime_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a chat response using a Gemini model.
        
        Args:
            messages: List of message objects (role, parts)
            model: Name of the Gemini model
            temperature: Temperature for generation
            max_output_tokens: Maximum number of tokens to generate
            response_mime_type: MIME type for the response (e.g., "application/json")
            
        Returns:
            Chat response
        """
        try:
            # Configure generation parameters
            generation_config = {}
            
            if temperature is not None:
                generation_config["temperature"] = temperature
                
            if max_output_tokens:
                generation_config["max_output_tokens"] = max_output_tokens
                
            if response_mime_type:
                generation_config["response_mime_type"] = response_mime_type
            
            # Create the model
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config
            )
            
            # Create chat session
            chat = model_instance.start_chat(history=[])
            
            # Process each message and add to chat
            for message in messages:
                role = message.get("role")
                content = message.get("content")
                
                if role == "user":
                    chat.send_message(content)
                elif role == "assistant":
                    # In a real implementation, we'd need to handle this properly
                    pass
            
            # Generate response for the last user message
            last_user_message = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            if last_user_message:
                response = chat.send_message(last_user_message.get("content", ""))
                return response
            else:
                raise ValueError("No user message found in the provided messages")
            
        except Exception as e:
            logger.error(f"Google AI API error: {str(e)}")
            raise