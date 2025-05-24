"""
Re-Ranker for APEGA.
Re-ranks search results for improved relevance using LLM-based evaluation.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import os
from openai import OpenAI

from src.models.data_models import RetrievedContext


class ReRanker:
    """
    Re-ranks search results for improved relevance using LLM-based evaluation.
    Uses OpenAI's o4-mini model to score the relevance of retrieved contexts.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "o4-mini",
        temperature: float = 1,
        verbose: bool = False
    ):
        """
        Initialize the ReRanker.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model_name: Name of the OpenAI model to use
            temperature: Temperature for the model (lower for more consistent scoring)
            verbose: Whether to log detailed output
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it via parameter or OPENAI_API_KEY environment variable.")
        
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
    
    def rerank_contexts(
        self, 
        query_text: str, 
        contexts: List[RetrievedContext], 
        top_n: int = 5
    ) -> List[RetrievedContext]:
        """
        Re-rank search results based on relevance to the query.
        
        Args:
            query_text: The original query text
            contexts: List of retrieved contexts to re-rank
            top_n: Number of top results to return after re-ranking
            
        Returns:
            Re-ranked list of RetrievedContext objects
        """
        if not contexts:
            logger.warning("No contexts to re-rank")
            return []
        
        logger.info(f"Re-ranking {len(contexts)} contexts")
        
        # Relevance scoring using LLM
        scored_contexts = self._score_contexts_with_llm(query_text, contexts)
        
        # Sort by LLM-assigned relevance score
        sorted_contexts = sorted(scored_contexts, key=lambda x: x.rerank_score or 0, reverse=True)
        
        # Return top N results
        top_contexts = sorted_contexts[:top_n]
        
        logger.info(f"Re-ranking complete. Selected top {len(top_contexts)} contexts.")
        return top_contexts
    
    def _score_contexts_with_llm(
        self, 
        query_text: str, 
        contexts: List[RetrievedContext]
    ) -> List[RetrievedContext]:
        """
        Score contexts for relevance using an LLM.
        
        Args:
            query_text: The original query text
            contexts: List of retrieved contexts to score
            
        Returns:
            List of RetrievedContext objects with rerank_score set
        """
        scored_contexts = []
        
        for i, context in enumerate(contexts):
            try:
                # Create prompt for the LLM to evaluate relevance
                prompt = self._create_scoring_prompt(query_text, context.text)
                
                # Call OpenAI API with JSON response format
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that evaluates the relevance of text passages to a query."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    max_completion_tokens=150
                )
                
                # Extract score from response
                response_text = response.choices[0].message.content
                
                if self.verbose:
                    logger.debug(f"Response for context {i}: {response_text}")
                
                # Parse the score from the JSON response
                import json
                try:
                    response_json = json.loads(response_text)
                    score = float(response_json.get("relevance_score", 0))
                    # Ensure score is within 0-1 range
                    score = max(0, min(1, score))
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Error parsing score: {e}. Setting score to 0.")
                    score = 0
                
                # Update context with re-ranking score
                context.rerank_score = score
                scored_contexts.append(context)
                
                if self.verbose:
                    logger.debug(f"Scored context {i}: {score}")
                
            except Exception as e:
                logger.error(f"Error scoring context {i}: {str(e)}")
                # If scoring fails, keep the original context but with a low score
                context.rerank_score = 0.01
                scored_contexts.append(context)
        
        return scored_contexts
    
    def _create_scoring_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the LLM to score the relevance of a context to a query.
        
        Args:
            query: The query text
            context: The context text to evaluate
            
        Returns:
            Prompt for the LLM
        """
        return f"""
Evaluate how relevant the following passage is to the query. 
The passage should be considered highly relevant if it contains information that 
would be useful for generating an accurate, comprehensive response to the query,
especially in the context of creating Certified Licensing Professional (CLP) exam questions.

Query: {query}

Passage:
{context}

Return a JSON object with the following schema:
{{
    "relevance_score": <float between 0 and 1>,
    "justification": <brief explanation of why this score was assigned>
}}

Where 0 means completely irrelevant and 1 means highly relevant.
"""