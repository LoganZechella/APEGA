"""
Deep Analyzer for APEGA.
Performs in-depth analysis of retrieved content using Google's Gemini 2.5 Pro.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import os
import google.generativeai as genai

from src.models.data_models import RetrievedContext, SynthesizedKnowledge


class DeepAnalyzer:
    """
    Performs in-depth analysis of retrieved content using Google's Gemini 2.5 Pro.
    Synthesizes knowledge from retrieved contexts to provide a deeper understanding.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-pro-preview-05-06",
        temperature: float = 0.2,
        max_output_tokens: int = 8192,
        verbose: bool = False
    ):
        """
        Initialize the DeepAnalyzer.
        
        Args:
            api_key: Google API key (defaults to environment variable)
            model_name: Name of the Google Gemini model to use
            temperature: Temperature for the model
            max_output_tokens: Maximum number of output tokens
            verbose: Whether to log detailed output
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set it via parameter or GOOGLE_API_KEY environment variable.")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.verbose = verbose
        
        # Initialize Google Gemini API
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": "application/json"
        }
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
        )
    
    def synthesize_knowledge(
        self,
        query_details: Dict[str, Any],
        contexts: List[RetrievedContext]
    ) -> SynthesizedKnowledge:
        """
        Synthesize knowledge from retrieved contexts.
        
        Args:
            query_details: Details about the query (e.g., target CLP domain/task)
            contexts: List of retrieved and re-ranked contexts
            
        Returns:
            SynthesizedKnowledge object with analyzed and synthesized content
        """
        if not contexts:
            logger.warning("No contexts provided for knowledge synthesis")
            return SynthesizedKnowledge(
                summary="No context available for synthesis.",
                source_chunk_ids=[]
            )
        
        logger.info(f"Synthesizing knowledge from {len(contexts)} contexts")
        
        # Extract text from contexts and create a consolidated input
        context_texts = [f"Context {i+1}:\n{ctx.text}\n" for i, ctx in enumerate(contexts)]
        consolidated_context = "\n\n".join(context_texts)
        
        # Create prompt for Gemini
        prompt = self._create_synthesis_prompt(query_details, consolidated_context)
        
        try:
            # Call Gemini API
            response = self.model.generate_content(prompt)
            
            if self.verbose:
                logger.debug(f"Gemini response: {response.text}")
            
            # Parse the response into a SynthesizedKnowledge object
            synthesized = self._parse_synthesis_response(response.text, contexts)
            
            logger.info(f"Successfully synthesized knowledge with {len(synthesized.key_concepts)} key concepts")
            return synthesized
            
        except Exception as e:
            logger.error(f"Error in knowledge synthesis: {str(e)}")
            # Return minimal synthesized knowledge in case of error
            return SynthesizedKnowledge(
                summary=f"Error during knowledge synthesis: {str(e)}",
                source_chunk_ids=[ctx.chunk_id for ctx in contexts]
            )
    
    def _create_synthesis_prompt(self, query_details: Dict[str, Any], consolidated_context: str) -> str:
        """
        Create a prompt for Gemini to synthesize knowledge.
        
        Args:
            query_details: Details about the query
            consolidated_context: Combined text from all contexts
            
        Returns:
            Prompt for Gemini
        """
        # Extract relevant details
        domain_id = query_details.get("clp_domain_id", "")
        task_id = query_details.get("clp_task_id", "")
        task_description = query_details.get("task", "Generate exam questions")
        
        # Build the prompt with appropriate instructions
        prompt = f"""
Task: Comprehensive Analysis and Synthesis of CLP Source Material for Exam Preparation

Role: You are an expert curriculum developer and subject matter expert for the Certified Licensing Professional (CLP) exam. You excel at in-depth analysis, synthesis, and integration of complex concepts.

Context: The following text segments have been retrieved as relevant to the CLP exam {"for " + domain_id if domain_id else ""} {"specifically for task " + task_id if task_id else ""}.

{consolidated_context}

Instructions:
1. Thoroughly analyze the provided text segments to understand the core concepts, principles, frameworks, and procedures described.
2. Cross-reference information across all segments to identify connections, complementary ideas, and potential contradictions.
3. Synthesize this information into a comprehensive, structured knowledge representation.
4. Focus on the following aspects:
   - Core principles, definitions, and legal frameworks
   - Key factors to consider (e.g., legal, commercial, resource-based)
   - Procedural steps or decision-making processes
   - Relationships and distinctions between different concepts
   - Potential areas of ambiguity or complexity that might be tested in the exam
5. Highlight areas that appear to be of high importance or complexity, suggesting they are likely candidates for challenging examination questions.

Output: Provide your synthesized knowledge in JSON format with the following structure:
{{
  "summary": "<A comprehensive summary of the core knowledge, 1-2 paragraphs>",
  "key_concepts": [
    {{
      "concept": "<Name or title of the concept>",
      "explanation": "<Clear, concise explanation>",
      "importance": "<Why this is important for the CLP exam>"
    }},
    ...
  ],
  "potential_exam_areas": [
    "<Topic or scenario that would make for a good exam question>",
    ...
  ]
}}

Ensure your analysis is comprehensive yet focused, emphasizing content that would be valuable for {task_description}.
"""
        return prompt
    
    def _parse_synthesis_response(
        self, 
        response_text: str, 
        contexts: List[RetrievedContext]
    ) -> SynthesizedKnowledge:
        """
        Parse the Gemini response into a SynthesizedKnowledge object.
        
        Args:
            response_text: Response from Gemini
            contexts: Original contexts used for synthesis
            
        Returns:
            SynthesizedKnowledge object
        """
        try:
            # Try to parse as JSON
            import json
            response_json = json.loads(response_text)
            
            # Extract fields from the response
            summary = response_json.get("summary", "")
            key_concepts = response_json.get("key_concepts", [])
            potential_exam_areas = response_json.get("potential_exam_areas", [])
            
            # Create SynthesizedKnowledge object
            return SynthesizedKnowledge(
                summary=summary,
                key_concepts=key_concepts,
                potential_exam_areas=potential_exam_areas,
                source_chunk_ids=[ctx.chunk_id for ctx in contexts]
            )
            
        except json.JSONDecodeError:
            # If not valid JSON, use the text as-is for the summary
            logger.warning("Gemini response is not valid JSON. Using as plain text.")
            return SynthesizedKnowledge(
                summary=response_text,
                source_chunk_ids=[ctx.chunk_id for ctx in contexts]
            )
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {str(e)}")
            return SynthesizedKnowledge(
                summary=f"Error parsing synthesis response: {str(e)}",
                source_chunk_ids=[ctx.chunk_id for ctx in contexts]
            )