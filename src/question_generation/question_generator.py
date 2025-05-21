"""
Question Generator for APEGA.
Generates multiple-choice questions using Google's Gemini 2.5 Pro.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import os
import json
import time
import google.generativeai as genai
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.models.data_models import GeneratedMCQ, Option, SynthesizedKnowledge


class QuestionGenerator:
    """
    Generates multiple-choice questions (MCQs) using Google's Gemini 2.5 Pro.
    Transforms synthesized knowledge into high-quality exam questions.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-pro-preview-05-06",
        temperature: float = 0.7,
        max_output_tokens: int = 8192,
        verbose: bool = False
    ):
        """
        Initialize the QuestionGenerator.
        
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
        self.generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": "application/json"
        }
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True
    )
    def generate_mcqs(
        self,
        prompt: str,
        synthesized_knowledge: SynthesizedKnowledge,
        num_questions: int,
        target_details: Dict[str, str]
    ) -> List[GeneratedMCQ]:
        """
        Generate multiple-choice questions based on synthesized knowledge.
        
        Args:
            prompt: Optimized prompt from PromptEngineer
            synthesized_knowledge: Synthesized knowledge from RAG system
            num_questions: Number of questions to generate
            target_details: Details about the target domain/task
            
        Returns:
            List of GeneratedMCQ objects
        """
        logger.info(f"Generating {num_questions} MCQs using Gemini 2.5 Pro")
        
        # Create final prompt with all necessary components
        domain_specific = f"for Domain {target_details.get('clp_domain_id', '')} {target_details.get('clp_domain_name', '')}"
        if target_details.get('clp_task_id') and target_details.get('clp_task_statement'):
            domain_specific += f", Task {target_details.get('clp_task_id')}: {target_details.get('clp_task_statement')}"
        
        # Substitute placeholders in prompt if needed
        final_prompt = prompt
        if "{num_questions}" in prompt:
            final_prompt = prompt.replace("{num_questions}", str(num_questions))
        if "{domain_specific}" in prompt:
            final_prompt = final_prompt.replace("{domain_specific}", domain_specific)
        if "{domain_id}" in prompt:
            final_prompt = final_prompt.replace("{domain_id}", target_details.get('clp_domain_id', ''))
        if "{domain_name}" in prompt:
            final_prompt = final_prompt.replace("{domain_name}", target_details.get('clp_domain_name', ''))
        if "{task_id}" in prompt:
            final_prompt = final_prompt.replace("{task_id}", target_details.get('clp_task_id', ''))
        if "{task_statement}" in prompt:
            final_prompt = final_prompt.replace("{task_statement}", target_details.get('clp_task_statement', ''))
        
        # Ensure the knowledge context is included
        if "{knowledge_context}" in prompt and synthesized_knowledge.summary:
            knowledge_summary = f"""Summary: {synthesized_knowledge.summary}\n\n"""
            
            if synthesized_knowledge.key_concepts:
                knowledge_summary += "Key Concepts:\n"
                for i, concept in enumerate(synthesized_knowledge.key_concepts):
                    knowledge_summary += f"- {concept.get('concept', '')}: {concept.get('explanation', '')}\n"
            
            if synthesized_knowledge.potential_exam_areas:
                knowledge_summary += "\nPotential Exam Areas:\n"
                for area in synthesized_knowledge.potential_exam_areas:
                    knowledge_summary += f"- {area}\n"
                    
            final_prompt = final_prompt.replace("{knowledge_context}", knowledge_summary)
        
        if self.verbose:
            logger.debug(f"Final prompt for MCQ generation: {final_prompt}")
        
        try:
            # Call Gemini API
            response = self.model.generate_content(final_prompt)
            
            if self.verbose:
                logger.debug(f"Gemini response: {response.text}")
            
            # Parse the response into GeneratedMCQ objects
            mcqs = self._parse_mcq_response(response.text, target_details)
            
            # If we didn't get enough questions, try to generate more
            if len(mcqs) < num_questions:
                logger.warning(f"Only generated {len(mcqs)} MCQs, expected {num_questions}")
                
                # Return what we have
                return mcqs
            
            logger.info(f"Successfully generated {len(mcqs)} MCQs")
            return mcqs
            
        except Exception as e:
            logger.error(f"Error generating MCQs: {str(e)}")
            raise
    
    def regenerate_single_mcq_with_feedback(
        self,
        original_mcq_data: Dict[str, Any],
        synthesized_knowledge_context: str,
        qa_feedback: str,
        target_details: Dict[str, Any],
        output_schema: Dict[str, Any]
    ) -> Optional[GeneratedMCQ]:
        """
        Regenerate a single MCQ based on QA feedback.
        
        Args:
            original_mcq_data: Data from the original MCQ
            synthesized_knowledge_context: The synthesized knowledge context
            qa_feedback: Feedback from the QA system
            target_details: Details about the target domain/task
            output_schema: Schema for the output format
            
        Returns:
            Regenerated GeneratedMCQ object or None if regeneration fails
        """
        logger.info(f"Regenerating MCQ based on QA feedback")
        
        # Create prompt for regeneration
        prompt = f"""
You are an expert Certified Licensing Professional (CLP) exam question writer. Your task is to revise a multiple-choice question based on quality assessment feedback.

## Original Question
Question ID: {original_mcq_data.get('question_id', '')}
Domain: {target_details.get('clp_domain_id', '')} - {target_details.get('clp_domain_name', '')}

Question Stem: {original_mcq_data.get('question_stem', '')}

Options:
{self._format_options(original_mcq_data.get('options', []))}

Correct Answer: {original_mcq_data.get('correct_option_id', '')}

Explanation: {original_mcq_data.get('explanation', '')}

## Quality Assessment Feedback
{qa_feedback}

## Knowledge Context
{synthesized_knowledge_context}

## Task
Revise the question to address all the issues identified in the feedback. You may:
1. Rewrite the question stem for clarity
2. Revise answer options to make them more plausible, distinct, or clear
3. Change the correct answer if necessary
4. Update the explanation
5. Make any other improvements needed to address the feedback

Base your revisions strictly on the provided knowledge context.

## Output Format
Provide your revised question as a single JSON object with this structure:
```json
{
  "question_id": "string (keep original ID)",
  "clp_domain_id": "string",
  "clp_domain_name": "string",
  "question_stem": "string",
  "options": [
    {"option_id": "A", "text": "string"},
    {"option_id": "B", "text": "string"},
    {"option_id": "C", "text": "string"},
    {"option_id": "D", "text": "string"}
  ],
  "correct_option_id": "string (A, B, C, or D)",
  "explanation": "string"
}
```
"""
        
        try:
            # Call Gemini API
            response = self.model.generate_content(prompt)
            
            if self.verbose:
                logger.debug(f"Gemini regeneration response: {response.text}")
            
            # Parse the response into a GeneratedMCQ object
            mcq = self._parse_single_mcq(response.text, target_details)
            
            if mcq:
                logger.info(f"Successfully regenerated MCQ")
                return mcq
            else:
                logger.warning(f"Failed to parse regenerated MCQ")
                return None
                
        except Exception as e:
            logger.error(f"Error regenerating MCQ: {str(e)}")
            return None
    
    def _parse_mcq_response(self, response_text: str, target_details: Dict[str, str]) -> List[GeneratedMCQ]:
        """
        Parse the Gemini response into GeneratedMCQ objects.
        
        Args:
            response_text: Response from Gemini
            target_details: Details about the target domain/task
            
        Returns:
            List of GeneratedMCQ objects
        """
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no JSON code block, try to parse the entire response
                json_str = response_text
            
            # Parse the JSON
            mcq_data = json.loads(json_str)
            
            # Handle both array and single object responses
            if isinstance(mcq_data, dict):
                mcq_data = [mcq_data]
            
            # Convert to GeneratedMCQ objects
            mcqs = []
            for i, data in enumerate(mcq_data):
                mcq = self._create_mcq_from_data(data, target_details, i)
                if mcq:
                    mcqs.append(mcq)
            
            return mcqs
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from response: {str(e)}")
            logger.debug(f"Problematic response: {response_text}")
            
            # Try to extract any valid questions with regex as a fallback
            return self._extract_questions_with_regex(response_text, target_details)
        except Exception as e:
            logger.error(f"Error parsing MCQ response: {str(e)}")
            return []
    
    def _parse_single_mcq(self, response_text: str, target_details: Dict[str, str]) -> Optional[GeneratedMCQ]:
        """
        Parse a single MCQ from the Gemini response.
        
        Args:
            response_text: Response from Gemini
            target_details: Details about the target domain/task
            
        Returns:
            GeneratedMCQ object or None if parsing fails
        """
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no JSON code block, try to parse the entire response
                json_str = response_text
            
            # Parse the JSON
            mcq_data = json.loads(json_str)
            
            # Create MCQ object
            return self._create_mcq_from_data(mcq_data, target_details, 0)
            
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error parsing single MCQ: {str(e)}")
            return None
    
    def _create_mcq_from_data(
        self, 
        data: Dict[str, Any], 
        target_details: Dict[str, str],
        index: int
    ) -> Optional[GeneratedMCQ]:
        """
        Create a GeneratedMCQ object from parsed data.
        
        Args:
            data: Parsed MCQ data
            target_details: Details about the target domain/task
            index: Index for generating a question ID if none provided
            
        Returns:
            GeneratedMCQ object or None if data is invalid
        """
        try:
            # Extract options
            options_data = data.get('options', [])
            options = []
            
            for option_data in options_data:
                option_id = option_data.get('option_id')
                text = option_data.get('text')
                
                if not option_id or not text:
                    continue
                    
                options.append(Option(option_id=option_id, text=text))
            
            # Ensure we have at least 2 options
            if len(options) < 2:
                logger.warning(f"MCQ has fewer than 2 options, skipping")
                return None
            
            # Generate a question ID if not provided
            question_id = data.get('question_id')
            if not question_id:
                domain_id = target_details.get('clp_domain_id', 'unknown')
                question_id = f"{domain_id}_Q{index + 1}"
            
            # Create GeneratedMCQ object
            mcq = GeneratedMCQ(
                question_id=question_id,
                clp_domain_id=data.get('clp_domain_id') or target_details.get('clp_domain_id', ''),
                clp_domain_name=data.get('clp_domain_name') or target_details.get('clp_domain_name', ''),
                clp_task_id=data.get('clp_task_id') or target_details.get('clp_task_id', ''),
                clp_task_statement=data.get('clp_task_statement') or target_details.get('clp_task_statement', ''),
                question_stem=data.get('question_stem', ''),
                options=options,
                correct_option_id=data.get('correct_option_id', ''),
                explanation=data.get('explanation', ''),
                difficulty_level_assessed=data.get('difficulty_level_assessed', ''),
                cognitive_skill_targeted=data.get('cognitive_skill_targeted', '')
            )
            
            return mcq
            
        except Exception as e:
            logger.error(f"Error creating MCQ from data: {str(e)}")
            return None
    
    def _extract_questions_with_regex(
        self, 
        response_text: str, 
        target_details: Dict[str, str]
    ) -> List[GeneratedMCQ]:
        """
        Attempt to extract questions using regex as a fallback.
        
        Args:
            response_text: Response from Gemini
            target_details: Details about the target domain/task
            
        Returns:
            List of GeneratedMCQ objects
        """
        logger.warning("Attempting to extract questions with regex as fallback")
        
        mcqs = []
        
        # Try to find question patterns
        question_pattern = r'#+\s*(?:Question|Q)\s*\d+.*?(?=#+\s*(?:Question|Q)|$)'
        question_matches = re.finditer(question_pattern, response_text, re.DOTALL)
        
        for i, match in enumerate(question_matches):
            question_text = match.group(0)
            
            try:
                # Extract question stem
                stem_match = re.search(r'(?:Question stem|Stem):\s*(.*?)(?:\n\n|\nOptions)', question_text, re.DOTALL)
                question_stem = stem_match.group(1).strip() if stem_match else ""
                
                # Extract options
                options = []
                option_matches = re.finditer(r'([A-D])\.?\s*(.*?)(?=\n[A-D]\.|\n\n|$)', question_text, re.DOTALL)
                
                for option_match in option_matches:
                    option_id = option_match.group(1)
                    option_text = option_match.group(2).strip()
                    options.append(Option(option_id=option_id, text=option_text))
                
                # Extract correct answer
                correct_match = re.search(r'(?:Correct answer|Answer):\s*([A-D])', question_text, re.IGNORECASE)
                correct_id = correct_match.group(1) if correct_match else ""
                
                # Extract explanation
                explanation_match = re.search(r'(?:Explanation|Rationale):\s*(.*?)(?:\n\n|$)', question_text, re.DOTALL)
                explanation = explanation_match.group(1).strip() if explanation_match else ""
                
                # Create MCQ if we have the minimum required fields
                if question_stem and options and correct_id:
                    mcq = GeneratedMCQ(
                        question_id=f"{target_details.get('clp_domain_id', 'unknown')}_Q{i + 1}",
                        clp_domain_id=target_details.get('clp_domain_id', ''),
                        clp_domain_name=target_details.get('clp_domain_name', ''),
                        clp_task_id=target_details.get('clp_task_id', ''),
                        clp_task_statement=target_details.get('clp_task_statement', ''),
                        question_stem=question_stem,
                        options=options,
                        correct_option_id=correct_id,
                        explanation=explanation
                    )
                    mcqs.append(mcq)
            except Exception as e:
                logger.error(f"Error extracting question with regex: {str(e)}")
                continue
        
        logger.info(f"Extracted {len(mcqs)} questions with regex")
        return mcqs
    
    def _format_options(self, options: List[Dict[str, str]]) -> str:
        """Format options for a prompt."""
        return "\n".join(f"{opt.get('option_id', '?')}. {opt.get('text', '')}" for opt in options)