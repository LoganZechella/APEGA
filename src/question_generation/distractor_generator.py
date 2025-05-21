"""
Distractor Generator for APEGA.
Specialized generator for multiple-choice question distractors.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import os
import google.generativeai as genai

from src.models.data_models import GeneratedMCQ, Option


class DistractorGenerator:
    """
    Specialized generator for multiple-choice question distractors.
    Can generate new distractors or improve existing ones.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-pro-preview-05-06",
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        verbose: bool = False
    ):
        """
        Initialize the DistractorGenerator.
        
        Args:
            api_key: Google API key (defaults to environment variable)
            model_name: Name of the Google Gemini model to use
            temperature: Temperature for the model (higher for more creative distractors)
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
    
    def generate_distractors(
        self,
        question_stem: str,
        correct_answer: str,
        context: str,
        num_distractors: int = 3,
        existing_distractors: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate plausible distractors for a multiple-choice question.
        
        Args:
            question_stem: The question stem
            correct_answer: The correct answer option
            context: The context or knowledge to base distractors on
            num_distractors: Number of distractors to generate
            existing_distractors: List of existing distractors to avoid duplicating
            
        Returns:
            List of generated distractor texts
        """
        logger.info(f"Generating {num_distractors} distractors for question")
        
        # Create prompt
        prompt = f"""
You are an expert Certified Licensing Professional (CLP) exam question writer specialized in creating plausible, challenging distractors for multiple-choice questions.

## Question Stem
{question_stem}

## Correct Answer
{correct_answer}

## Knowledge Context
{context}

## Task
Generate {num_distractors} plausible but definitively incorrect distractors for this question. 

Guidelines for high-quality distractors:
1. Each distractor must be clearly incorrect but seem plausible to someone with incomplete knowledge
2. Distractors should represent common misconceptions or errors in understanding the material
3. Distractors should be similar in length, complexity, and style to the correct answer
4. Avoid absolute terms like "always" or "never" unless the correct answer also uses them
5. Don't make distractors too obviously wrong - they should require thought to eliminate
6. Ensure all distractors are different from one another and from the correct answer
7. Base distractors strictly on the provided knowledge context

{"## Existing Distractors to Avoid\n" + "\n".join(existing_distractors) if existing_distractors else ""}

## Output Format
Provide your response as a JSON array of strings, each containing one distractor:
```json
[
  "First distractor text",
  "Second distractor text",
  "Third distractor text"
]
```
"""
        
        try:
            # Call Gemini API
            response = self.model.generate_content(prompt)
            
            if self.verbose:
                logger.debug(f"Gemini distractor response: {response.text}")
            
            # Parse the response
            import json
            import re
            
            # Try to extract JSON from the response
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no JSON code block, try to parse the entire response
                json_str = response.text
            
            # Parse the JSON
            distractor_data = json.loads(json_str)
            
            # Ensure we have the right format
            if not isinstance(distractor_data, list):
                logger.warning(f"Expected list of distractors, got {type(distractor_data)}")
                return []
            
            # Extract distractor texts
            distractors = [str(d) for d in distractor_data if d]
            
            # Limit to requested number
            distractors = distractors[:num_distractors]
            
            logger.info(f"Generated {len(distractors)} distractors")
            return distractors
            
        except Exception as e:
            logger.error(f"Error generating distractors: {str(e)}")
            return []
    
    def improve_distractors(
        self,
        mcq: GeneratedMCQ,
        context: str,
        feedback: str
    ) -> List[Option]:
        """
        Improve existing distractors based on feedback.
        
        Args:
            mcq: The multiple-choice question with existing distractors
            context: The context or knowledge to base improvements on
            feedback: Feedback about what's wrong with the current distractors
            
        Returns:
            List of improved Option objects
        """
        logger.info(f"Improving distractors based on feedback")
        
        # Find the correct answer option
        correct_option = None
        for option in mcq.options:
            if option.option_id == mcq.correct_option_id:
                correct_option = option
                break
        
        if not correct_option:
            logger.warning(f"Correct option not found in MCQ")
            return mcq.options
        
        # Get current distractor options
        distractors = [option for option in mcq.options if option.option_id != mcq.correct_option_id]
        
        # Format options for the prompt
        options_text = "\n".join(f"{opt.option_id}. {opt.text}" for opt in mcq.options)
        
        # Create prompt
        prompt = f"""
You are an expert Certified Licensing Professional (CLP) exam question writer specialized in improving distractors for multiple-choice questions.

## Current Question
Question Stem: {mcq.question_stem}

Options:
{options_text}

Correct Answer: {mcq.correct_option_id}. {correct_option.text}

## Knowledge Context
{context}

## Feedback on Current Distractors
{feedback}

## Task
Improve the distractors (incorrect options) based on the feedback while keeping the correct answer and question stem unchanged.

Guidelines for improvements:
1. Address all issues mentioned in the feedback
2. Ensure distractors are plausible but definitively incorrect
3. Make distractors challenging but fair
4. Keep the same option IDs (A, B, C, D)
5. Base distractors strictly on the provided knowledge context

## Output Format
Provide your improved options as a JSON array of objects, maintaining the current option IDs:
```json
[
  {"option_id": "A", "text": "Improved option A text"},
  {"option_id": "B", "text": "Improved option B text"},
  {"option_id": "C", "text": "Improved option C text"},
  {"option_id": "D", "text": "Improved option D text"}
]
```
"""
        
        try:
            # Call Gemini API
            response = self.model.generate_content(prompt)
            
            if self.verbose:
                logger.debug(f"Gemini improved distractors response: {response.text}")
            
            # Parse the response
            import json
            import re
            
            # Try to extract JSON from the response
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response.text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no JSON code block, try to parse the entire response
                json_str = response.text
            
            # Parse the JSON
            options_data = json.loads(json_str)
            
            # Ensure we have the right format
            if not isinstance(options_data, list):
                logger.warning(f"Expected list of options, got {type(options_data)}")
                return mcq.options
            
            # Create new options
            improved_options = []
            
            for option_data in options_data:
                option_id = option_data.get("option_id")
                text = option_data.get("text")
                
                if not option_id or not text:
                    continue
                
                # Keep the correct answer text unchanged
                if option_id == mcq.correct_option_id:
                    improved_options.append(Option(option_id=option_id, text=correct_option.text))
                else:
                    improved_options.append(Option(option_id=option_id, text=text))
            
            # Ensure we still have the correct number of options
            if len(improved_options) != len(mcq.options):
                logger.warning(f"Number of improved options ({len(improved_options)}) doesn't match original ({len(mcq.options)})")
                return mcq.options
            
            logger.info(f"Successfully improved distractors")
            return improved_options
            
        except Exception as e:
            logger.error(f"Error improving distractors: {str(e)}")
            return mcq.options