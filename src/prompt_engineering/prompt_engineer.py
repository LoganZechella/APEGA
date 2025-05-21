"""
Prompt Engineer for APEGA.
Generates optimized prompts for downstream LLMs using meta-prompting.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import os
import json
from openai import OpenAI

from src.models.data_models import SynthesizedKnowledge


class PromptEngineer:
    """
    Generates and refines prompts for LLMs using meta-prompting.
    Uses OpenAI's o4-mini model to create optimal prompts for question generation and QA.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "o4-mini",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        verbose: bool = False
    ):
        """
        Initialize the PromptEngineer.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model_name: Name of the OpenAI model to use
            temperature: Temperature for the model
            max_tokens: Maximum number of tokens for generated prompts
            verbose: Whether to log detailed output
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it via parameter or OPENAI_API_KEY environment variable.")
            
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Cache of generated prompts to avoid redundant generation
        self.prompt_cache = {}
    
    def generate_llm_prompt(
        self,
        task_description: str,
        target_llm_info: Dict[str, str],
        output_format_schema: Dict[str, Any],
        num_prompts: int = 1
    ) -> List[str]:
        """
        Generate prompts for a downstream LLM.
        
        Args:
            task_description: Description of the task for the target LLM
            target_llm_info: Information about the target LLM
            output_format_schema: Schema for the desired output format
            num_prompts: Number of prompts to generate
            
        Returns:
            List of generated prompts
        """
        # Check if we have a cached prompt for this task
        cache_key = f"{task_description}_{target_llm_info.get('name')}_{json.dumps(output_format_schema)}"
        if cache_key in self.prompt_cache:
            logger.info("Using cached prompt")
            return [self.prompt_cache[cache_key]]
        
        logger.info(f"Generating prompt for task: {task_description}")
        
        # Create meta-prompt
        meta_prompt = self._create_meta_prompt(task_description, target_llm_info, output_format_schema, num_prompts)
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert AI Prompt Engineer who excels at creating effective prompts for other language models."},
                    {"role": "user", "content": meta_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            generated_text = response.choices[0].message.content
            
            if self.verbose:
                logger.debug(f"Generated prompt: {generated_text}")
            
            # Parse the generated prompts
            prompts = self._parse_generated_prompts(generated_text, num_prompts)
            
            # Cache the first prompt
            if prompts:
                self.prompt_cache[cache_key] = prompts[0]
            
            logger.info(f"Generated {len(prompts)} prompts")
            return prompts
            
        except Exception as e:
            logger.error(f"Error generating prompts: {str(e)}")
            # Return a simple fallback prompt
            fallback_prompt = self._create_fallback_prompt(task_description, target_llm_info, output_format_schema)
            return [fallback_prompt]
    
    def refine_llm_prompt(
        self,
        original_prompt: str,
        qa_feedback: str,
        task_description: str,
        target_llm_info: Dict[str, str]
    ) -> str:
        """
        Refine a prompt based on QA feedback.
        
        Args:
            original_prompt: The original prompt to refine
            qa_feedback: Feedback from the QA system
            task_description: Description of the task
            target_llm_info: Information about the target LLM
            
        Returns:
            Refined prompt
        """
        logger.info("Refining prompt based on QA feedback")
        
        # Create meta-prompt for refinement
        meta_prompt = f"""
You are an expert AI Prompt Engineer. Your task is to refine an existing prompt based on quality assurance feedback.

Original Prompt:
---
{original_prompt}
---

QA Feedback:
---
{qa_feedback}
---

Task Description: {task_description}

Target LLM: {target_llm_info.get('name', 'Unknown')}
Target LLM Capabilities: {target_llm_info.get('capabilities', '')}

Analyze the feedback and original prompt, then generate an improved version that addresses the issues identified in the feedback. The refined prompt should:
1. Maintain the overall structure and purpose of the original prompt
2. Specifically address the problems mentioned in the QA feedback
3. Be clearer and more specific where ambiguity existed
4. Include additional constraints or guidelines to prevent the issues from recurring

Refined Prompt:
"""
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert AI Prompt Engineer who excels at refining prompts based on feedback."},
                    {"role": "user", "content": meta_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            refined_prompt = response.choices[0].message.content.strip()
            
            if self.verbose:
                logger.debug(f"Refined prompt: {refined_prompt}")
            
            logger.info("Successfully refined prompt")
            return refined_prompt
            
        except Exception as e:
            logger.error(f"Error refining prompt: {str(e)}")
            # Return the original prompt if refinement fails
            return original_prompt
    
    def generate_question_prompt(
        self,
        domain_info: Dict[str, str],
        synthesized_knowledge: SynthesizedKnowledge,
        target_llm_info: Dict[str, str],
        num_questions: int = 5,
        difficulty_level: str = "mixed"
    ) -> str:
        """
        Generate a prompt specifically for question generation.
        
        Args:
            domain_info: Information about the CLP domain and task
            synthesized_knowledge: The synthesized knowledge from the RAG system
            target_llm_info: Information about the target LLM
            num_questions: Number of questions to request
            difficulty_level: Desired difficulty level
            
        Returns:
            Generated prompt for question generation
        """
        logger.info(f"Generating question prompt for domain: {domain_info.get('clp_domain_id', 'Unknown')}")
        
        # Extract domain info
        domain_id = domain_info.get("clp_domain_id", "")
        domain_name = domain_info.get("clp_domain_name", "")
        task_id = domain_info.get("clp_task_id", "")
        task_statement = domain_info.get("clp_task_statement", "")
        
        # Create task description
        task_description = f"Generate {num_questions} multiple-choice questions for the Certified Licensing Professional (CLP) exam, "
        
        if domain_id and domain_name:
            task_description += f"specifically for Domain {domain_id}: {domain_name}, "
        if task_id and task_statement:
            task_description += f"Task {task_id}: {task_statement}, "
            
        task_description += f"with {difficulty_level} difficulty levels."
        
        # Create output format schema
        output_format_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "clp_domain_id": {"type": "string"},
                    "clp_domain_name": {"type": "string"},
                    "clp_task_id": {"type": "string", "description": "Optional"},
                    "clp_task_statement": {"type": "string", "description": "Optional"},
                    "question_stem": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "option_id": {"type": "string", "enum": ["A", "B", "C", "D"]},
                                "text": {"type": "string"}
                            }
                        }
                    },
                    "correct_option_id": {"type": "string", "enum": ["A", "B", "C", "D"]},
                    "explanation": {"type": "string"},
                    "difficulty_level_assessed": {"type": "string", "enum": ["Easy", "Medium", "Hard"]},
                    "cognitive_skill_targeted": {"type": "string"}
                },
                "required": ["question_id", "clp_domain_id", "question_stem", "options", "correct_option_id", "explanation"]
            }
        }
        
        # Generate the prompt using meta-prompting
        prompts = self.generate_llm_prompt(
            task_description=task_description,
            target_llm_info=target_llm_info,
            output_format_schema=output_format_schema
        )
        
        if not prompts:
            return self._create_fallback_prompt(task_description, target_llm_info, output_format_schema)
        
        # Get the first generated prompt
        base_prompt = prompts[0]
        
        # Inject synthesized knowledge
        knowledge_section = f"""
## Synthesized Knowledge
{synthesized_knowledge.summary}

### Key Concepts:
{self._format_key_concepts(synthesized_knowledge.key_concepts)}

### Potential Exam Areas:
{self._format_exam_areas(synthesized_knowledge.potential_exam_areas)}
"""
        
        # Insert knowledge section before the final instructions
        final_prompt = base_prompt + "\n\n" + knowledge_section + "\n\nGenerate questions based ONLY on the above synthesized knowledge."
        
        return final_prompt
    
    def generate_qa_prompt(
        self,
        mcq_info: Dict[str, Any],
        source_context: str,
        criterion: str,
        target_llm_info: Dict[str, str]
    ) -> str:
        """
        Generate a prompt for quality assurance evaluation.
        
        Args:
            mcq_info: Information about the MCQ to evaluate
            source_context: The source context used to generate the MCQ
            criterion: The specific QA criterion to evaluate
            target_llm_info: Information about the target LLM
            
        Returns:
            Generated prompt for QA evaluation
        """
        # Create task description
        task_description = f"Evaluate a multiple-choice question for the CLP exam against the criterion: {criterion}"
        
        # Create output format schema
        output_format_schema = {
            "type": "object",
            "properties": {
                "score": {"type": "string", "description": "Score or rating according to the criterion"},
                "justification": {"type": "string", "description": "Detailed justification for the score"}
            },
            "required": ["score", "justification"]
        }
        
        # Generate the prompt using meta-prompting
        prompts = self.generate_llm_prompt(
            task_description=task_description,
            target_llm_info=target_llm_info,
            output_format_schema=output_format_schema
        )
        
        if not prompts:
            return self._create_fallback_qa_prompt(mcq_info, source_context, criterion)
        
        # Get the first generated prompt
        base_prompt = prompts[0]
        
        # Inject MCQ information
        mcq_section = f"""
## Multiple-Choice Question to Evaluate
Question Stem: {mcq_info.get('question_stem', '')}

Options:
{self._format_options(mcq_info.get('options', []))}

Correct Answer: {self._get_correct_option_text(mcq_info)}

## Source Context
{source_context}

## Criterion to Evaluate
{criterion}
"""
        
        # Insert MCQ section before the final instructions
        final_prompt = base_prompt + "\n\n" + mcq_section
        
        return final_prompt
    
    def _create_meta_prompt(
        self,
        task_description: str,
        target_llm_info: Dict[str, str],
        output_format_schema: Dict[str, Any],
        num_prompts: int
    ) -> str:
        """
        Create a meta-prompt for prompt generation.
        
        Args:
            task_description: Description of the task
            target_llm_info: Information about the target LLM
            output_format_schema: Schema for the desired output format
            num_prompts: Number of prompts to generate
            
        Returns:
            Meta-prompt for prompt generation
        """
        schema_str = json.dumps(output_format_schema, indent=2)
        
        meta_prompt = f"""
You are an expert AI Prompt Engineer. Your task is to generate {num_prompts} highly effective prompt{"s" if num_prompts > 1 else ""} for another Large Language Model (the 'Target LLM').

Target LLM: {target_llm_info.get('name', 'Unknown')}
Target LLM Capabilities: {target_llm_info.get('capabilities', '')}

Task Description for Target LLM: {task_description}

Desired Output Format from Target LLM: A structured output following this JSON schema:
{schema_str}

Generate {num_prompts} diverse, high-quality prompt{"s" if num_prompts > 1 else ""} for the Target LLM. Each prompt should:

1. Be clear, specific, and unambiguous.
2. Assign an appropriate role or persona to the Target LLM (e.g., 'You are an expert CLP exam question author').
3. Provide explicit step-by-step instructions if the task is complex.
4. Specify all constraints (e.g., number of questions, difficulty, style).
5. Clearly request the output in the specified format.
6. Incorporate best practices for prompting the Target LLM family.

{"Please separate each prompt with '---' if generating multiple prompts." if num_prompts > 1 else ""}
"""
        return meta_prompt
    
    def _parse_generated_prompts(self, generated_text: str, num_prompts: int) -> List[str]:
        """
        Parse generated prompts from the meta-prompting response.
        
        Args:
            generated_text: Text generated by the meta-prompting model
            num_prompts: Expected number of prompts
            
        Returns:
            List of parsed prompts
        """
        if num_prompts == 1:
            return [generated_text.strip()]
        
        # For multiple prompts, split by the separator
        prompts = [p.strip() for p in generated_text.split("---") if p.strip()]
        
        # If we didn't get the expected number, but we got at least one
        if len(prompts) < num_prompts and len(prompts) > 0:
            logger.warning(f"Expected {num_prompts} prompts but got {len(prompts)}")
            return prompts
        
        # If we didn't get any valid prompts, treat the whole text as one prompt
        if not prompts:
            logger.warning("No prompt separators found, treating the entire response as one prompt")
            return [generated_text.strip()]
        
        return prompts
    
    def _create_fallback_prompt(
        self,
        task_description: str,
        target_llm_info: Dict[str, str],
        output_format_schema: Dict[str, Any]
    ) -> str:
        """
        Create a fallback prompt if meta-prompting fails.
        
        Args:
            task_description: Description of the task
            target_llm_info: Information about the target LLM
            output_format_schema: Schema for the desired output format
            
        Returns:
            Fallback prompt
        """
        schema_str = json.dumps(output_format_schema, indent=2)
        
        fallback_prompt = f"""
You are an expert in the Certified Licensing Professional (CLP) exam and professional licensing content.

Task: {task_description}

Output Format:
Please provide your response following this JSON schema:
{schema_str}

Instructions:
1. Create content that is accurate, relevant, and challenging.
2. Use clear, professional language.
3. Ensure all content is based on factual information about licensing practices.
4. Provide detailed explanations for correct answers.
"""
        return fallback_prompt
    
    def _create_fallback_qa_prompt(
        self,
        mcq_info: Dict[str, Any],
        source_context: str,
        criterion: str
    ) -> str:
        """
        Create a fallback prompt for QA evaluation if meta-prompting fails.
        
        Args:
            mcq_info: Information about the MCQ to evaluate
            source_context: The source context used to generate the MCQ
            criterion: The specific QA criterion to evaluate
            
        Returns:
            Fallback QA prompt
        """
        fallback_prompt = f"""
You are an expert Quality Assurance Specialist for the Certified Licensing Professional (CLP) exam.

Evaluate the following multiple-choice question against the criterion: {criterion}

Question Stem: {mcq_info.get('question_stem', '')}

Options:
{self._format_options(mcq_info.get('options', []))}

Correct Answer: {self._get_correct_option_text(mcq_info)}

Source Context:
{source_context}

Evaluate strictly based on the provided criterion. Provide your evaluation as a JSON object with the following structure:
{{
  "score": "<score or rating according to the criterion>",
  "justification": "<detailed justification for the score>"
}}
"""
        return fallback_prompt
    
    def _format_key_concepts(self, key_concepts: List[Dict[str, str]]) -> str:
        """Format key concepts for inclusion in a prompt."""
        if not key_concepts:
            return "No key concepts available."
        
        formatted = []
        for i, concept in enumerate(key_concepts):
            concept_str = f"{i+1}. {concept.get('concept', '')}"
            
            if explanation := concept.get('explanation'):
                concept_str += f"\n   {explanation}"
                
            if importance := concept.get('importance'):
                concept_str += f"\n   Importance: {importance}"
                
            formatted.append(concept_str)
            
        return "\n\n".join(formatted)
    
    def _format_exam_areas(self, exam_areas: List[str]) -> str:
        """Format potential exam areas for inclusion in a prompt."""
        if not exam_areas:
            return "No potential exam areas defined."
        
        return "\n".join(f"- {area}" for area in exam_areas)
    
    def _format_options(self, options: List[Dict[str, str]]) -> str:
        """Format MCQ options for inclusion in a prompt."""
        if not options:
            return "No options available."
        
        return "\n".join(f"{option.get('option_id', '?')}. {option.get('text', '')}" for option in options)
    
    def _get_correct_option_text(self, mcq_info: Dict[str, Any]) -> str:
        """Get the text of the correct option for inclusion in a prompt."""
        correct_id = mcq_info.get('correct_option_id')
        options = mcq_info.get('options', [])
        
        for option in options:
            if option.get('option_id') == correct_id:
                return f"{correct_id}. {option.get('text', '')}"
        
        return f"Option {correct_id} (text not available)"