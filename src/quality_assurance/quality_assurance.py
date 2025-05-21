"""
Quality Assurance for APEGA.
Evaluates generated MCQs using OpenAI's o4-mini model.
"""

from typing import List, Dict, Any, Optional, Union
from loguru import logger
import os
import json
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.models.data_models import GeneratedMCQ, QAResult, QACriterionEvaluation
from src.prompt_engineering.prompt_templates import PromptTemplates


class QACriterion:
    """Base class for quality assurance criteria."""
    
    name: str = "base_criterion"
    description: str = "Base criterion for QA evaluation"
    
    @classmethod
    def create_prompt(cls, mcq: GeneratedMCQ, source_context: str) -> str:
        """Create a prompt for evaluating this criterion."""
        raise NotImplementedError("Subclasses must implement this method")
    
    @classmethod
    def parse_result(cls, response: str) -> QACriterionEvaluation:
        """Parse the response for this criterion."""
        try:
            # Try to parse as JSON
            response_json = json.loads(response)
            score = str(response_json.get("score", "Fail"))
            justification = str(response_json.get("justification", "No justification provided"))
            
            return QACriterionEvaluation(
                score=score,
                justification=justification
            )
        except json.JSONDecodeError:
            # If not valid JSON, use a default result
            logger.warning(f"Failed to parse QA response as JSON: {response}")
            return QACriterionEvaluation(
                score="Fail",
                justification=f"Failed to parse response: {response[:100]}..."
            )


class FactualAccuracyCriterion(QACriterion):
    """Criterion for evaluating factual accuracy of the correct answer."""
    
    name: str = "factual_accuracy"
    description: str = "The designated correct answer must be verifiably true and accurate based only on the provided source context."
    
    @classmethod
    def create_prompt(cls, mcq: GeneratedMCQ, source_context: str) -> str:
        """Create a prompt for evaluating factual accuracy."""
        # Get the correct option text
        correct_text = ""
        for option in mcq.options:
            if option.option_id == mcq.correct_option_id:
                correct_text = option.text
                break
                
        # Format options
        options_text = "\n".join([f"{o.option_id}. {o.text}" for o in mcq.options])
        
        # Create prompt using template
        template = PromptTemplates.qa_evaluation_template()
        prompt = template.replace(
            "{criterion}", "factual_accuracy"
        ).replace(
            "{criterion_description}", PromptTemplates.get_criterion_description("factual_accuracy")
        ).replace(
            "{question_stem}", mcq.question_stem
        ).replace(
            "{options_text}", options_text
        ).replace(
            "{correct_answer}", f"{mcq.correct_option_id}. {correct_text}"
        ).replace(
            "{source_context}", source_context
        )
        
        return prompt


class ClarityCriterion(QACriterion):
    """Criterion for evaluating clarity and unambiguity of the question stem."""
    
    name: str = "clarity"
    description: str = "The question stem should be phrased clearly, concisely, and without ambiguity, leading to a single, best interpretation."
    
    @classmethod
    def create_prompt(cls, mcq: GeneratedMCQ, source_context: str) -> str:
        """Create a prompt for evaluating clarity."""
        # Format options
        options_text = "\n".join([f"{o.option_id}. {o.text}" for o in mcq.options])
        
        # Create prompt using template
        template = PromptTemplates.qa_evaluation_template()
        prompt = template.replace(
            "{criterion}", "clarity"
        ).replace(
            "{criterion_description}", PromptTemplates.get_criterion_description("clarity")
        ).replace(
            "{question_stem}", mcq.question_stem
        ).replace(
            "{options_text}", options_text
        ).replace(
            "{correct_answer}", f"{mcq.correct_option_id}"
        ).replace(
            "{source_context}", source_context
        )
        
        return prompt


class DistractorPlausibilityCriterion(QACriterion):
    """Criterion for evaluating plausibility of distractors."""
    
    name: str = "distractor_plausibility"
    description: str = "All incorrect options (distractors) should be believable and attractive to candidates with incomplete knowledge, but are definitively incorrect."
    
    @classmethod
    def create_prompt(cls, mcq: GeneratedMCQ, source_context: str) -> str:
        """Create a prompt for evaluating distractor plausibility."""
        # Get the correct option text
        correct_text = ""
        for option in mcq.options:
            if option.option_id == mcq.correct_option_id:
                correct_text = option.text
                break
                
        # Format options
        options_text = "\n".join([f"{o.option_id}. {o.text}" for o in mcq.options])
        
        # Create prompt using template
        template = PromptTemplates.qa_evaluation_template()
        prompt = template.replace(
            "{criterion}", "distractor_plausibility"
        ).replace(
            "{criterion_description}", PromptTemplates.get_criterion_description("distractor_plausibility")
        ).replace(
            "{question_stem}", mcq.question_stem
        ).replace(
            "{options_text}", options_text
        ).replace(
            "{correct_answer}", f"{mcq.correct_option_id}. {correct_text}"
        ).replace(
            "{source_context}", source_context
        )
        
        return prompt


class RelevanceCriterion(QACriterion):
    """Criterion for evaluating relevance to learning objectives."""
    
    name: str = "relevance"
    description: str = "The question should effectively assess knowledge or skills pertinent to the specified CLP Domain and Task Statement."
    
    @classmethod
    def create_prompt(cls, mcq: GeneratedMCQ, source_context: str) -> str:
        """Create a prompt for evaluating relevance."""
        # Format options
        options_text = "\n".join([f"{o.option_id}. {o.text}" for o in mcq.options])
        
        # Add domain and task info to source context
        domain_info = f"CLP Domain: {mcq.clp_domain_id} - {mcq.clp_domain_name}\n"
        if mcq.clp_task_id and mcq.clp_task_statement:
            domain_info += f"CLP Task: {mcq.clp_task_id} - {mcq.clp_task_statement}\n"
        
        extended_context = f"{domain_info}\n{source_context}"
        
        # Create prompt using template
        template = PromptTemplates.qa_evaluation_template()
        prompt = template.replace(
            "{criterion}", "relevance"
        ).replace(
            "{criterion_description}", PromptTemplates.get_criterion_description("relevance")
        ).replace(
            "{question_stem}", mcq.question_stem
        ).replace(
            "{options_text}", options_text
        ).replace(
            "{correct_answer}", f"{mcq.correct_option_id}"
        ).replace(
            "{source_context}", extended_context
        )
        
        return prompt


class NoCluesCriterion(QACriterion):
    """Criterion for evaluating absence of clues in the question stem or options."""
    
    name: str = "no_clues"
    description: str = "The question stem should not inadvertently provide clues to the correct answer, and options should be grammatically consistent."
    
    @classmethod
    def create_prompt(cls, mcq: GeneratedMCQ, source_context: str) -> str:
        """Create a prompt for evaluating the absence of clues."""
        # Format options
        options_text = "\n".join([f"{o.option_id}. {o.text}" for o in mcq.options])
        
        # Create prompt using template
        template = PromptTemplates.qa_evaluation_template()
        prompt = template.replace(
            "{criterion}", "no_clues"
        ).replace(
            "{criterion_description}", PromptTemplates.get_criterion_description("no_clues")
        ).replace(
            "{question_stem}", mcq.question_stem
        ).replace(
            "{options_text}", options_text
        ).replace(
            "{correct_answer}", f"{mcq.correct_option_id}"
        ).replace(
            "{source_context}", source_context
        )
        
        return prompt


class NoBiasCriterion(QACriterion):
    """Criterion for evaluating freedom from bias."""
    
    name: str = "no_bias"
    description: str = "The question should be free from cultural, gender, or other forms of bias, and avoid sensitive or offensive content."
    
    @classmethod
    def create_prompt(cls, mcq: GeneratedMCQ, source_context: str) -> str:
        """Create a prompt for evaluating freedom from bias."""
        # Format options
        options_text = "\n".join([f"{o.option_id}. {o.text}" for o in mcq.options])
        
        # Create prompt using template
        template = PromptTemplates.qa_evaluation_template()
        prompt = template.replace(
            "{criterion}", "no_bias"
        ).replace(
            "{criterion_description}", PromptTemplates.get_criterion_description("no_bias")
        ).replace(
            "{question_stem}", mcq.question_stem
        ).replace(
            "{options_text}", options_text
        ).replace(
            "{correct_answer}", f"{mcq.correct_option_id}"
        ).replace(
            "{source_context}", source_context
        )
        
        return prompt


class OverallQualityCriterion(QACriterion):
    """Criterion for evaluating overall question quality."""
    
    name: str = "overall_quality"
    description: str = "Holistic assessment of the question's suitability for a challenging professional certification practice exam."
    
    @classmethod
    def create_prompt(cls, mcq: GeneratedMCQ, source_context: str) -> str:
        """Create a prompt for evaluating overall quality."""
        # Get the correct option text
        correct_text = ""
        for option in mcq.options:
            if option.option_id == mcq.correct_option_id:
                correct_text = option.text
                break
                
        # Format options
        options_text = "\n".join([f"{o.option_id}. {o.text}" for o in mcq.options])
        
        # Create prompt using template
        template = PromptTemplates.qa_evaluation_template()
        prompt = template.replace(
            "{criterion}", "overall_quality"
        ).replace(
            "{criterion_description}", PromptTemplates.get_criterion_description("overall_quality")
        ).replace(
            "{question_stem}", mcq.question_stem
        ).replace(
            "{options_text}", options_text
        ).replace(
            "{correct_answer}", f"{mcq.correct_option_id}. {correct_text}"
        ).replace(
            "{source_context}", source_context
        )
        
        return prompt


class QualityAssurance:
    """
    Evaluates generated MCQs using OpenAI's o4-mini model.
    Checks against multiple criteria to ensure questions meet quality standards.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "o4-mini",
        temperature: float = 0.0,
        verbose: bool = False,
        criteria: Optional[List[QACriterion]] = None
    ):
        """
        Initialize the QualityAssurance evaluator.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model_name: Name of the OpenAI model to use
            temperature: Temperature for the model (lower for more consistent evaluation)
            verbose: Whether to log detailed output
            criteria: List of QA criteria to evaluate (defaults to all standard criteria)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it via parameter or OPENAI_API_KEY environment variable.")
            
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Set up evaluation criteria
        self.criteria = criteria or [
            FactualAccuracyCriterion,
            ClarityCriterion,
            DistractorPlausibilityCriterion,
            RelevanceCriterion,
            NoCluesCriterion,
            NoBiasCriterion,
            OverallQualityCriterion
        ]
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True
    )
    def evaluate_mcq(
        self,
        mcq: GeneratedMCQ,
        source_context: str
    ) -> QAResult:
        """
        Evaluate a multiple-choice question against quality criteria.
        
        Args:
            mcq: The MCQ to evaluate
            source_context: The source context used to generate the MCQ
            
        Returns:
            QAResult object with evaluation results
        """
        logger.info(f"Evaluating MCQ {mcq.question_id}")
        
        # Evaluate against each criterion
        criteria_feedback = {}
        
        for criterion_class in self.criteria:
            criterion_name = criterion_class.name
            
            try:
                # Create prompt for this criterion
                prompt = criterion_class.create_prompt(mcq, source_context)
                
                if self.verbose:
                    logger.debug(f"Prompt for {criterion_name}: {prompt}")
                
                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert Quality Assurance Specialist for the Certified Licensing Professional (CLP) exam."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    max_tokens=500
                )
                
                # Parse response
                response_text = response.choices[0].message.content
                
                if self.verbose:
                    logger.debug(f"Response for {criterion_name}: {response_text}")
                
                # Parse the criterion-specific result
                evaluation = criterion_class.parse_result(response_text)
                criteria_feedback[criterion_name] = evaluation
                
                logger.info(f"Criterion {criterion_name}: {evaluation.score}")
                
            except Exception as e:
                logger.error(f"Error evaluating criterion {criterion_name}: {str(e)}")
                # Add a default failure evaluation
                criteria_feedback[criterion_name] = QACriterionEvaluation(
                    score="Fail",
                    justification=f"Evaluation error: {str(e)}"
                )
        
        # Determine overall pass/fail
        overall_pass = self._determine_overall_result(criteria_feedback)
        
        # Generate revision suggestions if needed
        revision_suggestions = None
        if not overall_pass:
            revision_suggestions = self._generate_revision_suggestions(mcq, criteria_feedback)
        
        return QAResult(
            mcq_id=mcq.question_id,
            overall_pass=overall_pass,
            criteria_feedback=criteria_feedback,
            revision_suggestions=revision_suggestions
        )
    
    def _determine_overall_result(self, criteria_feedback: Dict[str, QACriterionEvaluation]) -> bool:
        """
        Determine overall pass/fail based on criteria evaluations.
        
        Args:
            criteria_feedback: Dictionary of criterion evaluations
            
        Returns:
            Boolean indicating overall pass (True) or fail (False)
        """
        # Critical criteria that must pass
        critical_criteria = ["factual_accuracy", "no_bias"]
        
        # Check critical criteria
        for criterion in critical_criteria:
            if criterion in criteria_feedback:
                evaluation = criteria_feedback[criterion]
                # If the score contains "fail" (case-insensitive)
                if "fail" in evaluation.score.lower():
                    return False
        
        # Count other failures
        failure_count = 0
        for criterion, evaluation in criteria_feedback.items():
            if criterion not in critical_criteria and "fail" in evaluation.score.lower():
                failure_count += 1
        
        # Overall quality criterion
        overall_quality_score = 0
        if "overall_quality" in criteria_feedback:
            overall_score = criteria_feedback["overall_quality"].score
            # Try to extract a numeric score
            try:
                if "/" in overall_score:
                    # Format: "X/Y"
                    parts = overall_score.split("/")
                    overall_quality_score = float(parts[0]) / float(parts[1])
                else:
                    # Try to parse as a direct number
                    overall_quality_score = float(overall_score)
            except ValueError:
                # If not a number, check for keywords
                if "high" in overall_score.lower() or "good" in overall_score.lower() or "excellent" in overall_score.lower():
                    overall_quality_score = 0.8
                elif "medium" in overall_score.lower() or "average" in overall_score.lower() or "moderate" in overall_score.lower():
                    overall_quality_score = 0.5
                else:
                    overall_quality_score = 0.3
        
        # Pass if fewer than 2 non-critical failures AND overall quality is acceptable
        return failure_count < 2 and overall_quality_score >= 0.5
    
    def _generate_revision_suggestions(
        self,
        mcq: GeneratedMCQ,
        criteria_feedback: Dict[str, QACriterionEvaluation]
    ) -> str:
        """
        Generate suggestions for revising a failing MCQ.
        
        Args:
            mcq: The MCQ that failed evaluation
            criteria_feedback: Dictionary of criterion evaluations
            
        Returns:
            String with revision suggestions
        """
        suggestions = []
        
        # Add suggestions for each failing criterion
        for criterion, evaluation in criteria_feedback.items():
            if "fail" in evaluation.score.lower() or (criterion == "overall_quality" and float(evaluation.score) < 0.5):
                suggestions.append(f"{criterion.replace('_', ' ').title()}: {evaluation.justification}")
        
        # Combine all suggestions
        return "\n\n".join(suggestions)