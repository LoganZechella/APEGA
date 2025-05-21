"""
Prompt Templates for APEGA.
Contains predefined prompt templates for various tasks.
"""

from typing import Dict, Any, List


class PromptTemplates:
    """
    Static class containing predefined prompt templates for various APEGA tasks.
    These serve as fallbacks or starting points for the PromptEngineer.
    """
    
    @staticmethod
    def question_generation_template() -> str:
        """Template for question generation."""
        return """
You are an expert Certified Licensing Professional (CLP) exam question writer and subject matter expert in intellectual property commercialization.

## Task
Generate {num_questions} multiple-choice questions for the CLP exam {domain_specific}. Each question should have one correct answer and three plausible, challenging distractors.

## Guidelines
1. Create questions that test understanding, application, and analysis of concepts, not just recall.
2. Ensure questions are clear, unambiguous, and use professional language.
3. Make distractors plausible but unambiguously incorrect to a knowledgeable professional.
4. Include an explanation for why the correct answer is correct and why distractors are incorrect.
5. Questions should reflect the complexity and nuance expected in a professional certification exam.

## Format
Provide your response in the following JSON format:
```json
[
  {
    "question_id": "CLP_{domain_id}_Q1",
    "clp_domain_id": "{domain_id}",
    "clp_domain_name": "{domain_name}",
    "clp_task_id": "{task_id}",
    "clp_task_statement": "{task_statement}",
    "question_stem": "Question text goes here?",
    "options": [
      {"option_id": "A", "text": "Option A text"},
      {"option_id": "B", "text": "Option B text"},
      {"option_id": "C", "text": "Option C text"},
      {"option_id": "D", "text": "Option D text"}
    ],
    "correct_option_id": "B",
    "explanation": "Explanation of why B is correct and why other options are incorrect.",
    "difficulty_level_assessed": "Medium",
    "cognitive_skill_targeted": "Application"
  }
]
```

## Knowledge Context
{knowledge_context}

Generate questions based ONLY on the provided knowledge context.
"""
    
    @staticmethod
    def qa_evaluation_template() -> str:
        """Template for quality assurance evaluation."""
        return """
You are an AI Exam Quality Assurance Specialist for the Certified Licensing Professional (CLP) exam.

## Quality Assurance Task
Evaluate the provided multiple-choice question against the criterion: {criterion}

## Multiple-Choice Question to Evaluate
Question Stem: {question_stem}

Options:
{options_text}

Correct Answer: {correct_answer}

## Source Context
{source_context}

## Evaluation Criterion: {criterion}
{criterion_description}

## Evaluation Guidelines
1. Focus solely on the specified criterion
2. Base your evaluation only on the provided source context
3. Be objective and thorough in your assessment
4. Provide specific examples from the question to support your evaluation

## Output Format
Provide your evaluation as a JSON object with the following structure:
```json
{
  "score": "Pass/Fail or numeric score as appropriate for this criterion",
  "justification": "Detailed explanation of your evaluation with specific examples"
}
```
"""
    
    @staticmethod
    def deep_analysis_template() -> str:
        """Template for deep content analysis."""
        return """
Task: Comprehensive Analysis and Synthesis of CLP Source Material for Exam Preparation

Role: You are an expert curriculum developer and subject matter expert for the Certified Licensing Professional (CLP) exam. Your task is to thoroughly analyze the provided source materials.

Context: The following text segments have been retrieved as relevant to {domain_specific}.

{contexts}

Instructions:
1. Thoroughly review and analyze the provided text segments.
2. Identify and extract:
   - Core principles, definitions, and legal frameworks
   - Key factors to consider (e.g., legal, commercial, resource-based)
   - Procedural steps or decision-making processes
   - Relationships and distinctions between different concepts
   - Potential areas of ambiguity or complexity that might be tested
3. Synthesize this analysis into a structured knowledge representation.
4. Highlight areas that appear to be of high importance or complexity, suggesting they are likely candidates for examination questions.

Output Format: Provide the synthesized knowledge in JSON format with the following structure:
```json
{
  "summary": "A comprehensive summary of the core knowledge, 1-2 paragraphs",
  "key_concepts": [
    {
      "concept": "Name or title of the concept",
      "explanation": "Clear, concise explanation",
      "importance": "Why this is important for the CLP exam"
    }
  ],
  "potential_exam_areas": [
    "Topic or scenario that would make for a good exam question"
  ]
}
```
"""
    
    @staticmethod
    def criteria_descriptions() -> Dict[str, str]:
        """Descriptions for QA evaluation criteria."""
        return {
            "factual_accuracy": "The designated correct answer must be verifiably true and accurate based only on the provided source context.",
            
            "clarity": "The question stem should be phrased clearly, concisely, and without ambiguity, leading to a single, best interpretation.",
            
            "distractor_plausibility": "All incorrect options (distractors) should be believable and attractive to candidates with incomplete knowledge, but are definitively incorrect.",
            
            "relevance": "The question should effectively assess knowledge or skills pertinent to the specified CLP Domain and Task Statement.",
            
            "no_clues": "The question stem should not inadvertently provide clues to the correct answer, and options should be grammatically consistent.",
            
            "no_bias": "The question should be free from cultural, gender, or other forms of bias, and avoid sensitive or offensive content.",
            
            "overall_quality": "Holistic assessment of the question's suitability for a challenging professional certification practice exam."
        }
    
    @staticmethod
    def get_criterion_description(criterion: str) -> str:
        """Get the description for a specific criterion."""
        descriptions = PromptTemplates.criteria_descriptions()
        return descriptions.get(criterion, "No description available for this criterion.")
