"""
HTML Generator for APEGA.
Generates interactive HTML practice exams using Jinja2 templates.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import os
import json
from jinja2 import Environment, FileSystemLoader
import datetime

from src.models.data_models import GeneratedMCQ


class HTMLGenerator:
    """
    Generates interactive HTML practice exams using Jinja2 templates.
    Creates user-friendly, interactive exam interfaces.
    """
    
    def __init__(
        self,
        templates_dir: Optional[str] = None,
        static_dir: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the HTMLGenerator.
        
        Args:
            templates_dir: Directory containing Jinja2 templates
            static_dir: Directory containing static assets (CSS, JS)
            verbose: Whether to log detailed output
        """
        self.templates_dir = templates_dir or os.getenv("TEMPLATES_DIR", "templates")
        self.static_dir = static_dir or os.path.join(self.templates_dir, "static")
        self.verbose = verbose
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=True
        )
        
        # Add custom filters
        self.env.filters['json'] = lambda obj: json.dumps(obj)
    
    def generate_html(
        self,
        mcqs: List[GeneratedMCQ],
        output_filepath: str,
        exam_title: str = "CLP Practice Exam",
        additional_context: Dict[str, Any] = None
    ) -> str:
        """
        Generate an HTML practice exam from MCQs.
        
        Args:
            mcqs: List of validated MCQs
            output_filepath: Path to save the HTML file
            exam_title: Title of the exam
            additional_context: Additional context data for the template
            
        Returns:
            Path to the generated HTML file
        """
        logger.info(f"Generating HTML exam with {len(mcqs)} questions")
        
        # Prepare template context
        context = {
            'exam_title': exam_title,
            'questions': [self._prepare_question_for_template(mcq) for mcq in mcqs],
            'generation_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            'total_questions': len(mcqs),
            'domains': self._extract_domains(mcqs)
        }
        
        # Add additional context if provided
        if additional_context:
            context.update(additional_context)
        
        try:
            # Get exam template
            template = self.env.get_template('exam_template.html')
            
            # Render the template
            rendered_html = template.render(**context)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
            
            # Write to file
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(rendered_html)
            
            logger.info(f"HTML exam generated and saved to {output_filepath}")
            return output_filepath
            
        except Exception as e:
            logger.error(f"Error generating HTML exam: {str(e)}")
            # Create a simple fallback HTML if template fails
            return self._generate_fallback_html(mcqs, output_filepath, exam_title)
    
    def _prepare_question_for_template(self, mcq: GeneratedMCQ) -> Dict[str, Any]:
        """
        Prepare a question for the Jinja2 template.
        
        Args:
            mcq: GeneratedMCQ object
            
        Returns:
            Dictionary with formatted question data
        """
        # Convert options to a format suitable for the template
        options = []
        for option in mcq.options:
            options.append({
                'id': option.option_id,
                'text': option.text,
                'is_correct': option.option_id == mcq.correct_option_id
            })
        
        # Format the question
        return {
            'id': mcq.question_id,
            'stem': mcq.question_stem,
            'options': options,
            'correct_id': mcq.correct_option_id,
            'explanation': mcq.explanation or "No explanation provided.",
            'domain': mcq.clp_domain_name or mcq.clp_domain_id,
            'domain_id': mcq.clp_domain_id,
            'task': mcq.clp_task_statement,
            'task_id': mcq.clp_task_id,
            'difficulty': mcq.difficulty_level_assessed or "Medium",
            'cognitive_skill': mcq.cognitive_skill_targeted or "Knowledge"
        }
    
    def _extract_domains(self, mcqs: List[GeneratedMCQ]) -> List[Dict[str, Any]]:
        """
        Extract unique domains from MCQs for navigation.
        
        Args:
            mcqs: List of MCQs
            
        Returns:
            List of domain information dictionaries
        """
        domains = {}
        
        for mcq in mcqs:
            domain_id = mcq.clp_domain_id
            if domain_id and domain_id not in domains:
                domains[domain_id] = {
                    'id': domain_id,
                    'name': mcq.clp_domain_name or domain_id,
                    'count': 1
                }
            elif domain_id:
                domains[domain_id]['count'] += 1
        
        return list(domains.values())
    
    def _generate_fallback_html(
        self,
        mcqs: List[GeneratedMCQ],
        output_filepath: str,
        exam_title: str
    ) -> str:
        """
        Generate a simple fallback HTML if templates fail.
        
        Args:
            mcqs: List of MCQs
            output_filepath: Path to save the HTML file
            exam_title: Title of the exam
            
        Returns:
            Path to the generated HTML file
        """
        logger.warning("Using fallback HTML generation")
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{exam_title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .question {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
        .question-stem {{ font-weight: bold; margin-bottom: 10px; }}
        .options {{ list-style-type: none; padding-left: 0; }}
        .option {{ margin-bottom: 8px; padding: 8px; border: 1px solid #eee; border-radius: 4px; }}
        .option label {{ display: block; cursor: pointer; }}
        .explanation {{ display: none; margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-left: 3px solid #28a745; }}
        .show-explanation {{ margin-top: 10px; }}
        button {{ padding: 8px 12px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }}
        button:hover {{ background-color: #0056b3; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{exam_title}</h1>
            <p>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </div>
        
        <div class="exam-container">
"""
        
        # Add each question
        for i, mcq in enumerate(mcqs):
            question_html = f"""
            <div class="question" id="question-{i+1}">
                <div class="question-number">Question {i+1} of {len(mcqs)}</div>
                <div class="question-stem">{mcq.question_stem}</div>
                <div class="options">
"""
            
            # Add options
            for option in mcq.options:
                question_html += f"""
                    <div class="option">
                        <label>
                            <input type="radio" name="q{i+1}" value="{option.option_id}">
                            <span class="option-text">{option.option_id}. {option.text}</span>
                        </label>
                    </div>
"""
            
            # Add explanation and domain info
            domain_info = f"{mcq.clp_domain_name} ({mcq.clp_domain_id})" if mcq.clp_domain_name else mcq.clp_domain_id
            
            question_html += f"""
                </div>
                <button class="show-explanation" onclick="toggleExplanation({i+1})">Show Explanation</button>
                <div class="explanation" id="explanation-{i+1}">
                    <p><strong>Correct Answer:</strong> {mcq.correct_option_id}</p>
                    <p><strong>Explanation:</strong> {mcq.explanation or "No explanation provided."}</p>
                    <p><strong>Domain:</strong> {domain_info}</p>
                </div>
            </div>
"""
            
            html += question_html
        
        # Add JavaScript for interactivity
        html += """
        </div>
    </div>
    
    <script>
        function toggleExplanation(questionNum) {
            const explanationDiv = document.getElementById(`explanation-${questionNum}`);
            const button = document.querySelector(`#question-${questionNum} .show-explanation`);
            
            if (explanationDiv.style.display === 'block') {
                explanationDiv.style.display = 'none';
                button.textContent = 'Show Explanation';
            } else {
                explanationDiv.style.display = 'block';
                button.textContent = 'Hide Explanation';
            }
        }
    </script>
</body>
</html>
"""
        
        # Write to file
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(html)
            
            logger.info(f"Fallback HTML exam generated and saved to {output_filepath}")
            return output_filepath
        except Exception as e:
            logger.error(f"Error generating fallback HTML: {str(e)}")
            raise