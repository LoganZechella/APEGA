"""
Output Formatter for APEGA.
Formats validated MCQs into structured output formats.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import os
import json
import datetime

from src.models.data_models import GeneratedMCQ
from src.output_formatting.html_generator import HTMLGenerator
from src.output_formatting.pdf_generator import PDFGenerator


class OutputFormatter:
    """
    Formats validated MCQs into various output formats.
    Manages the overall output process for HTML, PDF, and structured data formats.
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        templates_dir: Optional[str] = None,
        css_filepath: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the OutputFormatter.
        
        Args:
            output_dir: Directory to save output files
            templates_dir: Directory containing Jinja2 templates
            css_filepath: Path to custom CSS file for PDF styling
            verbose: Whether to log detailed output
        """
        self.output_dir = output_dir or os.getenv("OUTPUT_DIR", "output")
        self.templates_dir = templates_dir or os.getenv("TEMPLATES_DIR", "templates")
        self.css_filepath = css_filepath
        self.verbose = verbose
        
        # Create required directories
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize HTML and PDF generators
        self.html_generator = HTMLGenerator(templates_dir=self.templates_dir, verbose=self.verbose)
        self.pdf_generator = PDFGenerator(css_filepath=self.css_filepath, verbose=self.verbose)
    
    def generate_outputs(
        self,
        mcqs: List[GeneratedMCQ],
        output_name: Optional[str] = None,
        exam_title: str = "CLP Practice Exam",
        formats: List[str] = ['json', 'html', 'pdf']
    ) -> Dict[str, str]:
        """
        Generate output files in specified formats.
        
        Args:
            mcqs: List of validated MCQs
            output_name: Base name for output files (without extension)
            exam_title: Title of the exam
            formats: List of output formats ('json', 'html', 'pdf')
            
        Returns:
            Dictionary mapping format to output filepath
        """
        logger.info(f"Generating outputs in formats: {formats}")
        
        # Generate timestamp-based output name if not provided
        if not output_name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"clp_practice_exam_{timestamp}"
        
        # Initialize results dictionary
        results = {}
        
        # Generate each requested format
        for fmt in formats:
            try:
                if fmt.lower() == 'json':
                    json_path = self._generate_json(mcqs, output_name)
                    results['json'] = json_path
                
                elif fmt.lower() == 'html':
                    html_path = self._generate_html(mcqs, output_name, exam_title)
                    results['html'] = html_path
                
                elif fmt.lower() == 'pdf':
                    # Check if HTML was already generated
                    html_path = results.get('html')
                    
                    # If HTML not already generated, generate it first
                    if not html_path:
                        html_path = self._generate_html(mcqs, output_name, exam_title)
                        results['html'] = html_path
                    
                    # Generate PDF from HTML
                    pdf_path = self._generate_pdf(html_path, output_name)
                    results['pdf'] = pdf_path
                    
                else:
                    logger.warning(f"Unsupported output format: {fmt}")
            
            except Exception as e:
                logger.error(f"Error generating {fmt} output: {str(e)}")
        
        logger.info(f"Output generation complete: {list(results.keys())}")
        return results
    
    def _generate_json(self, mcqs: List[GeneratedMCQ], output_name: str) -> str:
        """
        Generate JSON output file.
        
        Args:
            mcqs: List of MCQs
            output_name: Base name for output file
            
        Returns:
            Path to the generated JSON file
        """
        json_path = os.path.join(self.output_dir, f"{output_name}.json")
        
        try:
            # Convert MCQs to JSON-serializable format
            mcq_dicts = []
            for mcq in mcqs:
                # Convert to dict and process any nested Pydantic models
                mcq_dict = mcq.model_dump()
                mcq_dicts.append(mcq_dict)
            
            # Add metadata
            output_data = {
                'metadata': {
                    'exam_title': f"CLP Practice Exam ({len(mcqs)} questions)",
                    'generation_timestamp': datetime.datetime.now().isoformat(),
                    'question_count': len(mcqs)
                },
                'questions': mcq_dicts
            }
            
            # Write to file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"JSON output generated and saved to {json_path}")
            return json_path
            
        except Exception as e:
            logger.error(f"Error generating JSON output: {str(e)}")
            raise
    
    def _generate_html(
        self,
        mcqs: List[GeneratedMCQ],
        output_name: str,
        exam_title: str
    ) -> str:
        """
        Generate HTML output file.
        
        Args:
            mcqs: List of MCQs
            output_name: Base name for output file
            exam_title: Title of the exam
            
        Returns:
            Path to the generated HTML file
        """
        html_path = os.path.join(self.output_dir, f"{output_name}.html")
        
        try:
            # Generate HTML
            html_path = self.html_generator.generate_html(
                mcqs=mcqs,
                output_filepath=html_path,
                exam_title=exam_title
            )
            
            logger.info(f"HTML output generated and saved to {html_path}")
            return html_path
            
        except Exception as e:
            logger.error(f"Error generating HTML output: {str(e)}")
            raise
    
    def _generate_pdf(self, html_path: str, output_name: str) -> str:
        """
        Generate PDF output file from HTML.
        
        Args:
            html_path: Path to HTML file
            output_name: Base name for output file
            
        Returns:
            Path to the generated PDF file
        """
        pdf_path = os.path.join(self.output_dir, f"{output_name}.pdf")
        
        try:
            # Generate PDF
            pdf_path = self.pdf_generator.generate_pdf(
                html_filepath=html_path,
                output_filepath=pdf_path
            )
            
            logger.info(f"PDF output generated and saved to {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error generating PDF output: {str(e)}")
            raise