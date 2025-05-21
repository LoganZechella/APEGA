"""
PDF Generator for APEGA.
Converts HTML exams to PDF format using WeasyPrint.
"""

from typing import Optional
from loguru import logger
import os
from weasyprint import HTML, CSS
import tempfile

class PDFGenerator:
    """
    Converts HTML exams to PDF format using WeasyPrint.
    Provides high-quality, print-friendly PDF versions of practice exams.
    """
    
    def __init__(
        self,
        css_filepath: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the PDFGenerator.
        
        Args:
            css_filepath: Path to custom CSS file for PDF styling
            verbose: Whether to log detailed output
        """
        self.css_filepath = css_filepath
        self.verbose = verbose
    
    def generate_pdf(
        self,
        html_filepath: str,
        output_filepath: str,
        page_size: str = 'A4',
        margin: str = '1in'
    ) -> str:
        """
        Generate a PDF from an HTML file.
        
        Args:
            html_filepath: Path to the HTML file
            output_filepath: Path to save the PDF file
            page_size: Page size (e.g., 'A4', 'Letter')
            margin: Page margin (e.g., '1in', '2.54cm')
            
        Returns:
            Path to the generated PDF file
        """
        logger.info(f"Generating PDF from {html_filepath}")
        
        try:
            # Load HTML file
            html = HTML(filename=html_filepath)
            
            # CSS for print formatting
            css = []
            
            # Add custom CSS if provided
            if self.css_filepath and os.path.exists(self.css_filepath):
                css.append(CSS(filename=self.css_filepath))
            
            # Add default print CSS
            default_css = self._get_default_css(page_size, margin)
            css.append(CSS(string=default_css))
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
            
            # Render PDF
            html.write_pdf(output_filepath, stylesheets=css)
            
            logger.info(f"PDF generated and saved to {output_filepath}")
            return output_filepath
            
        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}")
            # Try alternative approach if initial method fails
            return self._generate_pdf_alternative(html_filepath, output_filepath, page_size, margin)
    
    def html_to_pdf(
        self,
        html_content: str,
        output_filepath: str,
        page_size: str = 'A4',
        margin: str = '1in'
    ) -> str:
        """
        Generate a PDF directly from HTML content.
        
        Args:
            html_content: HTML content as string
            output_filepath: Path to save the PDF file
            page_size: Page size (e.g., 'A4', 'Letter')
            margin: Page margin (e.g., '1in', '2.54cm')
            
        Returns:
            Path to the generated PDF file
        """
        logger.info("Generating PDF from HTML content")
        
        try:
            # Create a temporary file for the HTML content
            fd, temp_html_path = tempfile.mkstemp(suffix='.html')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                # Generate PDF from the temporary HTML file
                return self.generate_pdf(temp_html_path, output_filepath, page_size, margin)
                
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_html_path):
                    os.remove(temp_html_path)
                    
        except Exception as e:
            logger.error(f"Error generating PDF from HTML content: {str(e)}")
            raise
    
    def _generate_pdf_alternative(
        self,
        html_filepath: str,
        output_filepath: str,
        page_size: str,
        margin: str
    ) -> str:
        """
        Alternative approach to generate PDF if the standard method fails.
        
        Args:
            html_filepath: Path to the HTML file
            output_filepath: Path to save the PDF file
            page_size: Page size
            margin: Page margin
            
        Returns:
            Path to the generated PDF file
        """
        logger.warning("Using alternative PDF generation method")
        
        try:
            # Read HTML content
            with open(html_filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Create HTML object from string
            html = HTML(string=html_content, base_url=os.path.dirname(html_filepath))
            
            # CSS for print formatting
            css = []
            
            # Add custom CSS if provided
            if self.css_filepath and os.path.exists(self.css_filepath):
                css.append(CSS(filename=self.css_filepath))
            
            # Add default print CSS
            default_css = self._get_default_css(page_size, margin)
            css.append(CSS(string=default_css))
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
            
            # Render PDF
            html.write_pdf(output_filepath, stylesheets=css)
            
            logger.info(f"PDF generated using alternative method and saved to {output_filepath}")
            return output_filepath
            
        except Exception as e:
            logger.error(f"Alternative PDF generation also failed: {str(e)}")
            raise
    
    def _get_default_css(self, page_size: str, margin: str) -> str:
        """
        Get default CSS for PDF formatting.
        
        Args:
            page_size: Page size
            margin: Page margin
            
        Returns:
            CSS string
        """
        return f"""
            @page {{
                size: {page_size};
                margin: {margin};
                @bottom-right {{
                    content: "Page " counter(page) " of " counter(pages);
                    font-size: 10pt;
                }}
            }}
            
            body {{
                font-family: "Helvetica", "Arial", sans-serif;
                line-height: 1.5;
                font-size: 12pt;
                color: #333;
            }}
            
            h1, h2, h3, h4, h5, h6 {{
                margin-top: 1em;
                margin-bottom: 0.5em;
                page-break-after: avoid;
            }}
            
            .question {{
                page-break-inside: avoid;
                margin-bottom: 1.5em;
                border: 1px solid #ddd;
                padding: 1em;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            
            .question-stem {{
                font-weight: bold;
                margin-bottom: 0.5em;
            }}
            
            .options {{
                margin-left: 1.5em;
            }}
            
            .option {{
                margin-bottom: 0.5em;
            }}
            
            .correct-answer {{
                font-weight: bold;
                color: #28a745;
            }}
            
            .explanation {{
                margin-top: 1em;
                padding: 0.5em;
                background-color: #f8f9fa;
                border-left: 3px solid #28a745;
            }}
            
            /* Classes for PDF-only display */
            .pdf-only {{
                display: none;
            }}
            
            @media print {{
                .pdf-only {{
                    display: block;
                }}
                
                .web-only {{
                    display: none;
                }}
                
                a {{
                    text-decoration: none;
                    color: #333;
                }}
                
                button, input, .show-explanation {{
                    display: none;
                }}
                
                .explanation {{
                    display: block !important;
                }}
            }}
        """