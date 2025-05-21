"""
APEGA: Automated Practice Exam Generation Agent
Main entry point for the application.
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from loguru import logger

# Ensure src is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main controller
from src.orchestration.controller import APEGAController

def setup_logging(log_level):
    """Configure logging"""
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level=log_level)
    logger.add("logs/apega_{time}.log", rotation="10 MB", level=log_level)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="APEGA: Automated Practice Exam Generation Agent")
    parser.add_argument("--config", default="config/config.env", help="Path to configuration file")
    parser.add_argument("--query", default=None, help="Natural language query to generate the exam")
    parser.add_argument("--questions", type=int, default=10, help="Number of questions to generate")
    parser.add_argument("--output-dir", default=None, help="Output directory for generated exams")
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Load configuration
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Load the configuration file
    load_dotenv(args.config)
    
    # Load .env file for sensitive credentials if it exists
    if os.path.exists(".env"):
        load_dotenv(".env", override=True)
    
    # Setup logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    setup_logging(log_level)
    
    # Override config with command line arguments if provided
    output_dir = args.output_dir or os.getenv("OUTPUT_DIR")
    query = args.query or "Generate a comprehensive CLP practice exam covering all domains"
    
    logger.info(f"Starting APEGA with query: {query}")
    logger.info(f"Targeting {args.questions} questions")
    
    # Initialize and run the controller
    try:
        controller = APEGAController()
        result = controller.create_practice_exam(
            natural_language_query=query,
            num_questions=args.questions,
            output_dir=output_dir
        )
        
        if result.get("status") == "success":
            logger.info(f"Successfully generated {result.get('num_questions')} questions")
            logger.info(f"HTML exam available at: {result.get('html_path')}")
            logger.info(f"PDF exam available at: {result.get('pdf_path')}")
        else:
            logger.error(f"Exam generation failed: {result.get('message')}")
            
    except Exception as e:
        logger.exception(f"Error during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()