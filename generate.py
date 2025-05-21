#!/usr/bin/env python
"""
Exam Generator for APEGA.
Script for generating practice exams using the APEGA system.
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from loguru import logger

# Ensure src is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestration.controller import APEGAController

def setup_logging(log_level):
    """Configure logging"""
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level=log_level)
    logger.add("logs/generate_{time}.log", rotation="10 MB", level=log_level)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="APEGA Exam Generation Tool")
    parser.add_argument("--config", default="config/config.env", help="Path to configuration file")
    parser.add_argument("--query", required=True, help="Natural language query describing the exam to generate")
    parser.add_argument("--questions", type=int, default=10, help="Number of questions to generate")
    parser.add_argument("--output-dir", help="Output directory for generated exams")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Load the configuration file
    load_dotenv(args.config)
    
    # Load .env file for sensitive credentials if it exists
    if os.path.exists(".env"):
        load_dotenv(".env", override=True)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else os.getenv("LOG_LEVEL", "INFO")
    setup_logging(log_level)
    
    # Initialize APEGA controller
    try:
        logger.info(f"Initializing APEGA controller")
        controller = APEGAController(
            config_path=args.config,
            verbose=args.verbose
        )
        
        # Generate practice exam
        logger.info(f"Generating practice exam with query: {args.query}")
        result = controller.create_practice_exam(
            natural_language_query=args.query,
            num_questions=args.questions,
            output_dir=args.output_dir
        )
        
        # Log results
        if result.get("status") == "success":
            logger.info(f"Successfully generated practice exam with {result.get('num_questions', 0)} questions")
            logger.info(f"HTML exam available at: {result.get('html_path')}")
            logger.info(f"PDF exam available at: {result.get('pdf_path')}")
            
            # Print results to console for user convenience
            print("\nExam Generation Successful!")
            print(f"Generated {result.get('num_questions', 0)} questions")
            print(f"HTML exam: {result.get('html_path')}")
            print(f"PDF exam: {result.get('pdf_path')}")
        else:
            logger.error(f"Exam generation failed: {result.get('message')}")
            print(f"\nExam Generation Failed: {result.get('message')}")
            sys.exit(1)
            
    except Exception as e:
        logger.exception(f"Error during exam generation: {e}")
        print(f"\nError during exam generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()