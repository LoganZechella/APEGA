"""
Controller for APEGA.
Main controller class that orchestrates the entire APEGA workflow.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import os
import datetime
import time
import uuid
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnablePassthrough
import json

from src.models.data_models import ExamGenerationJob, GeneratedMCQ, SynthesizedKnowledge, RetrievedContext
from src.knowledge_ingestion import KnowledgeIngestion
from src.knowledge_ingestion.embedding_generator import EmbeddingGenerator
from src.knowledge_ingestion.vector_db_manager import VectorDBManager
from src.rag_engine import RAGEngine
from src.prompt_engineering.prompt_engineer import PromptEngineer
from src.question_generation.question_generator import QuestionGenerator
from src.question_generation.distractor_generator import DistractorGenerator
from src.quality_assurance.quality_assurance import QualityAssurance
from src.output_formatting.output_formatter import OutputFormatter


class APEGAController:
    """
    Main controller for the APEGA system.
    Orchestrates the entire workflow from query to generated exam.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the APEGA controller.
        
        Args:
            config_path: Path to configuration file
            openai_api_key: OpenAI API key
            google_api_key: Google API key
            verbose: Whether to log detailed output
        """
        logger.info("Initializing APEGA Controller")
        
        # Load configuration
        self.config = self._load_configuration(config_path)
        
        # API keys (prioritize direct parameters, then fallback to environment variables)
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.warning("OpenAI API key not found. Some components will not function properly.")
            
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            logger.warning("Google API key not found. Some components will not function properly.")
        
        self.verbose = verbose
        
        # Set up embedding generator (shared by multiple components)
        self.embedding_generator = EmbeddingGenerator(
            api_key=self.openai_api_key,
            model_name=self.config.get("EMBEDDING_MODEL", "text-embedding-3-large"),
            dimensions=self.config.get("VECTOR_DIMENSIONS", 3072)
        )
        
        # Set up vector database manager (shared by multiple components)
        self.vector_db_manager = VectorDBManager(
            url=self.config.get("QDRANT_URL", "http://localhost:6333"),
            api_key=self.config.get("QDRANT_API_KEY"),
            collection_name=self.config.get("QDRANT_COLLECTION_NAME", "clp_knowledge"),
            vector_dimensions=self.config.get("VECTOR_DIMENSIONS", 3072)
        )
        
        # Initialize component registry
        self.components = {}
        self._initialize_components()
        
        # Set up workflow graph
        self.workflow_graph = self._create_workflow_graph()
    
    def _load_configuration(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file or environment.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        # Default configuration path
        if not config_path:
            config_path = os.getenv("CONFIG_PATH", "config/config.env")
        
        # Try to load from file if it exists
        if os.path.exists(config_path):
            logger.info(f"Loading configuration from {config_path}")
            try:
                with open(config_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if not line or line.startswith('#'):
                            continue
                        # Parse key-value pairs
                        if '=' in line:
                            key, value = line.split('=', 1)
                            config[key.strip()] = value.strip()
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        else:
            logger.warning(f"Configuration file {config_path} not found. Using environment variables.")
        
        # Load environment variables as fallback or override
        for key, value in os.environ.items():
            if key not in config:
                config[key] = value
        
        # Set default values for required configuration
        defaults = {
            "SOURCE_DOCUMENTS_DIR": "Context",
            "OUTPUT_DIR": "output",
            "TEMPLATES_DIR": "templates",
            "LOG_LEVEL": "INFO",
            "CHUNK_SIZE_TOKENS": "1024",
            "CHUNK_OVERLAP_TOKENS": "200",
            "VECTOR_DIMENSIONS": "3072",
            "QDRANT_COLLECTION_NAME": "clp_knowledge",
            "TOP_K_DENSE": "10",
            "TOP_K_SPARSE": "10",
            "TOP_K_RERANK": "5",
            "MAX_API_RETRIES": "5",
            "MAX_QA_RETRIES": "2"
        }
        
        # Apply defaults for missing values
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def _initialize_components(self) -> None:
        """Initialize all required components."""
        try:
            # Knowledge Ingestion
            self.components["knowledge_ingestion"] = KnowledgeIngestion(
                source_paths=[self.config.get("SOURCE_DOCUMENTS_DIR", "Context")],
                openai_api_key=self.openai_api_key,
                qdrant_url=self.config.get("QDRANT_URL"),
                qdrant_api_key=self.config.get("QDRANT_API_KEY"),
                collection_name=self.config.get("QDRANT_COLLECTION_NAME", "clp_knowledge"),
                chunking_strategy=self.config.get("CHUNKING_STRATEGY", "hybrid_hierarchical_semantic"),
                max_chunk_size_tokens=int(self.config.get("CHUNK_SIZE_TOKENS", 1024)),
                chunk_overlap_tokens=int(self.config.get("CHUNK_OVERLAP_TOKENS", 200)),
                vector_dimensions=int(self.config.get("VECTOR_DIMENSIONS", 3072)),
                use_ocr=False
            )
            
            # RAG Engine
            self.components["rag_engine"] = RAGEngine(
                vector_db=self.vector_db_manager,
                embedding_generator=self.embedding_generator,
                openai_api_key=self.openai_api_key,
                google_api_key=self.google_api_key,
                top_k_dense=int(self.config.get("TOP_K_DENSE", 10)),
                top_k_sparse=int(self.config.get("TOP_K_SPARSE", 10)),
                top_k_rerank=int(self.config.get("TOP_K_RERANK", 5))
            )
            
            # Prompt Engineering
            self.components["prompt_engineer"] = PromptEngineer(
                api_key=self.openai_api_key,
                model_name=self.config.get("PROMPT_ENGINEERING_MODEL", "o4-mini"),
                verbose=self.verbose
            )
            
            # Question Generation
            self.components["question_generator"] = QuestionGenerator(
                api_key=self.google_api_key,
                model_name=self.config.get("GENERATION_MODEL", "gemini-2.5-pro-preview-05-06"),
                verbose=self.verbose
            )
            
            # Distractor Generation
            self.components["distractor_generator"] = DistractorGenerator(
                api_key=self.google_api_key,
                model_name=self.config.get("GENERATION_MODEL", "gemini-2.5-pro-preview-05-06"),
                verbose=self.verbose
            )
            
            # Quality Assurance
            self.components["quality_assurance"] = QualityAssurance(
                api_key=self.openai_api_key,
                model_name=self.config.get("QA_MODEL", "o4-mini"),
                verbose=self.verbose
            )
            
            # Output Formatting
            self.components["output_formatter"] = OutputFormatter(
                output_dir=self.config.get("OUTPUT_DIR", "output"),
                templates_dir=self.config.get("TEMPLATES_DIR", "templates"),
                verbose=self.verbose
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def _create_workflow_graph(self) -> StateGraph:
        """
        Create the workflow graph using LangGraph.
        
        Returns:
            StateGraph instance
        """
        # Define the workflow graph
        workflow = StateGraph(ExamGenerationJob)
        
        # Process documents node
        workflow.add_node("process_documents", self._process_documents)
        
        # Parse query node
        workflow.add_node("parse_query", self._parse_query)
        
        # Generate questions node for each domain
        workflow.add_node("generate_questions_for_domain", self._generate_questions_for_domain)
        
        # Format output node
        workflow.add_node("format_output", self._format_output)
        
        # Define edges
        workflow.add_edge("process_documents", "parse_query")
        workflow.add_edge("parse_query", "generate_questions_for_domain")
        
        # Conditional edge: If more domains to process, continue generating questions
        workflow.add_conditional_edges(
            "generate_questions_for_domain",
            self._check_if_more_domains,
            {
                "continue": "generate_questions_for_domain",
                "complete": "format_output"
            }
        )
        
        workflow.add_edge("format_output", END)
        
        # Set the entry point
        workflow.set_entry_point("process_documents")
        
        return workflow
    
    def create_practice_exam(
        self,
        natural_language_query: str,
        num_questions: int,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a practice exam based on a natural language query.
        
        Args:
            natural_language_query: Natural language description of the exam to generate
            num_questions: Number of questions to generate
            output_dir: Directory to save output files
            
        Returns:
            Dictionary with results of the exam generation
        """
        logger.info(f"Creating practice exam with query: {natural_language_query}")
        
        # If output_dir is provided, use it to override the config
        if output_dir:
            self.components["output_formatter"].output_dir = output_dir
        
        # Create a job ID
        job_id = f"job_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Set up initial state
        initial_state = ExamGenerationJob(
            job_id=job_id,
            query=natural_language_query,
            target_num_questions=num_questions,
            status="initialized"
        )
        
        try:
            # Execute the workflow
            final_state = self.workflow_graph.invoke(initial_state)
            
            # Return the results
            return {
                "status": "success",
                "job_id": job_id,
                "num_questions": len(final_state.validated_mcqs),
                "html_path": final_state.output_html_path,
                "pdf_path": final_state.output_pdf_path
            }
            
        except Exception as e:
            logger.error(f"Error creating practice exam: {str(e)}")
            return {
                "status": "error",
                "job_id": job_id,
                "message": str(e)
            }
    
    def _process_documents(self, state: ExamGenerationJob) -> ExamGenerationJob:
        """
        Process documents for knowledge ingestion.
        
        Args:
            state: Current job state
            
        Returns:
            Updated job state
        """
        logger.info("Processing documents")
        state.status = "processing_documents"
        
        try:
            # Process documents
            result = self.components["knowledge_ingestion"].process_documents()
            
            logger.info(f"Processed {result.get('documents_processed', 0)} documents")
            
            # Update state
            state.status = "documents_processed"
            return state
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            state.status = "failed"
            state.error_message = f"Error processing documents: {str(e)}"
            return state
    
    def _parse_query(self, state: ExamGenerationJob) -> ExamGenerationJob:
        """
        Parse natural language query to extract domains and constraints.
        
        Args:
            state: Current job state
            
        Returns:
            Updated job state
        """
        logger.info(f"Parsing query: {state.query}")
        state.status = "parsing_query"
        
        try:
            # Simple parsing logic - this would be more sophisticated in a real implementation
            # For now, just checking if the query mentions specific domains
            
            # Predefined CLP domains (could come from a database or extracted from documents)
            clp_domains = [
                {"id": "Domain_1", "name": "Opportunity Assessment, Development, and Valuation"},
                {"id": "Domain_2", "name": "Intellectual Property Protection"},
                {"id": "Domain_3", "name": "Strategy Management Commercialization"},
                {"id": "Domain_4", "name": "Negotiation Agreement Development"},
                {"id": "Domain_5", "name": "Agreement Management"}
            ]
            
            # Check if query explicitly mentions domains
            mentioned_domains = []
            
            for domain in clp_domains:
                domain_id = domain["id"]
                domain_name = domain["name"]
                
                # Check for domain number (e.g., "Domain 1", "Domain 2")
                if f"Domain {domain_id.split('_')[1]}" in state.query or f"domain {domain_id.split('_')[1]}" in state.query.lower():
                    mentioned_domains.append(domain_id)
                    continue
                    
                # Check for domain name
                if domain_name.lower() in state.query.lower():
                    mentioned_domains.append(domain_id)
            
            # If no specific domains mentioned, include all domains
            state.target_domains = mentioned_domains if mentioned_domains else [domain["id"] for domain in clp_domains]
            
            logger.info(f"Identified target domains: {state.target_domains}")
            
            # Update state
            state.status = "query_parsed"
            return state
            
        except Exception as e:
            logger.error(f"Error parsing query: {str(e)}")
            state.status = "failed"
            state.error_message = f"Error parsing query: {str(e)}"
            return state
    
    def _generate_questions_for_domain(self, state: ExamGenerationJob) -> ExamGenerationJob:
        """
        Generate questions for the current domain.
        
        Args:
            state: Current job state
            
        Returns:
            Updated job state
        """
        # Get next domain to process
        unprocessed_domains = [d for d in state.target_domains if d not in state.processed_domains]
        
        if not unprocessed_domains:
            logger.warning("No more domains to process")
            state.status = "questions_generated"
            return state
        
        current_domain = unprocessed_domains[0]
        logger.info(f"Generating questions for domain: {current_domain}")
        state.status = f"generating_questions_for_{current_domain}"
        
        # Calculate number of questions to generate for this domain
        remaining_questions = state.target_num_questions - len(state.validated_mcqs)
        domains_left = len(unprocessed_domains)
        questions_for_domain = max(1, remaining_questions // domains_left)
        
        try:
            # Prepare domain details for context retrieval
            domain_info = {
                "clp_domain_id": current_domain,
                "clp_domain_name": self._get_domain_name(current_domain)
            }
            
            # Create search query based on domain
            search_query = f"Detailed information about CLP exam {domain_info['clp_domain_name']} for question generation"
            
            # Retrieve and analyze context
            filters = {"clp_domain_id": current_domain} if current_domain else None
            retrieved_contexts, synthesized_knowledge = self.components["rag_engine"].retrieve_and_analyze(
                query_text=search_query,
                query_details=domain_info,
                filters=filters
            )
            
            if not synthesized_knowledge.summary:
                logger.warning(f"No knowledge synthesized for domain {current_domain}")
                state.processed_domains.append(current_domain)
                return state
            
            # Generate question prompt
            question_prompt = self.components["prompt_engineer"].generate_question_prompt(
                domain_info=domain_info,
                synthesized_knowledge=synthesized_knowledge,
                target_llm_info={"name": "Gemini 2.5 Pro", "capabilities": "JSON output, long context, reasoning"},
                num_questions=questions_for_domain,
                difficulty_level="mixed"
            )
            
            # Generate questions
            generated_mcqs = self.components["question_generator"].generate_mcqs(
                prompt=question_prompt,
                synthesized_knowledge=synthesized_knowledge,
                num_questions=questions_for_domain,
                target_details=domain_info
            )
            
            logger.info(f"Generated {len(generated_mcqs)} questions for domain {current_domain}")
            
            # Validate questions with QA
            validated_mcqs = []
            
            for mcq in generated_mcqs:
                # Create source context for QA
                source_context = synthesized_knowledge.summary
                
                # Evaluate MCQ
                qa_result = self.components["quality_assurance"].evaluate_mcq(mcq, source_context)
                
                if qa_result.overall_pass:
                    logger.info(f"MCQ {mcq.question_id} passed QA")
                    validated_mcqs.append(mcq)
                else:
                    logger.warning(f"MCQ {mcq.question_id} failed QA")
                    
                    # Try to regenerate once if we have feedback
                    if qa_result.revision_suggestions:
                        logger.info(f"Attempting to regenerate MCQ {mcq.question_id}")
                        
                        regenerated_mcq = self.components["question_generator"].regenerate_single_mcq_with_feedback(
                            original_mcq_data=mcq.model_dump(),
                            synthesized_knowledge_context=source_context,
                            qa_feedback=qa_result.revision_suggestions,
                            target_details=domain_info,
                            output_schema={}  # Default schema
                        )
                        
                        if regenerated_mcq:
                            # Re-evaluate regenerated MCQ
                            regen_qa_result = self.components["quality_assurance"].evaluate_mcq(
                                regenerated_mcq, 
                                source_context
                            )
                            
                            if regen_qa_result.overall_pass:
                                logger.info(f"Regenerated MCQ {regenerated_mcq.question_id} passed QA")
                                validated_mcqs.append(regenerated_mcq)
                            else:
                                logger.warning(f"Regenerated MCQ also failed QA")
            
            # Add validated questions to state
            state.validated_mcqs.extend(validated_mcqs)
            
            # Mark domain as processed
            state.processed_domains.append(current_domain)
            
            logger.info(f"Validated {len(validated_mcqs)} questions for domain {current_domain}")
            logger.info(f"Total validated questions so far: {len(state.validated_mcqs)}/{state.target_num_questions}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating questions for domain {current_domain}: {str(e)}")
            # Mark domain as processed to avoid infinite loop
            state.processed_domains.append(current_domain)
            return state
    
    def _check_if_more_domains(self, state: ExamGenerationJob) -> str:
        """
        Check if there are more domains to process.
        
        Args:
            state: Current job state
            
        Returns:
            'continue' if more domains to process, 'complete' otherwise
        """
        # Check if we have enough questions
        if len(state.validated_mcqs) >= state.target_num_questions:
            logger.info(f"Target number of questions ({state.target_num_questions}) reached")
            return "complete"
        
        # Check if there are unprocessed domains
        unprocessed_domains = [d for d in state.target_domains if d not in state.processed_domains]
        
        if unprocessed_domains:
            logger.info(f"{len(unprocessed_domains)} domains left to process")
            return "continue"
        else:
            logger.info("All domains processed")
            return "complete"
    
    def _format_output(self, state: ExamGenerationJob) -> ExamGenerationJob:
        """
        Format and save the final output.
        
        Args:
            state: Current job state
            
        Returns:
            Updated job state
        """
        logger.info(f"Formatting output with {len(state.validated_mcqs)} validated questions")
        state.status = "formatting_output"
        
        try:
            # Generate output name based on job ID
            output_name = f"clp_practice_exam_{state.job_id}"
            
            # Generate exam title
            exam_domains = list(set([mcq.clp_domain_id for mcq in state.validated_mcqs]))
            if len(exam_domains) == 1:
                domain_name = self._get_domain_name(exam_domains[0])
                exam_title = f"CLP Practice Exam: {domain_name} ({len(state.validated_mcqs)} Questions)"
            else:
                exam_title = f"CLP Practice Exam: Multiple Domains ({len(state.validated_mcqs)} Questions)"
            
            # Generate outputs
            output_results = self.components["output_formatter"].generate_outputs(
                mcqs=state.validated_mcqs,
                output_name=output_name,
                exam_title=exam_title,
                formats=['json', 'html', 'pdf']
            )
            
            # Update state with output paths
            state.output_html_path = output_results.get('html')
            state.output_pdf_path = output_results.get('pdf')
            
            # Mark as complete
            state.status = "completed"
            state.end_time = datetime.datetime.now()
            
            logger.info(f"Exam generation complete")
            return state
            
        except Exception as e:
            logger.error(f"Error formatting output: {str(e)}")
            state.status = "failed"
            state.error_message = f"Error formatting output: {str(e)}"
            return state
    
    def _get_domain_name(self, domain_id: str) -> str:
        """
        Get domain name from domain ID.
        
        Args:
            domain_id: Domain ID
            
        Returns:
            Domain name
        """
        # Hardcoded domain names (in a real implementation, these would come from a database)
        domain_names = {
            "Domain_1": "Opportunity Assessment, Development, and Valuation",
            "Domain_2": "Intellectual Property Protection",
            "Domain_3": "Strategy Management Commercialization",
            "Domain_4": "Negotiation Agreement Development",
            "Domain_5": "Agreement Management"
        }
        
        return domain_names.get(domain_id, domain_id)