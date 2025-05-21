"""
Core data models for the APEGA system.
These models represent the various data structures used throughout the application.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Types of documents that can be processed."""
    PDF = "pdf"
    HTML = "html"
    TEXT = "text"


class ChunkType(str, Enum):
    """Types of document chunks."""
    TEXT = "text"
    TABLE = "table"
    HEADING = "heading"
    LIST = "list"


class StructuredTable(BaseModel):
    """Representation of a table extracted from a document."""
    table_id: str
    data: List[List[str]]
    caption: Optional[str] = None
    page_number: Optional[int] = None


class ParsedDocument(BaseModel):
    """A document after parsing, containing text and metadata."""
    document_id: str
    title: Optional[str] = None
    source_path: str
    document_type: DocumentType
    text_content: str
    tables: List[StructuredTable] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parsed_at: datetime = Field(default_factory=datetime.now)


class TextChunk(BaseModel):
    """A chunk of text from a document with metadata."""
    chunk_id: str
    document_id: str
    text: str
    chunk_type: ChunkType = ChunkType.TEXT
    page_number: Optional[int] = None
    section_path: Optional[List[str]] = None  # Hierarchical path of headings
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddedChunk(BaseModel):
    """A text chunk with its vector embedding."""
    chunk_id: str
    document_id: str
    text: str
    embedding_vector: Optional[List[float]] = None  # Can be None if embedding failed
    chunk_type: ChunkType = ChunkType.TEXT
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedContext(BaseModel):
    """A context chunk retrieved from the knowledge base."""
    chunk_id: str
    document_id: str
    text: str
    initial_score: float  # Score from initial hybrid search
    rerank_score: Optional[float] = None  # Score after reranking, if performed
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SynthesizedKnowledge(BaseModel):
    """Knowledge synthesized by deep analysis of retrieved contexts."""
    summary: str
    key_concepts: List[Dict[str, str]] = Field(default_factory=list)
    potential_exam_areas: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Option(BaseModel):
    """An option for a multiple-choice question."""
    option_id: str  # e.g., "A", "B", "C", "D"
    text: str


class GeneratedMCQ(BaseModel):
    """A generated multiple-choice question."""
    question_id: str
    clp_domain_id: str
    clp_domain_name: Optional[str] = None
    clp_task_id: Optional[str] = None
    clp_task_statement: Optional[str] = None
    question_stem: str
    options: List[Option]
    correct_option_id: str
    explanation: Optional[str] = None
    difficulty_level_assessed: Optional[str] = None
    cognitive_skill_targeted: Optional[str] = None
    source_references: List[Dict[str, str]] = Field(default_factory=list)
    
    def to_json_str(self) -> str:
        """Return a JSON string representation of the MCQ."""
        return self.model_dump_json(indent=2)


class QACriterionEvaluation(BaseModel):
    """Evaluation of a question against a specific QA criterion."""
    score: str  # e.g., "Pass", "Fail", "3/5", depending on criterion
    justification: str


class QAResult(BaseModel):
    """Result of quality assurance evaluation for an MCQ."""
    mcq_id: str
    overall_pass: bool
    criteria_feedback: Dict[str, QACriterionEvaluation]
    revision_suggestions: Optional[str] = None


class ExamGenerationJob(BaseModel):
    """Represents the state of an exam generation job."""
    job_id: str = Field(default_factory=lambda: f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    query: str
    target_num_questions: int
    validated_mcqs: List[GeneratedMCQ] = Field(default_factory=list)
    target_domains: List[str] = Field(default_factory=list)
    processed_domains: List[str] = Field(default_factory=list)
    status: str = "initialized"  # initialized, in_progress, completed, failed
    error_message: Optional[str] = None
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    output_html_path: Optional[str] = None
    output_pdf_path: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if the job is complete."""
        return len(self.validated_mcqs) >= self.target_num_questions or self.status == "completed"
