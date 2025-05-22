"""
Text Chunker for APEGA.
Splits parsed documents into manageable, semantically coherent chunks.
"""

import re
import nltk
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import os
from enum import Enum

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from src.models.data_models import ParsedDocument, TextChunk, ChunkType


class ChunkingStrategy(str, Enum):
    """Strategies for text chunking."""
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    SLIDING_WINDOW = "sliding_window"
    HIERARCHICAL = "hierarchical"
    SEMANTIC = "semantic"
    HYBRID_HIERARCHICAL_SEMANTIC = "hybrid_hierarchical_semantic"


class TextChunker:
    """
    Splits parsed documents into manageable, semantically coherent chunks.
    Supports various chunking strategies with graceful fallbacks.
    """
    
    def __init__(
        self, 
        strategy: str = 'hybrid_hierarchical_semantic',
        max_chunk_size_tokens: int = 1024,
        chunk_overlap_tokens: int = 200
    ):
        """
        Initialize the TextChunker.
        
        Args:
            strategy: Chunking strategy to use
            max_chunk_size_tokens: Maximum number of tokens per chunk
            chunk_overlap_tokens: Number of tokens to overlap between chunks
        """
        self.strategy = strategy
        self.max_chunk_size_tokens = max_chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        
        # Optional semantic splitter with improved error handling
        self.semantic_splitter = None
        self.semantic_available = False
        
        if strategy in [ChunkingStrategy.SEMANTIC, ChunkingStrategy.HYBRID_HIERARCHICAL_SEMANTIC]:
            self.semantic_available = self._initialize_semantic_splitter()
            
            # If semantic chunking was requested but not available, adjust strategy
            if not self.semantic_available:
                if strategy == ChunkingStrategy.SEMANTIC:
                    logger.warning("Semantic chunking requested but not available. Falling back to paragraph chunking.")
                    self.strategy = ChunkingStrategy.PARAGRAPH
                elif strategy == ChunkingStrategy.HYBRID_HIERARCHICAL_SEMANTIC:
                    logger.warning("Hybrid semantic chunking requested but semantic component not available. Using hierarchical chunking only.")
                    self.strategy = ChunkingStrategy.HIERARCHICAL
    
    def _initialize_semantic_splitter(self) -> bool:
        """
        Initialize the semantic splitter with proper error handling.
        
        Returns:
            True if semantic splitter is available, False otherwise
        """
        try:
            # Import here to avoid unnecessary dependencies if not using semantic chunking
            from sentence_transformers import SentenceTransformer
            
            logger.info("Initializing semantic text splitter...")
            self.semantic_splitter = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Semantic splitter initialized successfully")
            return True
            
        except ImportError as e:
            logger.warning(f"sentence-transformers library not available: {e}")
            return False
        except Exception as e:
            # This will catch network errors, model download failures, etc.
            logger.warning(f"Failed to initialize semantic splitter: {e}")
            logger.info("This could be due to:")
            logger.info("  - No internet connection to download the model")
            logger.info("  - Network restrictions blocking access to huggingface.co")
            logger.info("  - Insufficient disk space for model download")
            logger.info("  - Missing dependencies for the transformers library")
            logger.info("Continuing with non-semantic chunking strategies...")
            return False
    
    def chunk_document(self, parsed_doc: ParsedDocument) -> List[TextChunk]:
        """
        Split a parsed document into chunks according to the selected strategy.
        
        Args:
            parsed_doc: The parsed document to chunk
            
        Returns:
            A list of TextChunk objects
        """
        # Get document structure from metadata if available
        toc = parsed_doc.metadata.get('toc', [])
        
        # Create the document hierarchy based on TOC
        document_hierarchy = self._create_document_hierarchy(parsed_doc.text_content, toc)
        
        # Choose chunking strategy with fallback handling
        try:
            if self.strategy == ChunkingStrategy.PARAGRAPH:
                chunks = self._paragraph_chunking(parsed_doc)
            elif self.strategy == ChunkingStrategy.SENTENCE:
                chunks = self._sentence_chunking(parsed_doc)
            elif self.strategy == ChunkingStrategy.SLIDING_WINDOW:
                chunks = self._sliding_window_chunking(parsed_doc)
            elif self.strategy == ChunkingStrategy.HIERARCHICAL:
                chunks = self._hierarchical_chunking(parsed_doc, document_hierarchy)
            elif self.strategy == ChunkingStrategy.SEMANTIC:
                chunks = self._semantic_chunking(parsed_doc)
            elif self.strategy == ChunkingStrategy.HYBRID_HIERARCHICAL_SEMANTIC:
                chunks = self._hybrid_hierarchical_semantic_chunking(parsed_doc, document_hierarchy)
            else:
                logger.warning(f"Unknown chunking strategy: {self.strategy}. Falling back to paragraph chunking.")
                chunks = self._paragraph_chunking(parsed_doc)
        except Exception as e:
            logger.error(f"Error during {self.strategy} chunking: {e}")
            logger.info("Falling back to paragraph chunking...")
            chunks = self._paragraph_chunking(parsed_doc)
        
        # Add table chunks if there are tables
        try:
            table_chunks = self._create_table_chunks(parsed_doc)
            chunks.extend(table_chunks)
        except Exception as e:
            logger.warning(f"Error creating table chunks: {e}")
        
        # Ensure chunks are within token limits
        try:
            chunks = self._enforce_chunk_size_limits(chunks)
        except Exception as e:
            logger.warning(f"Error enforcing chunk size limits: {e}")
        
        logger.info(f"Successfully created {len(chunks)} chunks using {self.strategy} strategy")
        return chunks
    
    def _create_document_hierarchy(self, text_content: str, toc: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a hierarchical structure of the document based on the table of contents.
        
        Args:
            text_content: The full text content of the document
            toc: The table of contents from the document metadata
            
        Returns:
            A nested dictionary representing the document structure
        """
        # If no TOC, try to infer structure from text
        if not toc:
            return self._infer_structure_from_text(text_content)
        
        # Create a hierarchical structure
        root = {"title": "Document Root", "level": 0, "children": [], "content": ""}
        current_nodes = {0: root}
        
        # Sort TOC entries by page number
        sorted_toc = sorted(toc, key=lambda x: x.get('page', 0))
        
        for item in sorted_toc:
            level = item.get('level', 1)
            title = item.get('title', '')
            page = item.get('page', 0)
            
            # Create new node
            new_node = {"title": title, "level": level, "page": page, "children": [], "content": ""}
            
            # Find the parent node
            parent_level = max(l for l in current_nodes.keys() if l < level)
            current_nodes[parent_level]["children"].append(new_node)
            current_nodes[level] = new_node
            
            # Remove any nodes with higher levels (they are no longer current)
            higher_levels = [l for l in current_nodes.keys() if l > level]
            for l in higher_levels:
                if l in current_nodes:
                    del current_nodes[l]
        
        return root
    
    def _infer_structure_from_text(self, text_content: str) -> Dict[str, Any]:
        """
        Infer document structure from text when no TOC is available.
        
        Args:
            text_content: The full text content of the document
            
        Returns:
            A nested dictionary representing the inferred document structure
        """
        root = {"title": "Document Root", "level": 0, "children": [], "content": ""}
        
        # Split by pages
        pages = re.split(r'PAGE \d+\n', text_content)
        if len(pages) > 1:
            pages = pages[1:]  # Skip the first empty split if it exists
        
        # Find potential headings using regex patterns
        heading_patterns = [
            # Chapter or section patterns like "1. Introduction" or "Chapter 1: Introduction"
            r'^(?:Chapter|Section)?\s*(\d+(?:\.\d+)*)\.?\s*([A-Z].*?)$',
            # Heading patterns like "INTRODUCTION" or "Introduction"
            r'^([A-Z][A-Z\s]+)$',
            r'^([A-Z][a-z].*?)$'
        ]
        
        current_level = 0
        current_node = root
        
        for page in pages:
            lines = page.split('\n')
            current_content = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if the line matches a heading pattern
                is_heading = False
                for pattern in heading_patterns:
                    match = re.match(pattern, line)
                    if match:
                        # Process previous content if any
                        if current_content:
                            current_node["content"] += '\n'.join(current_content) + '\n'
                            current_content = []
                        
                        # Extract heading level (if numbered) or use default
                        if len(match.groups()) > 1 and '.' in match.group(1):
                            # For numbered headings like "1.2.3", use the number of levels
                            level = len(match.group(1).split('.'))
                        else:
                            # For other headings, use a default level based on text properties
                            if line.isupper():
                                level = 1  # All caps suggests a major heading
                            else:
                                level = 2  # Default for other patterns
                        
                        # Create new node
                        title = line
                        new_node = {"title": title, "level": level, "children": [], "content": ""}
                        
                        # Adjust the tree structure
                        if level <= current_level:
                            # Move up to appropriate parent
                            while current_node != root and current_node["level"] >= level:
                                current_node = [node for node in root["children"] if current_node in node["children"]][0]
                        
                        # Add as child of current node
                        current_node["children"].append(new_node)
                        current_node = new_node
                        current_level = level
                        
                        is_heading = True
                        break
                
                if not is_heading:
                    current_content.append(line)
            
            # Add remaining content to current node
            if current_content:
                current_node["content"] += '\n'.join(current_content) + '\n'
        
        return root
    
    def _paragraph_chunking(self, parsed_doc: ParsedDocument) -> List[TextChunk]:
        """
        Split the document into chunks based on paragraphs.
        
        Args:
            parsed_doc: The parsed document
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        text = parsed_doc.text_content
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk_text = ""
        current_chunk_tokens = 0
        chunk_index = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Estimate token count (rough approximation)
            para_tokens = len(paragraph.split())
            
            # Skip empty, very short, or header-like paragraphs
            if para_tokens < 5 and not re.match(r'^[A-Z].*[\.!?]$', paragraph):
                continue
            
            # If adding this paragraph would exceed max size, create a new chunk
            if current_chunk_tokens + para_tokens > self.max_chunk_size_tokens and current_chunk_text:
                chunk = TextChunk(
                    chunk_id=f"{parsed_doc.document_id}_chunk_{chunk_index}",
                    document_id=parsed_doc.document_id,
                    text=current_chunk_text.strip(),
                    metadata={
                        "source": parsed_doc.source_path,
                        "chunk_strategy": ChunkingStrategy.PARAGRAPH
                    }
                )
                chunks.append(chunk)
                
                chunk_index += 1
                current_chunk_text = paragraph + "\n\n"
                current_chunk_tokens = para_tokens
            else:
                current_chunk_text += paragraph + "\n\n"
                current_chunk_tokens += para_tokens
        
        # Add the last chunk if not empty
        if current_chunk_text.strip():
            chunk = TextChunk(
                chunk_id=f"{parsed_doc.document_id}_chunk_{chunk_index}",
                document_id=parsed_doc.document_id,
                text=current_chunk_text.strip(),
                metadata={
                    "source": parsed_doc.source_path,
                    "chunk_strategy": ChunkingStrategy.PARAGRAPH
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _sentence_chunking(self, parsed_doc: ParsedDocument) -> List[TextChunk]:
        """
        Split the document into chunks based on sentences.
        
        Args:
            parsed_doc: The parsed document
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        text = parsed_doc.text_content
        sentences = nltk.sent_tokenize(text)
        
        current_chunk_text = ""
        current_chunk_tokens = 0
        chunk_index = 0
        
        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Estimate token count
            sent_tokens = len(sentence.split())
            
            # If adding this sentence would exceed max size, create a new chunk
            if current_chunk_tokens + sent_tokens > self.max_chunk_size_tokens and current_chunk_text:
                chunk = TextChunk(
                    chunk_id=f"{parsed_doc.document_id}_chunk_{chunk_index}",
                    document_id=parsed_doc.document_id,
                    text=current_chunk_text.strip(),
                    metadata={
                        "source": parsed_doc.source_path,
                        "chunk_strategy": ChunkingStrategy.SENTENCE
                    }
                )
                chunks.append(chunk)
                
                chunk_index += 1
                current_chunk_text = sentence + " "
                current_chunk_tokens = sent_tokens
            else:
                current_chunk_text += sentence + " "
                current_chunk_tokens += sent_tokens
        
        # Add the last chunk if not empty
        if current_chunk_text.strip():
            chunk = TextChunk(
                chunk_id=f"{parsed_doc.document_id}_chunk_{chunk_index}",
                document_id=parsed_doc.document_id,
                text=current_chunk_text.strip(),
                metadata={
                    "source": parsed_doc.source_path,
                    "chunk_strategy": ChunkingStrategy.SENTENCE
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _sliding_window_chunking(self, parsed_doc: ParsedDocument) -> List[TextChunk]:
        """
        Split the document using a sliding window approach.
        
        Args:
            parsed_doc: The parsed document
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        text = parsed_doc.text_content
        
        # First, split into sentences for more precise chunking
        sentences = nltk.sent_tokenize(text)
        
        # Skip empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunk_index = 0
        i = 0
        
        while i < len(sentences):
            # Start a new chunk
            current_chunk = []
            current_tokens = 0
            
            # Add sentences until we reach max chunk size
            while i < len(sentences) and current_tokens < self.max_chunk_size_tokens:
                sentence = sentences[i]
                sentence_tokens = len(sentence.split())
                
                if current_tokens + sentence_tokens <= self.max_chunk_size_tokens:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
                    i += 1
                else:
                    break
            
            # Create a chunk from the collected sentences
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk = TextChunk(
                    chunk_id=f"{parsed_doc.document_id}_chunk_{chunk_index}",
                    document_id=parsed_doc.document_id,
                    text=chunk_text,
                    metadata={
                        "source": parsed_doc.source_path,
                        "chunk_strategy": ChunkingStrategy.SLIDING_WINDOW
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Slide the window back for overlap
            overlap_tokens = 0
            i_temp = i - 1
            
            while i_temp >= 0 and overlap_tokens < self.chunk_overlap_tokens:
                sentence = sentences[i_temp]
                sentence_tokens = len(sentence.split())
                overlap_tokens += sentence_tokens
                i_temp -= 1
            
            # Set starting point for next chunk to include overlap
            i = max(0, i_temp + 1)
        
        return chunks
    
    def _hierarchical_chunking(self, parsed_doc: ParsedDocument, document_hierarchy: Dict[str, Any]) -> List[TextChunk]:
        """
        Split the document based on its hierarchical structure (sections, subsections).
        
        Args:
            parsed_doc: The parsed document
            document_hierarchy: Hierarchical structure of the document
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        
        # Recursive function to traverse the hierarchy
        def process_node(node, path=None):
            if path is None:
                path = []
            
            # Create a new path including this node
            current_path = path + [node.get("title", "Untitled")]
            
            # Extract content for this node
            content = node.get("content", "").strip()
            
            # Only process if there's content or children
            if content or node.get("children"):
                # Create chunks for this node's content if not empty and not too small
                if content and len(content.split()) > 20:  # Skip if too short
                    chunk = TextChunk(
                        chunk_id=f"{parsed_doc.document_id}_section_{'_'.join(str(p) for p in current_path if p)}",
                        document_id=parsed_doc.document_id,
                        text=content,
                        section_path=current_path,
                        metadata={
                            "source": parsed_doc.source_path,
                            "chunk_strategy": ChunkingStrategy.HIERARCHICAL,
                            "section_level": node.get("level", 0),
                            "section_title": node.get("title", "")
                        }
                    )
                    chunks.append(chunk)
                
                # Process children recursively
                for child in node.get("children", []):
                    process_node(child, current_path)
        
        # Start processing from the root
        process_node(document_hierarchy)
        
        # If no chunks were created, fall back to paragraph chunking
        if not chunks:
            logger.warning("Hierarchical chunking produced no chunks. Falling back to paragraph chunking.")
            return self._paragraph_chunking(parsed_doc)
        
        return chunks
    
    def _semantic_chunking(self, parsed_doc: ParsedDocument) -> List[TextChunk]:
        """
        Split the document based on semantic meaning shifts.
        
        Args:
            parsed_doc: The parsed document
            
        Returns:
            List of TextChunk objects
        """
        # If semantic_splitter is not available, fall back to paragraph chunking
        if not self.semantic_splitter:
            logger.warning("Semantic chunker not available. Falling back to paragraph chunking.")
            return self._paragraph_chunking(parsed_doc)
        
        chunks = []
        text = parsed_doc.text_content
        
        # First, split into paragraphs
        paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        # Skip if too few paragraphs
        if len(paragraphs) <= 1:
            return self._paragraph_chunking(parsed_doc)
        
        # Get embeddings for each paragraph
        try:
            embeddings = self.semantic_splitter.encode(paragraphs)
            
            # Detect semantic shifts using cosine similarity
            from numpy import dot
            from numpy.linalg import norm
            
            current_chunk_paras = [paragraphs[0]]
            current_chunk_tokens = len(paragraphs[0].split())
            chunk_index = 0
            
            # Group paragraphs with similar embeddings
            for i in range(1, len(paragraphs)):
                para = paragraphs[i]
                para_tokens = len(para.split())
                
                # Always respect max token limit
                if current_chunk_tokens + para_tokens > self.max_chunk_size_tokens:
                    # Create chunk from collected paragraphs
                    chunk_text = "\n\n".join(current_chunk_paras)
                    chunk = TextChunk(
                        chunk_id=f"{parsed_doc.document_id}_semantic_{chunk_index}",
                        document_id=parsed_doc.document_id,
                        text=chunk_text,
                        metadata={
                            "source": parsed_doc.source_path,
                            "chunk_strategy": ChunkingStrategy.SEMANTIC
                        }
                    )
                    chunks.append(chunk)
                    
                    chunk_index += 1
                    current_chunk_paras = [para]
                    current_chunk_tokens = para_tokens
                    continue
                
                # Check semantic similarity with last paragraph in current chunk
                prev_embedding = embeddings[i-1]
                curr_embedding = embeddings[i]
                
                similarity = dot(prev_embedding, curr_embedding) / (norm(prev_embedding) * norm(curr_embedding))
                
                # If similarity is high, add to current chunk, otherwise start a new chunk
                if similarity > 0.7:  # Threshold can be adjusted
                    current_chunk_paras.append(para)
                    current_chunk_tokens += para_tokens
                else:
                    # Create chunk from collected paragraphs
                    chunk_text = "\n\n".join(current_chunk_paras)
                    chunk = TextChunk(
                        chunk_id=f"{parsed_doc.document_id}_semantic_{chunk_index}",
                        document_id=parsed_doc.document_id,
                        text=chunk_text,
                        metadata={
                            "source": parsed_doc.source_path,
                            "chunk_strategy": ChunkingStrategy.SEMANTIC
                        }
                    )
                    chunks.append(chunk)
                    
                    chunk_index += 1
                    current_chunk_paras = [para]
                    current_chunk_tokens = para_tokens
            
            # Add the last chunk if not empty
            if current_chunk_paras:
                chunk_text = "\n\n".join(current_chunk_paras)
                chunk = TextChunk(
                    chunk_id=f"{parsed_doc.document_id}_semantic_{chunk_index}",
                    document_id=parsed_doc.document_id,
                    text=chunk_text,
                    metadata={
                        "source": parsed_doc.source_path,
                        "chunk_strategy": ChunkingStrategy.SEMANTIC
                    }
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error during semantic chunking: {e}. Falling back to paragraph chunking.")
            return self._paragraph_chunking(parsed_doc)
    
    def _hybrid_hierarchical_semantic_chunking(self, parsed_doc: ParsedDocument, document_hierarchy: Dict[str, Any]) -> List[TextChunk]:
        """
        Split the document using a hybrid approach: hierarchical first, then semantic within sections.
        
        Args:
            parsed_doc: The parsed document
            document_hierarchy: Hierarchical structure of the document
            
        Returns:
            List of TextChunk objects
        """
        # First get hierarchical chunks
        hierarchical_chunks = self._hierarchical_chunking(parsed_doc, document_hierarchy)
        
        # If no hierarchical chunks or semantic splitting not available, return hierarchical chunks
        if not hierarchical_chunks or not self.semantic_splitter:
            if not self.semantic_splitter:
                logger.info("Semantic component not available for hybrid chunking, using hierarchical only")
            return hierarchical_chunks
        
        # For each hierarchical chunk that's too large, apply semantic chunking
        final_chunks = []
        
        for h_chunk in hierarchical_chunks:
            chunk_tokens = len(h_chunk.text.split())
            
            if chunk_tokens <= self.max_chunk_size_tokens:
                # Chunk is within size limit, keep as is
                final_chunks.append(h_chunk)
            else:
                # Create a mini document for semantic chunking
                mini_doc = ParsedDocument(
                    document_id=h_chunk.chunk_id,
                    source_path=parsed_doc.source_path,
                    document_type=parsed_doc.document_type,
                    text_content=h_chunk.text,
                    metadata=h_chunk.metadata
                )
                
                # Apply semantic chunking to this section
                semantic_chunks = self._semantic_chunking(mini_doc)
                
                # Update chunk IDs and section paths
                for i, s_chunk in enumerate(semantic_chunks):
                    s_chunk.chunk_id = f"{h_chunk.chunk_id}_sub_{i}"
                    s_chunk.section_path = h_chunk.section_path
                    s_chunk.metadata.update({
                        "chunk_strategy": ChunkingStrategy.HYBRID_HIERARCHICAL_SEMANTIC,
                        "section_level": h_chunk.metadata.get("section_level"),
                        "section_title": h_chunk.metadata.get("section_title")
                    })
                
                final_chunks.extend(semantic_chunks)
        
        return final_chunks
    
    def _create_table_chunks(self, parsed_doc: ParsedDocument) -> List[TextChunk]:
        """
        Create separate chunks for tables in the document.
        
        Args:
            parsed_doc: The parsed document
            
        Returns:
            List of TextChunk objects for tables
        """
        table_chunks = []
        
        for table in parsed_doc.tables:
            # Convert table data to a string representation
            table_text = ""
            
            # Add caption if available
            if table.caption:
                table_text += f"Table Caption: {table.caption}\n\n"
            
            # Add table content
            for row in table.data:
                table_text += " | ".join(row) + "\n"
            
            # Create a chunk for this table
            chunk = TextChunk(
                chunk_id=f"{parsed_doc.document_id}_{table.table_id}",
                document_id=parsed_doc.document_id,
                text=table_text.strip(),
                chunk_type=ChunkType.TABLE,
                page_number=table.page_number,
                metadata={
                    "source": parsed_doc.source_path,
                    "chunk_strategy": "table",
                    "table_id": table.table_id
                }
            )
            table_chunks.append(chunk)
        
        return table_chunks
    
    def _enforce_chunk_size_limits(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Ensure all chunks are within the token limit.
        Split any oversized chunks.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            List of TextChunk objects within token limits
        """
        final_chunks = []
        
        for chunk in chunks:
            chunk_tokens = len(chunk.text.split())
            
            if chunk_tokens <= self.max_chunk_size_tokens:
                final_chunks.append(chunk)
            else:
                # Chunk is too large, split it by sentences
                sentences = nltk.sent_tokenize(chunk.text)
                
                current_text = ""
                current_tokens = 0
                split_index = 0
                
                for sentence in sentences:
                    sentence_tokens = len(sentence.split())
                    
                    if current_tokens + sentence_tokens <= self.max_chunk_size_tokens:
                        current_text += sentence + " "
                        current_tokens += sentence_tokens
                    else:
                        # Create a new chunk with accumulated text
                        if current_text:
                            sub_chunk = TextChunk(
                                chunk_id=f"{chunk.chunk_id}_split_{split_index}",
                                document_id=chunk.document_id,
                                text=current_text.strip(),
                                chunk_type=chunk.chunk_type,
                                page_number=chunk.page_number,
                                section_path=chunk.section_path,
                                metadata=chunk.metadata.copy()
                            )
                            sub_chunk.metadata["split_from"] = chunk.chunk_id
                            final_chunks.append(sub_chunk)
                            
                            split_index += 1
                            current_text = sentence + " "
                            current_tokens = sentence_tokens
                
                # Add final sub-chunk if not empty
                if current_text:
                    sub_chunk = TextChunk(
                        chunk_id=f"{chunk.chunk_id}_split_{split_index}",
                        document_id=chunk.document_id,
                        text=current_text.strip(),
                        chunk_type=chunk.chunk_type,
                        page_number=chunk.page_number,
                        section_path=chunk.section_path,
                        metadata=chunk.metadata.copy()
                    )
                    sub_chunk.metadata["split_from"] = chunk.chunk_id
                    final_chunks.append(sub_chunk)
        
        return final_chunks