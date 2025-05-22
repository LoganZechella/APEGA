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
        # Validate input
        if not parsed_doc:
            logger.error("Received None parsed_doc")
            return []
        
        if not hasattr(parsed_doc, 'text_content') or not parsed_doc.text_content:
            logger.warning(f"Document {parsed_doc.document_id if hasattr(parsed_doc, 'document_id') else 'unknown'} has no text content")
            # Return a minimal chunk with empty content if document exists but has no text
            if hasattr(parsed_doc, 'document_id'):
                empty_chunk = TextChunk(
                    chunk_id=f"{parsed_doc.document_id}_empty_0",
                    document_id=parsed_doc.document_id,
                    text="",
                    metadata={
                        "source": getattr(parsed_doc, 'source_path', 'unknown'),
                        "chunk_strategy": "empty_document",
                        "note": "Document had no extractable text content"
                    }
                )
                return [empty_chunk]
            else:
                return []
        
        # Safely get document structure from metadata if available
        try:
            toc = parsed_doc.metadata.get('toc', []) if hasattr(parsed_doc, 'metadata') and parsed_doc.metadata else []
        except Exception as e:
            logger.warning(f"Error accessing document metadata: {e}")
            toc = []
        
        # Create the document hierarchy based on TOC with error handling
        try:
            document_hierarchy = self._create_document_hierarchy(parsed_doc.text_content, toc)
        except Exception as e:
            logger.error(f"Error creating document hierarchy: {e}")
            # Create a minimal hierarchy for fallback
            document_hierarchy = {"title": "Document Root", "level": 0, "children": [], "content": parsed_doc.text_content}
        
        # Choose chunking strategy with fallback handling
        chunks = []
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
            try:
                chunks = self._paragraph_chunking(parsed_doc)
            except Exception as fallback_error:
                logger.error(f"Fallback paragraph chunking also failed: {fallback_error}")
                # Create a single chunk with the entire content as last resort
                try:
                    emergency_chunk = TextChunk(
                        chunk_id=f"{parsed_doc.document_id}_emergency_0",
                        document_id=parsed_doc.document_id,
                        text=parsed_doc.text_content[:self.max_chunk_size_tokens * 4],  # Rough token limit
                        metadata={
                            "source": getattr(parsed_doc, 'source_path', 'unknown'),
                            "chunk_strategy": "emergency_fallback",
                            "note": "All chunking strategies failed, using emergency single chunk"
                        }
                    )
                    chunks = [emergency_chunk]
                except Exception as emergency_error:
                    logger.error(f"Even emergency chunking failed: {emergency_error}")
                    return []
        
        # Add table chunks if there are tables
        try:
            if hasattr(parsed_doc, 'tables') and parsed_doc.tables:
                table_chunks = self._create_table_chunks(parsed_doc)
                chunks.extend(table_chunks)
        except Exception as e:
            logger.warning(f"Error creating table chunks: {e}")
        
        # Ensure chunks are within token limits
        try:
            chunks = self._enforce_chunk_size_limits(chunks)
        except Exception as e:
            logger.warning(f"Error enforcing chunk size limits: {e}")
            # If chunk size enforcement fails, at least filter out empty chunks
            try:
                chunks = [chunk for chunk in chunks if chunk.text and chunk.text.strip()]
            except Exception as filter_error:
                logger.error(f"Error filtering chunks: {filter_error}")
        
        # Final validation - ensure we have at least one chunk
        if not chunks:
            logger.warning("No chunks created, creating emergency fallback chunk")
            try:
                emergency_chunk = TextChunk(
                    chunk_id=f"{parsed_doc.document_id}_fallback_0",
                    document_id=parsed_doc.document_id,
                    text=parsed_doc.text_content[:1000] if len(parsed_doc.text_content) > 1000 else parsed_doc.text_content,
                    metadata={
                        "source": getattr(parsed_doc, 'source_path', 'unknown'),
                        "chunk_strategy": "final_fallback",
                        "note": "No chunks could be created through normal methods"
                    }
                )
                chunks = [emergency_chunk]
            except Exception as final_error:
                logger.error(f"Final fallback chunking failed: {final_error}")
                return []
        
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
        # Validate inputs
        if not text_content:
            logger.warning("Empty text content provided to _create_document_hierarchy")
            text_content = ""
        
        if not toc or not isinstance(toc, list):
            logger.info("No valid TOC provided, inferring structure from text")
            return self._infer_structure_from_text(text_content)
        
        # Create a hierarchical structure
        root = {"title": "Document Root", "level": 0, "children": [], "content": ""}
        
        try:
            current_nodes = {0: root}
            
            # Sort TOC entries by page number, handling missing or invalid page numbers
            def get_page_number(item):
                try:
                    return item.get('page', 0) if isinstance(item, dict) else 0
                except:
                    return 0
            
            try:
                sorted_toc = sorted(toc, key=get_page_number)
            except Exception as sort_error:
                logger.warning(f"Error sorting TOC: {sort_error}, using original order")
                sorted_toc = toc
            
            for item_idx, item in enumerate(sorted_toc):
                try:
                    # Safely extract item properties
                    if not isinstance(item, dict):
                        logger.warning(f"TOC item {item_idx} is not a dictionary: {item}")
                        continue
                    
                    level = item.get('level', 1)
                    title = item.get('title', f'Untitled Section {item_idx}')
                    page = item.get('page', 0)
                    
                    # Validate level
                    if not isinstance(level, int) or level < 1:
                        logger.warning(f"Invalid level {level} in TOC item, using level 1")
                        level = 1
                    
                    # Create new node
                    new_node = {"title": str(title), "level": level, "page": page, "children": [], "content": ""}
                    
                    # Find the parent node safely
                    try:
                        # Find the highest level that's less than current level
                        parent_levels = [l for l in current_nodes.keys() if l < level]
                        if parent_levels:
                            parent_level = max(parent_levels)
                            parent_node = current_nodes[parent_level]
                        else:
                            # If no valid parent found, use root
                            parent_node = root
                            parent_level = 0
                        
                        # Add to parent
                        parent_node["children"].append(new_node)
                        current_nodes[level] = new_node
                        
                        # Remove any nodes with higher levels (they are no longer current)
                        higher_levels = [l for l in list(current_nodes.keys()) if l > level]
                        for l in higher_levels:
                            if l in current_nodes:
                                del current_nodes[l]
                                
                    except Exception as hierarchy_error:
                        logger.warning(f"Error building hierarchy for TOC item '{title}': {hierarchy_error}")
                        # Fallback: add directly to root
                        root["children"].append(new_node)
                        current_nodes = {0: root, level: new_node}
                        
                except Exception as item_error:
                    logger.warning(f"Error processing TOC item {item_idx}: {item_error}")
                    continue
            
        except Exception as e:
            logger.error(f"Error creating document hierarchy from TOC: {e}")
            logger.info("Falling back to text-based structure inference")
            return self._infer_structure_from_text(text_content)
        
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
        
        # Handle empty or None text content
        if not text_content or not text_content.strip():
            logger.warning("Empty or None text content provided for structure inference")
            root["content"] = text_content or ""
            return root
        
        # Split by pages
        pages = re.split(r'PAGE \d+\n', text_content)
        if len(pages) > 1:
            pages = pages[1:]  # Skip the first empty split if it exists
        else:
            # If no page breaks found, treat entire text as one page
            pages = [text_content]
        
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
        node_stack = [root]  # Keep track of node hierarchy for safe navigation
        
        for page_idx, page in enumerate(pages):
            if not page or not page.strip():
                continue
                
            lines = page.split('\n')
            current_content = []
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Check if the line matches a heading pattern
                is_heading = False
                for pattern in heading_patterns:
                    try:
                        match = re.match(pattern, line)
                        if match:
                            # Process previous content if any
                            if current_content:
                                current_node["content"] += '\n'.join(current_content) + '\n'
                                current_content = []
                            
                            # Extract heading level (if numbered) or use default
                            level = 1  # Default level
                            if len(match.groups()) > 1 and match.group(1) and '.' in match.group(1):
                                # For numbered headings like "1.2.3", use the number of levels
                                try:
                                    level = len(match.group(1).split('.'))
                                except:
                                    level = 1
                            else:
                                # For other headings, use a default level based on text properties
                                if line.isupper():
                                    level = 1  # All caps suggests a major heading
                                else:
                                    level = 2  # Default for other patterns
                            
                            # Create new node
                            title = line
                            new_node = {"title": title, "level": level, "children": [], "content": ""}
                            
                            # Safely adjust the tree structure
                            try:
                                # Find the correct parent level
                                target_parent_level = level - 1
                                
                                # Walk back through the stack to find appropriate parent
                                while len(node_stack) > 1 and node_stack[-1]["level"] >= level:
                                    node_stack.pop()
                                
                                # The parent is now the last item in the stack
                                parent_node = node_stack[-1] if node_stack else root
                                
                                # Add as child of parent node
                                parent_node["children"].append(new_node)
                                
                                # Update current node and add to stack
                                current_node = new_node
                                current_level = level
                                node_stack.append(new_node)
                                
                            except Exception as e:
                                logger.warning(f"Error adjusting tree structure for heading '{line}': {e}")
                                # Fallback: add to root
                                root["children"].append(new_node)
                                current_node = new_node
                                current_level = level
                                node_stack = [root, new_node]
                            
                            is_heading = True
                            break
                            
                    except Exception as e:
                        logger.warning(f"Error processing line '{line}' with pattern: {e}")
                        continue
                
                if not is_heading:
                    current_content.append(line)
            
            # Add remaining content to current node
            if current_content:
                try:
                    current_node["content"] += '\n'.join(current_content) + '\n'
                except Exception as e:
                    logger.warning(f"Error adding content to node: {e}")
                    # Fallback: add content to root
                    root["content"] += '\n'.join(current_content) + '\n'
        
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
        
        # Validate input
        if not parsed_doc or not hasattr(parsed_doc, 'text_content'):
            logger.error("Invalid parsed_doc provided to _paragraph_chunking")
            return chunks
        
        text = parsed_doc.text_content
        if not text or not text.strip():
            logger.warning("Empty text content in _paragraph_chunking")
            return chunks
        
        try:
            # Split into paragraphs with error handling
            paragraphs = re.split(r'\n\s*\n', text)
            if not paragraphs:
                # If regex split fails, create a single paragraph
                paragraphs = [text]
        except Exception as e:
            logger.warning(f"Error splitting text into paragraphs: {e}")
            # Fallback: treat entire text as one paragraph
            paragraphs = [text]
        
        current_chunk_text = ""
        current_chunk_tokens = 0
        chunk_index = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            try:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Estimate token count (rough approximation)
                try:
                    para_tokens = len(paragraph.split())
                except Exception as token_error:
                    logger.warning(f"Error counting tokens in paragraph {para_idx}: {token_error}")
                    para_tokens = 50  # Default estimate
                
                # Skip empty, very short, or header-like paragraphs with safety checks
                try:
                    if para_tokens < 5 and not re.match(r'^[A-Z].*[\.!?]$', paragraph):
                        continue
                except Exception as regex_error:
                    logger.warning(f"Error in regex check for paragraph {para_idx}: {regex_error}")
                    # If regex fails, just check token count
                    if para_tokens < 5:
                        continue
                
                # If adding this paragraph would exceed max size, create a new chunk
                if current_chunk_tokens + para_tokens > self.max_chunk_size_tokens and current_chunk_text:
                    try:
                        chunk = TextChunk(
                            chunk_id=f"{parsed_doc.document_id}_chunk_{chunk_index}",
                            document_id=parsed_doc.document_id,
                            text=current_chunk_text.strip(),
                            metadata={
                                "source": getattr(parsed_doc, 'source_path', 'unknown'),
                                "chunk_strategy": ChunkingStrategy.PARAGRAPH
                            }
                        )
                        chunks.append(chunk)
                    except Exception as chunk_creation_error:
                        logger.error(f"Error creating chunk {chunk_index}: {chunk_creation_error}")
                        # Continue anyway to avoid losing all chunks
                    
                    chunk_index += 1
                    current_chunk_text = paragraph + "\n\n"
                    current_chunk_tokens = para_tokens
                else:
                    current_chunk_text += paragraph + "\n\n"
                    current_chunk_tokens += para_tokens
                    
            except Exception as para_error:
                logger.warning(f"Error processing paragraph {para_idx}: {para_error}")
                continue
        
        # Add the last chunk if not empty
        if current_chunk_text.strip():
            try:
                chunk = TextChunk(
                    chunk_id=f"{parsed_doc.document_id}_chunk_{chunk_index}",
                    document_id=parsed_doc.document_id,
                    text=current_chunk_text.strip(),
                    metadata={
                        "source": getattr(parsed_doc, 'source_path', 'unknown'),
                        "chunk_strategy": ChunkingStrategy.PARAGRAPH
                    }
                )
                chunks.append(chunk)
            except Exception as final_chunk_error:
                logger.error(f"Error creating final chunk: {final_chunk_error}")
        
        # If no chunks were created, create one emergency chunk
        if not chunks and text.strip():
            try:
                logger.warning("No chunks created in paragraph chunking, creating emergency chunk")
                emergency_chunk = TextChunk(
                    chunk_id=f"{parsed_doc.document_id}_emergency_paragraph_0",
                    document_id=parsed_doc.document_id,
                    text=text[:self.max_chunk_size_tokens * 4],  # Rough token limit
                    metadata={
                        "source": getattr(parsed_doc, 'source_path', 'unknown'),
                        "chunk_strategy": "emergency_paragraph",
                        "note": "Paragraph chunking failed, created emergency chunk"
                    }
                )
                chunks.append(emergency_chunk)
            except Exception as emergency_error:
                logger.error(f"Emergency chunk creation failed: {emergency_error}")
        
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