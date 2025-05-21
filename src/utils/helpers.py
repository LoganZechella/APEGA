"""
Utility functions for APEGA.
Contains various utility functions used across the system.
"""

import os
import re
import json
import hashlib
from typing import Dict, Any, List, Optional, Union
from loguru import logger


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to ensure it's valid on all operating systems.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
    # Replace multiple underscores with a single one
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Ensure filename is not empty
    if not sanitized:
        sanitized = "file"
    return sanitized


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate MD5 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash of the file
    """
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {str(e)}")
        return ""


def ensure_dir_exists(directory: str) -> None:
    """
    Ensure a directory exists, creating it if needed.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)


def save_json(data: Any, file_path: str, indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to the JSON file
        indent: Indentation level
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Save data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
        
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {str(e)}")
        return False


def load_json(file_path: str) -> Optional[Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {str(e)}")
        return None


def count_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text.
    This is a simple approximation based on whitespace splitting.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Simple approximation: split by whitespace and count
    # In practice, this should use a proper tokenizer for the specific model
    return len(text.split())


def truncate_text(text: str, max_tokens: int) -> str:
    """
    Truncate text to a maximum number of tokens.
    
    Args:
        text: Input text
        max_tokens: Maximum number of tokens
        
    Returns:
        Truncated text
    """
    words = text.split()
    if len(words) <= max_tokens:
        return text
    
    return ' '.join(words[:max_tokens]) + "..."