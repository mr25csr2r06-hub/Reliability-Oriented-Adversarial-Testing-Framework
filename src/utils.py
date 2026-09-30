"""
Utility functions for Reliability-Oriented LLM Testing Framework.
"""

import json
import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime


def generate_execution_id(experiment_id: str, model_name: str, 
                         sample_id: str, variant_id: str, 
                         execution_number: int) -> str:
    """
    Generate unique execution ID.
    
    Args:
        experiment_id: Experiment identifier
        model_name: Model name
        sample_id: Sample identifier
        variant_id: Variant identifier
        execution_number: Execution repetition number
        
    Returns:
        Unique execution ID
    """
    return f"{experiment_id}_{model_name}_{sample_id}_{variant_id}_R{execution_number:03d}"


def generate_sample_id(category: str, index: int) -> str:
    """
    Generate sample ID.
    
    Args:
        category: Sample category
        index: Sample index
        
    Returns:
        Sample ID
    """
    category_prefix = {
        "prompt_injection": "PI",
        "jailbreak": "JB",
        "indirect_injection": "II",
        "benign": "BI"
    }
    prefix = category_prefix.get(category, "UNK")
    return f"{prefix}{index:04d}"


def generate_variant_id(sample_id: str, mutation_type: str, 
                       variant_number: int) -> str:
    """
    Generate variant ID.
    
    Args:
        sample_id: Parent sample ID
        mutation_type: Type of mutation
        variant_number: Variant number
        
    Returns:
        Variant ID
    """
    mutation_prefix = {
        "original": "V000",
        "semantic": "V001",
        "contextual": "V002",
        "paraphrase": "V003",
        "structural": "V004"
    }
    prefix = mutation_prefix.get(mutation_type, f"V{variant_number:03d}")
    return f"{sample_id}_{prefix}"


def generate_experiment_id() -> str:
    """
    Generate unique experiment ID.
    
    Returns:
        Experiment ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = str(uuid.uuid4())[:8]
    return f"EXP_{timestamp}_{random_suffix}"


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_hash(text: str) -> str:
    """
    Calculate SHA-256 hash of text.
    
    Args:
        text: Input text
        
    Returns:
        Hash string
    """
    return hashlib.sha256(text.encode()).hexdigest()


def ensure_directory(directory: str) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_memory_size(bytes_size: int) -> str:
    """
    Format memory size in human-readable format.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted memory string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f}PB"