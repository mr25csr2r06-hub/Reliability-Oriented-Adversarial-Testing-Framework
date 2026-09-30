"""
Environment information capture for Reliability-Oriented LLM Testing Framework.
"""

import platform
import os
import sys
import torch
import transformers
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime


def get_environment_info() -> Dict[str, Any]:
    """
    Capture comprehensive environment information.
    
    Returns:
        Dictionary containing environment details
    """
    env_info = {
        "timestamp": datetime.now().isoformat(),
        "python": {
            "version": sys.version,
            "version_major": sys.version_info.major,
            "version_minor": sys.version_info.minor,
            "version_patch": sys.version_info.micro,
            "executable": sys.executable
        },
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        },
        "hardware": {
            "cpu_count": os.cpu_count(),
            "processor": platform.processor()
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
        },
        "transformers": {
            "version": transformers.__version__
        },
        "gpu": {}
    }
    
    # Add GPU information if available
    if torch.cuda.is_available():
        env_info["gpu"] = {
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
            "current_device": torch.cuda.current_device() if torch.cuda.device_count() > 0 else None
        }
        
        # Get GPU memory info
        if torch.cuda.device_count() > 0:
            try:
                gpu_memory = torch.cuda.get_device_properties(0)
                env_info["gpu"]["total_memory_mb"] = gpu_memory.total_memory / (1024 * 1024)
                env_info["gpu"]["allocated_memory_mb"] = torch.cuda.memory_allocated(0) / (1024 * 1024)
                env_info["gpu"]["reserved_memory_mb"] = torch.cuda.memory_reserved(0) / (1024 * 1024)
            except Exception as e:
                env_info["gpu"]["memory_error"] = str(e)
    
    return env_info


def save_environment_info(output_dir: str = "results/manifests") -> str:
    """
    Save environment information to JSON file.
    
    Args:
        output_dir: Directory to save environment info
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    env_info = get_environment_info()
    env_file = output_path / "environment.json"
    
    with open(env_file, 'w', encoding='utf-8') as f:
        json.dump(env_info, f, indent=2, default=str)
    
    return str(env_file)


def check_python_version(min_version: tuple = (3, 8)) -> bool:
    """
    Check if Python version meets minimum requirements.
    
    Args:
        min_version: Minimum required version as tuple (major, minor)
        
    Returns:
        True if version is sufficient, False otherwise
    """
    current_version = (sys.version_info.major, sys.version_info.minor)
    return current_version >= min_version


def check_cuda_availability() -> bool:
    """
    Check if CUDA is available.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    return torch.cuda.is_available()


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get current GPU memory information.
    
    Returns:
        Dictionary with memory information in MB
    """
    if not torch.cuda.is_available():
        return {}
    
    memory_info = {}
    try:
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
            memory_info[f"gpu_{i}"] = {
                "allocated_mb": allocated,
                "reserved_mb": reserved
            }
    except Exception as e:
        memory_info["error"] = str(e)
    
    return memory_info