"""
Models configuration for Reliability-Oriented LLM Testing Framework.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Model configurations
MODELS_CONFIG = {
    "gemma": {
        "model_name": "gemma",
        "huggingface_id": "google/gemma-7b",
        "revision": "main",
        "enabled": True,
        "dtype": "float16",
        "device": "cuda",
        "requires_auth": False
    },
    "llama31": {
        "model_name": "llama31",
        "huggingface_id": "meta-llama/Llama-3.1-8B",
        "revision": "main",
        "enabled": True,
        "dtype": "float16",
        "device": "cuda",
        "requires_auth": True
    },
    "qwen": {
        "model_name": "qwen",
        "huggingface_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "main",
        "enabled": True,
        "dtype": "float16",
        "device": "cuda",
        "requires_auth": False
    },
    "phi": {
        "model_name": "phi",
        "huggingface_id": "microsoft/phi-2",
        "revision": "main",
        "enabled": True,
        "dtype": "float16",
        "device": "cuda",
        "requires_auth": False
    },
    "mistral": {
        "model_name": "mistral",
        "huggingface_id": "mistralai/Mistral-7B-v0.1",
        "revision": "main",
        "enabled": True,
        "dtype": "float16",
        "device": "cuda",
        "requires_auth": False
    },
    "deepseek": {
        "model_name": "deepseek",
        "huggingface_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "revision": "main",
        "enabled": True,
        "dtype": "float16",
        "device": "cuda",
        "requires_auth": False
    }
}

# Get enabled models
def get_enabled_models():
    """Return list of enabled model names."""
    return [name for name, config in MODELS_CONFIG.items() if config["enabled"]]

# Get HF token
def get_hf_token():
    """Get Hugging Face token from environment."""
    return os.getenv("HF_TOKEN")

# Model list
MODELS = get_enabled_models()