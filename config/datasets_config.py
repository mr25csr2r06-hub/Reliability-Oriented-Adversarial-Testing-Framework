"""
Datasets configuration for Reliability-Oriented LLM Testing Framework.
"""

# Dataset configurations
DATASETS_CONFIG = {
    "prompt_injection": {
        "dataset_name": "deepset/prompt-injections",
        "split": "train",
        "target_samples": 116,
        "category": "prompt_injection",
        "huggingface_id": "deepset/prompt-injections"
    },
    "jailbreak": {
        "dataset_name": "deepset/prompt-injections",
        "split": "train",
        "target_samples": 200,
        "category": "jailbreak",
        "huggingface_id": "deepset/prompt-injections"  # Use same dataset but filter for jailbreak
    },
    "indirect_injection": {
        "dataset_name": "deepset/prompt-injections",
        "split": "train",
        "target_samples": 100,
        "category": "indirect_injection",
        "huggingface_id": "deepset/prompt-injections"  # Use same dataset but filter for indirect
    },
    "benign": {
        "dataset_name": "deepset/prompt-injections",
        "split": "train",
        "target_samples": 100,
        "category": "benign",
        "huggingface_id": "deepset/prompt-injections"  # Use same dataset but filter for benign
    }
}

# Development dataset configurations
DEV_DATASETS_CONFIG = {
    "prompt_injection": {
        "dataset_name": "deepset/prompt-injections",
        "split": "train",
        "target_samples": 3,
        "category": "prompt_injection",
        "huggingface_id": "deepset/prompt-injections"
    },
    "jailbreak": {
        "dataset_name": "deepset/prompt-injections",
        "split": "train",
        "target_samples": 3,
        "category": "jailbreak",
        "huggingface_id": "deepset/prompt-injections"
    },
    "indirect_injection": {
        "dataset_name": "deepset/prompt-injections",
        "split": "train",
        "target_samples": 3,
        "category": "indirect_injection",
        "huggingface_id": "deepset/prompt-injections"
    },
    "benign": {
        "dataset_name": "deepset/prompt-injections",
        "split": "train",
        "target_samples": 3,
        "category": "benign",
        "huggingface_id": "deepset/prompt-injections"
    }
}

# Task types for benign dataset
BENIGN_TASK_TYPES = [
    "QA",
    "email",
    "summarization",
    "table_data",
    "coding"
]

# Task types for BIPIA dataset
BIPIA_TASK_TYPES = [
    "Email QA",
    "Web QA",
    "Table QA",
    "Summarization",
    "Code QA"
]