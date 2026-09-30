"""
Mutation configuration for Reliability-Oriented LLM Testing Framework.
"""

# Mutation types and their parameters
MUTATION_CONFIG = {
    "original": {
        "type": "original",
        "description": "Original prompt without mutation",
        "enabled": True
    },
    "semantic": {
        "type": "semantic",
        "description": "Semantic mutation while preserving meaning",
        "enabled": True,
        "similarity_threshold": 0.7,
        "max_length_ratio": 1.5
    },
    "contextual": {
        "type": "contextual",
        "description": "Contextual perturbation",
        "enabled": True,
        "context_types": ["neutral", "positive", "negative"]
    },
    "paraphrase": {
        "type": "paraphrase",
        "description": "Paraphrase transformation",
        "enabled": True,
        "similarity_threshold": 0.8
    },
    "structural": {
        "type": "structural",
        "description": "Structural mutation",
        "enabled": True,
        "structure_types": ["reorder", "split", "merge"]
    }
}

# Validation parameters
MUTATION_VALIDATION = {
    "min_similarity": 0.5,
    "max_similarity": 0.95,
    "min_length": 10,
    "max_length": 1000,
    "check_duplicates": True,
    "check_malformed": True
}