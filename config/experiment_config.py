"""
Experiment configuration for Reliability-Oriented LLM Testing Framework.
"""

# Experiment identification
EXPERIMENT_NAME = "Reliability-Oriented LLM Testing"
EXPERIMENT_VERSION = "1.0.0"

# Generation parameters
TEMPERATURE = 0.7
TOP_P = 1.0
MAX_NEW_TOKENS = 512
BASE_SEED = 2026
REPETITIONS = 5

# Development mode configuration
DEVELOPMENT_MODE = False

# Dataset sizes
PROMPT_INJECTION_SAMPLES = 116
JAILBREAK_SAMPLES = 200
INDIRECT_INJECTION_SAMPLES = 100
BENIGN_SAMPLES = 100

# Development dataset sizes (for testing)
DEV_PROMPT_INJECTION_SAMPLES = 5
DEV_JAILBREAK_SAMPLES = 5
DEV_INDIRECT_INJECTION_SAMPLES = 5
DEV_BENIGN_SAMPLES = 5

# Primary mutation conditions
MUTATIONS = [
    "original",
    "semantic",
    "contextual",
    "paraphrase",
    "structural"
]

# Statistical analysis parameters
ALPHA = 0.05
CONFIDENCE_LEVEL = 0.95

# File paths
DATA_DIR = "data"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
CACHE_DIR = "data/cache"
MANIFESTS_DIR = "data/manifests"

LOGS_DIR = "logs"
SYSTEM_LOG_DIR = "logs/system"
DATASET_LOG_DIR = "logs/dataset"
MUTATION_LOG_DIR = "logs/mutation"
EXECUTION_LOG_DIR = "logs/execution"
EVALUATION_LOG_DIR = "logs/evaluation"
METRICS_LOG_DIR = "logs/metrics"
ERRORS_LOG_DIR = "logs/errors"

RESULTS_DIR = "results"
RAW_RESULTS_DIR = "results/raw"
CLASSIFIED_RESULTS_DIR = "results/classified"
METRICS_RESULTS_DIR = "results/metrics"
STATISTICS_RESULTS_DIR = "results/statistics"
ABLATION_RESULTS_DIR = "results/ablation"
OVERHEAD_RESULTS_DIR = "results/overhead"
RESULTS_MANIFESTS_DIR = "results/manifests"
REPORTS_DIR = "results/reports"

PLOTS_DIR = "plots"
PRIMARY_PLOTS_DIR = "plots/primary"
ABLATION_PLOTS_DIR = "plots/ablation"
EXPLORATORY_PLOTS_DIR = "plots/exploratory"

TABLES_DIR = "tables"