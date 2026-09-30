# Reliability-Oriented Adversarial Testing Framework

A Python framework for evaluating the reliability of LLM-based software systems under adversarial prompt variants. The project downloads and prepares datasets, generates prompt mutations, executes enabled Hugging Face models, calculates reliability metrics, performs statistical and overhead analysis, and writes tables, plots, logs, and reports.

## Features

- Tests six configured LLM families: Gemma, Llama 3.1, Qwen, Phi, Mistral, and DeepSeek
- Builds four dataset categories: prompt injection, jailbreak, indirect injection, and benign
- Generates five prompt conditions: original, semantic, contextual, paraphrase, and structural
- Calculates reliability metrics including ASR, RR, PSV, FRS, JPR, and SRD
- Produces CSV tables, PNG/SVG plots, JSON reports, and human-readable reports
- Captures environment details and execution logs for reproducibility

## Requirements

- Python 3.8 or higher
- A CUDA-capable GPU is strongly recommended for the configured 7B-class models
- Hugging Face access token for gated models such as Llama 3.1
- Sufficient disk space for model and dataset caches

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then add your Hugging Face token:

```text
HF_TOKEN=your_huggingface_token_here
```

## Shell Command File

The project includes [run_project.sh](run_project.sh) with the main setup and run commands.

```bash
bash run_project.sh setup
bash run_project.sh phase
bash run_project.sh phase1
bash run_project.sh all
```

- `setup` creates `.venv`, installs `requirements.txt`, and creates `.env` from `.env.example` if needed.
- `phase` activates `.venv` and starts the interactive phase selector in `main.py`.
- `phase1` activates `.venv` and runs the quick Phase 1 helper through `auto_run.py`.
- `all` performs setup and then starts the interactive phase selector.

## Running The Framework

Start the main program:

```bash
python main.py
```

The program shows a phase selector:

```text
1. Phase 1: Quick Test (3 tasks per dataset)
2. Phase 2: Full Experiment (all tasks per dataset)
3. Exit
```

## Phase Selection

Phase selection controls the experiment size.

### Phase 1: Quick Test

Phase 1 is the recommended first run. It uses development-mode dataset settings and limits each dataset category to three prepared samples. After prompt variants are generated, the execution step selects representative variants from the available categories and runs them across all enabled models with five repetitions.

Use Phase 1 to verify that:

- Dependencies are installed correctly
- The Hugging Face token is available
- Datasets can be downloaded and prepared
- Models can be loaded sequentially
- Logs, tables, plots, and reports are generated correctly

Because it is intentionally small, Phase 1 is best for environment validation and debugging before a long experiment.

### Phase 2: Full Experiment

Phase 2 uses the full dataset configuration from `config/datasets_config.py`. It generates all configured mutation variants for all prepared samples, then executes every variant across all enabled models with five repetitions.

Use Phase 2 only after Phase 1 succeeds. It can require substantial GPU time, memory, model download time, and disk space.

## Project Structure

```text
.
+-- auto_run.py                  # Helper script for quick Phase 1 execution
+-- main.py                      # Main framework and phase selector
+-- run_phase1.py                # Legacy Phase 1 helper
+-- run_project.sh               # Setup and run command script
+-- requirements.txt             # Project dependencies
+-- config/                      # Experiment, model, dataset, mutation config
+-- llms/                        # Hugging Face model wrapper classes
+-- src/                         # Dataset, metrics, statistics, overhead, utilities
+-- data/                        # Raw, processed, cached data and manifests
+-- logs/                        # Runtime logs
+-- results/                     # Raw executions, metrics, reports, manifests
+-- plots/                       # Generated figures
+-- tables/                      # Generated CSV tables
```

## Experiment Flow

Each phase follows the same high-level workflow:

1. Check Python and CUDA availability
2. Create required output directories
3. Save environment metadata
4. Download datasets
5. Prepare and validate datasets
6. Generate prompt variants
7. Execute enabled LLMs sequentially
8. Calculate reliability metrics
9. Run statistical, ablation, and overhead analysis
10. Generate CSV tables, plots, and reports

## Configuration

Main configuration files:

- `config/experiment_config.py`: generation parameters, repetitions, mutation list, output paths
- `config/datasets_config.py`: dataset names, splits, categories, target sample sizes
- `config/models_config.py`: model IDs, enabled flags, auth requirements, device settings

To reduce the run size, disable models in `config/models_config.py` or reduce target samples in `config/datasets_config.py`.

## Outputs

- `results/raw/`: execution records
- `results/metrics/`: reliability metric JSON
- `results/statistics/`: statistical analysis JSON
- `results/ablation/`: ablation analysis output
- `results/overhead/`: overhead analysis output
- `results/reports/`: final reports
- `tables/`: CSV tables
- `plots/`: PNG/SVG figures
- `logs/`: system, dataset, execution, metrics, and error logs

## Troubleshooting

### CUDA Out Of Memory

Disable some models in `config/models_config.py`, run Phase 1 first, and ensure no other GPU-heavy process is running.

### Hugging Face Authentication Errors

Check `.env`, verify `HF_TOKEN` is set, and confirm your Hugging Face account has access to gated models.

### Dataset Download Errors

Check internet access, the Hugging Face dataset ID, and available disk space in `data/cache/`.

### Long Runtime

Run Phase 1 first. Phase 2 runs every generated variant across all enabled models and five repetitions, so runtime grows quickly with dataset size and model count.

## Notes

- The framework loads models sequentially to reduce GPU memory pressure.
- Raw execution responses are saved for traceability.
- Generated artifacts are written under `results/`, `tables/`, `plots/`, and `logs/`.
- This framework is intended for research and evaluation workflows.
