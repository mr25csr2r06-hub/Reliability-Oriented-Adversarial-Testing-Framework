# Customization Guide - Reducing Experiment Size

This guide explains how to modify the experiment configuration to reduce the number of tasks and executions for testing or quicker iterations.

## Quick Reference

To reduce experiment size, modify these key parameters:

| Parameter | File | Default | Reduced | Impact |
|-----------|------|---------|---------|--------|
| REPETITIONS | config/experiment_config.py | 5 | 2-3 | Reduces total executions by 40-60% |
| PROMPT_INJECTION_SAMPLES | config/experiment_config.py | 116 | 10-20 | Reduces seeds from this category |
| JAILBREAK_SAMPLES | config/experiment_config.py | 200 | 10-20 | Reduces seeds from this category |
| INDIRECT_INJECTION_SAMPLES | config/experiment_config.py | 100 | 5-10 | Reduces seeds from this category |
| BENIGN_SAMPLES | config/experiment_config.py | 100 | 5-10 | Reduces seeds from this category |
| Model count | config/models_config.py | 6 | 1-3 | Reduces executions proportionally |

## Configuration Files

### 1. experiment_config.py

**Location:** `config/experiment_config.py`

**Key Parameters:**

```python
# Generation parameters
REPETITIONS = 5  # Reduce to 2-3 for quicker testing

# Dataset sizes (primary experiment)
PROMPT_INJECTION_SAMPLES = 116  # Reduce to 10-20
JAILBREAK_SAMPLES = 200  # Reduce to 10-20
INDIRECT_INJECTION_SAMPLES = 100  # Reduce to 5-10
BENIGN_SAMPLES = 100  # Reduce to 5-10

# Dataset sizes (development experiment)
DEV_PROMPT_INJECTION_SAMPLES = 5  # Keep small for testing
DEV_JAILBREAK_SAMPLES = 5
DEV_INDIRECT_INJECTION_SAMPLES = 5
DEV_BENIGN_SAMPLES = 5
```

**Impact Calculation:**
- Default: 516 seeds × 5 mutations × 5 repetitions × 6 models = 77,400 executions
- Reduced (10 samples each, 3 repetitions, 3 models): 40 seeds × 5 mutations × 3 repetitions × 3 models = 1,800 executions

### 2. models_config.py

**Location:** `config/models_config.py`

**Key Parameters:**

```python
MODELS_CONFIG = {
    "gemma": {
        "enabled": True  # Set to False to disable
    },
    "llama31": {
        "enabled": True  # Set to False to disable
    },
    "qwen": {
        "enabled": False  # Disable to reduce executions
    },
    "phi": {
        "enabled": False  # Disable to reduce executions
    },
    "mistral": {
        "enabled": False  # Disable to reduce executions
    },
    "deepseek": {
        "enabled": False  # Disable to reduce executions
    }
}
```

**Impact Calculation:**
- 6 models enabled: 77,400 executions
- 3 models enabled: 38,700 executions (50% reduction)
- 1 model enabled: 12,900 executions (83% reduction)

### 3. datasets_config.py

**Location:** `config/datasets_config.py`

**Key Parameters:**

```python
DATASETS_CONFIG = {
    "prompt_injection": {
        "target_samples": 116  # Reduce to 10-20
    },
    "jailbreak": {
        "target_samples": 200  # Reduce to 10-20
    },
    "indirect_injection": {
        "target_samples": 100  # Reduce to 5-10
    },
    "benign": {
        "target_samples": 100  # Reduce to 5-10
    }
}
```

**Impact Calculation:**
- All datasets (516 total): 77,400 executions
- Only prompt_injection and jailbreak (316 total): 47,400 executions
- Only jailbreak (200 total): 30,000 executions

### 4. mutation_config.py

**Location:** `config/mutation_config.py`

**Key Parameters:**

```python
MUTATION_CONFIG = {
    "original": {
        "enabled": True
    },
    "semantic": {
        "enabled": True
    },
    "contextual": {
        "enabled": False  # Disable to reduce mutations
    },
    "paraphrase": {
        "enabled": False  # Disable to reduce mutations
    },
    "structural": {
        "enabled": False  # Disable to reduce mutations
    }
}
```

**Impact Calculation:**
- 5 mutations: 77,400 executions
- 3 mutations: 46,440 executions (40% reduction)
- 2 mutations: 30,960 executions (60% reduction)

## Recommended Configurations

### Ultra-Fast Testing (Minutes)

**Purpose:** Quick framework verification

```python
# config/experiment_config.py
REPETITIONS = 2
PROMPT_INJECTION_SAMPLES = 2
JAILBREAK_SAMPLES = 2
INDIRECT_INJECTION_SAMPLES = 1
BENIGN_SAMPLES = 1

# config/models_config.py
Enable only: gemma

# config/mutation_config.py
Enable only: original, semantic
```

**Total executions:** 12 × 2 × 2 × 1 = 48 executions

### Quick Development (Hours)

**Purpose:** Development and debugging

```python
# config/experiment_config.py
REPETITIONS = 3
PROMPT_INJECTION_SAMPLES = 5
JAILBREAK_SAMPLES = 5
INDIRECT_INJECTION_SAMPLES = 3
BENIGN_SAMPLES = 3

# config/models_config.py
Enable only: gemma, llama31

# config/mutation_config.py
Enable all mutations
```

**Total executions:** 16 × 5 × 3 × 2 = 480 executions

### Medium Experiment (Days)

**Purpose:** Preliminary results

```python
# config/experiment_config.py
REPETITIONS = 3
PROMPT_INJECTION_SAMPLES = 20
JAILBREAK_SAMPLES = 30
INDIRECT_INJECTION_SAMPLES = 15
BENIGN_SAMPLES = 15

# config/models_config.py
Enable: gemma, llama31, qwen

# config/mutation_config.py
Enable all mutations
```

**Total executions:** 80 × 5 × 3 × 3 = 3,600 executions

### Production Experiment (Weeks)

**Purpose:** Full research results

```python
# Use default configurations
# config/experiment_config.py
REPETITIONS = 5
PROMPT_INJECTION_SAMPLES = 116
JAILBREAK_SAMPLES = 200
INDIRECT_INJECTION_SAMPLES = 100
BENIGN_SAMPLES = 100

# config/models_config.py
Enable all 6 models

# config/mutation_config.py
Enable all mutations
```

**Total executions:** 516 × 5 × 5 × 6 = 77,400 executions

## Step-by-Step Reduction Process

### Step 1: Determine Your Goal

Ask yourself:
- How much time do you have?
- How much GPU memory is available?
- What's the minimum sample size for statistical significance?
- Which models are most important for your research?

### Step 2: Calculate Target Executions

Use this formula:
```
Total Executions = Total Seeds × Mutations × Repetitions × Models
```

### Step 3: Modify Configuration Files

1. **Start with repetitions** - easiest to change
2. **Then adjust dataset sizes** - maintains dataset balance
3. **Finally adjust model count** - biggest impact

### Step 4: Test with Development Mode

Always test with development mode first:
```bash
python main.py --development
```

### Step 5: Verify Expected Executions

The framework will display expected executions before running. Verify this matches your calculation.

## Important Considerations

### Statistical Significance

- **Minimum samples**: For statistical tests, aim for at least 30 samples per group
- **Balanced datasets**: Maintain proportional representation across categories
- **Repetitions**: At least 3 repetitions are recommended for reliability metrics

### Resource Requirements

**Approximate requirements per 1,000 executions:**
- **GPU time**: 2-4 hours (depending on model)
- **Disk space**: 100-200 MB (raw responses)
- **Memory**: 4-8 GB RAM

### Model Selection Strategy

If reducing models:
1. **Keep diverse architectures**: Different model families (Gemma vs Llama vs Mistral)
2. **Keep different sizes**: Mix of small and large models
3. **Consider research focus**: If focusing on specific capabilities, prioritize relevant models

## Restoration to Full Experiment

To restore the full experiment:

1. **Backup your modified config files**
2. **Restore from original defaults** or
3. **Reset each parameter to default values**
4. **Verify with development mode first**

## Validation

After making changes:

1. **Run:** `python main.py`
2. **Select:** Option 6 (Run primary experiment)
3. **Check:** The expected executions display
4. **Cancel:** When asked for confirmation (unless you want to proceed)
5. **Verify:** The number matches your calculation

## Common Mistakes to Avoid

1. **Setting repetitions to 1**: Invalid for reliability metrics
2. **Using only 1 mutation type**: Cannot calculate PSV
3. **Disabling all datasets**: Experiment will fail
4. **Unbalanced datasets**: Skews statistical analysis
5. **Forgetting to save config files**: Changes won't take effect

## Quick Modification Template

Copy this template for quick modifications:

```python
# ===== CUSTOM CONFIGURATION =====
# File: config/experiment_config.py

# Quick testing (48 executions)
QUICK_TEST = {
    "REPETITIONS": 2,
    "PROMPT_INJECTION_SAMPLES": 2,
    "JAILBREAK_SAMPLES": 2,
    "INDIRECT_INJECTION_SAMPLES": 1,
    "BENIGN_SAMPLES": 1
}

# Development (480 executions)
DEVELOPMENT = {
    "REPETITIONS": 3,
    "PROMPT_INJECTION_SAMPLES": 5,
    "JAILBREAK_SAMPLES": 5,
    "INDIRECT_INJECTION_SAMPLES": 3,
    "BENIGN_SAMPLES": 3
}

# Medium (3,600 executions)
MEDIUM = {
    "REPETITIONS": 3,
    "PROMPT_INJECTION_SAMPLES": 20,
    "JAILBREAK_SAMPLES": 30,
    "INDIRECT_INJECTION_SAMPLES": 15,
    "BENIGN_SAMPLES": 15
}

# Apply your chosen configuration
# Uncomment the one you want to use:
# REPETITIONS = QUICK_TEST["REPETITIONS"]
# ... and so on
```

## Contact and Support

If you encounter issues with configuration:
1. Check the logs in `logs/system/system.log`
2. Verify your changes were saved
3. Test with development mode first
4. Ensure Python syntax is correct in config files