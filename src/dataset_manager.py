"""
Dataset manager for Reliability-Oriented LLM Testing Framework.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import load_dataset
import hashlib
from datetime import datetime

from src.logger import get_logger
from src.utils import save_json, load_json, generate_sample_id, ensure_directory
import config.datasets_config as datasets_config
import config.experiment_config as config


class DatasetManager:
    """Manage dataset download, processing, and validation."""
    
    def __init__(self, development_mode: bool = False):
        """
        Initialize dataset manager.
        
        Args:
            development_mode: If True, use development dataset sizes
        """
        self.logger = get_logger('dataset')
        self.development_mode = development_mode
        self.datasets_config = datasets_config.DEV_DATASETS_CONFIG if development_mode else datasets_config.DATASETS_CONFIG
        
    def download_datasets(self) -> Dict[str, Any]:
        """
        Download all configured datasets.
        
        Returns:
            Dictionary with download results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {},
            'total_samples': 0,
            'success': True,
            'errors': []
        }
        
        self.logger.info("Starting dataset download...")
        
        for dataset_name, dataset_config in self.datasets_config.items():
            try:
                self.logger.info(f"Downloading dataset: {dataset_name}")
                
                if dataset_config['huggingface_id'] == 'custom':
                    # Custom dataset (benign) - will be generated later
                    self.logger.info(f"Skipping custom dataset: {dataset_name}")
                    results['datasets'][dataset_name] = {
                        'status': 'skipped',
                        'reason': 'custom dataset - will be generated',
                        'samples': 0
                    }
                    continue
                
                # Check if dataset already downloaded
                raw_file = Path(config.RAW_DATA_DIR) / f"{dataset_name}_raw.jsonl"
                if raw_file.exists():
                    self.logger.info(f"Dataset {dataset_name} already downloaded, skipping")
                    # Count existing samples
                    sample_count = 0
                    with open(raw_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                sample_count += 1
                    
                    results['datasets'][dataset_name] = {
                        'status': 'cached',
                        'huggingface_id': dataset_config['huggingface_id'],
                        'samples': sample_count,
                        'target_samples': dataset_config['target_samples'],
                        'file': str(raw_file)
                    }
                    results['total_samples'] += sample_count
                    continue
                
                # Download dataset from Hugging Face
                dataset = load_dataset(
                    dataset_config['huggingface_id'],
                    split=dataset_config['split'],
                    cache_dir=config.CACHE_DIR
                )
                
                # Save raw dataset as JSONL (JSON Lines) format
                raw_file = Path(config.RAW_DATA_DIR) / f"{dataset_name}_raw.jsonl"
                with open(raw_file, 'w', encoding='utf-8') as f:
                    for item in dataset:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                results['datasets'][dataset_name] = {
                    'status': 'success',
                    'huggingface_id': dataset_config['huggingface_id'],
                    'samples': len(dataset),
                    'target_samples': dataset_config['target_samples'],
                    'file': str(raw_file)
                }
                
                results['total_samples'] += len(dataset)
                self.logger.info(f"Downloaded {dataset_name}: {len(dataset)} samples")
                
            except Exception as e:
                error_msg = f"Failed to download {dataset_name}: {str(e)}"
                self.logger.error(error_msg)
                results['datasets'][dataset_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                results['errors'].append(error_msg)
                results['success'] = False
        
        # Save download manifest
        manifest_file = Path(config.MANIFESTS_DIR) / "download_manifest.json"
        save_json(results, manifest_file)
        
        self.logger.info(f"Dataset download completed. Total samples: {results['total_samples']}")
        return results
    
    def prepare_datasets(self) -> Dict[str, Any]:
        """
        Prepare and normalize datasets to unified format.
        
        Returns:
            Dictionary with preparation results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {},
            'total_samples': 0,
            'success': True,
            'errors': []
        }
        
        self.logger.info("Starting dataset preparation...")
        
        for dataset_name, dataset_config in self.datasets_config.items():
            try:
                self.logger.info(f"Preparing dataset: {dataset_name}")
                
                if dataset_config['huggingface_id'] == 'custom':
                    # Skip custom datasets for now
                    results['datasets'][dataset_name] = {
                        'status': 'skipped',
                        'reason': 'custom dataset - will be generated separately',
                        'samples': 0
                    }
                    continue
                
                # Load raw dataset
                raw_file = Path(config.RAW_DATA_DIR) / f"{dataset_name}_raw.jsonl"
                if not raw_file.exists():
                    raise FileNotFoundError(f"Raw dataset file not found: {raw_file}")
                
                # Normalize to unified format
                normalized_samples = self._normalize_dataset(dataset_name, dataset_config)
                
                # Save processed dataset
                processed_file = Path(config.PROCESSED_DATA_DIR) / f"{dataset_name}_processed.json"
                save_json(normalized_samples, processed_file)
                
                results['datasets'][dataset_name] = {
                    'status': 'success',
                    'samples': len(normalized_samples),
                    'target_samples': dataset_config['target_samples'],
                    'file': str(processed_file)
                }
                
                results['total_samples'] += len(normalized_samples)
                self.logger.info(f"Prepared {dataset_name}: {len(normalized_samples)} samples")
                
            except Exception as e:
                error_msg = f"Failed to prepare {dataset_name}: {str(e)}"
                self.logger.error(error_msg)
                results['datasets'][dataset_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                results['errors'].append(error_msg)
                results['success'] = False
        
        # Save preparation manifest
        manifest_file = Path(config.MANIFESTS_DIR) / "preparation_manifest.json"
        save_json(results, manifest_file)
        
        self.logger.info(f"Dataset preparation completed. Total samples: {results['total_samples']}")
        return results
    
    def _normalize_dataset(self, dataset_name: str, dataset_config: Dict) -> List[Dict[str, Any]]:
        """
        Normalize dataset to unified format.
        
        Args:
            dataset_name: Name of the dataset
            dataset_config: Dataset configuration
            
        Returns:
            List of normalized samples
        """
        # Load raw dataset (JSONL format)
        raw_file = Path(config.RAW_DATA_DIR) / f"{dataset_name}_raw.jsonl"
        
        # Read JSONL file
        raw_data = []
        try:
            with open(raw_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        raw_data.append(json.loads(line))
        except Exception as e:
            self.logger.error(f"Error reading {raw_file}: {e}")
            return []
        
        # Normalize based on dataset type
        normalized_samples = []
        
        for idx, sample in enumerate(raw_data[:dataset_config['target_samples']]):
            sample_id = generate_sample_id(dataset_config['category'], idx)
            
            # Extract prompt text (handle different field names)
            prompt_text = ""
            if isinstance(sample, dict):
                for field in ['text', 'prompt', 'question', 'input']:
                    if field in sample:
                        prompt_text = str(sample[field])
                        break
            
            normalized_sample = {
                "sample_id": sample_id,
                "source_dataset": dataset_config['huggingface_id'],
                "category": dataset_config['category'],
                "task_type": self._extract_task_type(sample, dataset_name),
                "original_prompt": prompt_text,
                "context": self._extract_context(sample, dataset_name),
                "metadata": {
                    "original_index": idx,
                    "dataset_name": dataset_name,
                    "hash": self._calculate_sample_hash(sample),
                    "label": sample.get('label', None) if isinstance(sample, dict) else None
                }
            }
            
            # Preserve BIPIA structure if applicable
            if dataset_name == "indirect_injection":
                self._preserve_bipia_structure(normalized_sample, sample)
            
            normalized_samples.append(normalized_sample)
        
        return normalized_samples
    
    def _extract_prompt(self, sample: Dict, dataset_name: str) -> str:
        """Extract prompt from sample based on dataset type."""
        # This is a simplified implementation
        # In production, this would handle different dataset structures
        if isinstance(sample, dict):
            # Try common prompt field names
            for field in ['prompt', 'text', 'question', 'input', 'user_query']:
                if field in sample:
                    return str(sample[field])
            # If no standard field found, use first string value
            for value in sample.values():
                if isinstance(value, str) and len(value) > 10:
                    return value
        return str(sample)
    
    def _extract_context(self, sample: Dict, dataset_name: str) -> str:
        """Extract context from sample."""
        if isinstance(sample, dict):
            for field in ['context', 'system_prompt', 'instruction', 'external_content']:
                if field in sample:
                    return str(sample[field])
        return ""
    
    def _extract_task_type(self, sample: Dict, dataset_name: str) -> str:
        """Extract task type from sample."""
        if isinstance(sample, dict):
            for field in ['task_type', 'type', 'category']:
                if field in sample:
                    return str(sample[field])
        return "general"
    
    def _preserve_bipia_structure(self, normalized_sample: Dict, original_sample: Dict):
        """Preserve BIPIA-specific structure."""
        if isinstance(original_sample, dict):
            if 'user_query' in original_sample:
                normalized_sample['user_query'] = original_sample['user_query']
            if 'external_content' in original_sample:
                normalized_sample['external_content'] = original_sample['external_content']
            if 'injected_instruction' in original_sample:
                normalized_sample['injected_instruction'] = original_sample['injected_instruction']
    
    def _calculate_sample_hash(self, sample: Any) -> str:
        """Calculate hash of sample for deduplication."""
        sample_str = json.dumps(sample, sort_keys=True, default=str)
        return hashlib.sha256(sample_str.encode()).hexdigest()[:16]
    
    def validate_datasets(self) -> Dict[str, Any]:
        """
        Validate prepared datasets.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {},
            'total_samples': 0,
            'total_valid': 0,
            'total_invalid': 0,
            'duplicates': 0,
            'success': True,
            'errors': []
        }
        
        self.logger.info("Starting dataset validation...")
        
        for dataset_name, dataset_config in self.datasets_config.items():
            try:
                self.logger.info(f"Validating dataset: {dataset_name}")
                
                processed_file = Path(config.PROCESSED_DATA_DIR) / f"{dataset_name}_processed.json"
                
                if not processed_file.exists():
                    results['datasets'][dataset_name] = {
                        'status': 'skipped',
                        'reason': 'processed file not found'
                    }
                    continue
                
                # Load processed dataset
                samples = load_json(processed_file)
                
                # Validate samples
                validation_result = self._validate_samples(samples, dataset_name)
                
                results['datasets'][dataset_name] = {
                    'status': 'success',
                    'total_samples': len(samples),
                    'valid_samples': validation_result['valid'],
                    'invalid_samples': validation_result['invalid'],
                    'duplicates': validation_result['duplicates'],
                    'target_samples': dataset_config['target_samples']
                }
                
                results['total_samples'] += len(samples)
                results['total_valid'] += validation_result['valid']
                results['total_invalid'] += validation_result['invalid']
                results['duplicates'] += validation_result['duplicates']
                
                self.logger.info(f"Validated {dataset_name}: {validation_result['valid']}/{len(samples)} valid")
                
            except Exception as e:
                error_msg = f"Failed to validate {dataset_name}: {str(e)}"
                self.logger.error(error_msg)
                results['datasets'][dataset_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                results['errors'].append(error_msg)
                results['success'] = False
        
        # Save validation manifest
        manifest_file = Path(config.MANIFESTS_DIR) / "validation_manifest.json"
        save_json(results, manifest_file)
        
        self.logger.info(f"Dataset validation completed. Valid: {results['total_valid']}/{results['total_samples']}")
        return results
    
    def _validate_samples(self, samples: List[Dict], dataset_name: str) -> Dict[str, int]:
        """
        Validate individual samples.
        
        Args:
            samples: List of samples to validate
            dataset_name: Name of dataset
            
        Returns:
            Dictionary with validation counts
        """
        valid = 0
        invalid = 0
        duplicates = 0
        
        seen_hashes = set()
        required_fields = ['sample_id', 'source_dataset', 'category', 'original_prompt']
        
        for sample in samples:
            # Check required fields
            if not all(field in sample for field in required_fields):
                invalid += 1
                continue
            
            # Check for empty prompts
            if not sample.get('original_prompt') or len(sample['original_prompt'].strip()) < 5:
                invalid += 1
                continue
            
            # Check for duplicates
            sample_hash = sample.get('metadata', {}).get('hash', '')
            if sample_hash in seen_hashes:
                duplicates += 1
            else:
                seen_hashes.add(sample_hash)
                valid += 1
        
        return {
            'valid': valid,
            'invalid': invalid,
            'duplicates': duplicates
        }
    
    def get_dataset_status(self) -> Dict[str, Any]:
        """
        Get current status of all datasets.
        
        Returns:
            Dictionary with dataset status
        """
        status = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {}
        }
        
        for dataset_name, dataset_config in self.datasets_config.items():
            raw_file = Path(config.RAW_DATA_DIR) / f"{dataset_name}_raw.json"
            processed_file = Path(config.PROCESSED_DATA_DIR) / f"{dataset_name}_processed.json"
            
            status['datasets'][dataset_name] = {
                'downloaded': raw_file.exists(),
                'processed': processed_file.exists(),
                'target_samples': dataset_config['target_samples'],
                'category': dataset_config['category']
            }
        
        return status