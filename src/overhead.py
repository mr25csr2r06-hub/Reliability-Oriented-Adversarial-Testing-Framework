"""
Testing overhead analysis for LLM experiments.
"""

import time
import psutil
import torch
from typing import Dict, List, Any
from collections import defaultdict

from src.logger import get_logger


class OverheadAnalysis:
    """Analyze computational and memory overhead of testing."""
    
    def __init__(self):
        """Initialize overhead analyzer."""
        self.logger = get_logger('metrics')
    
    def analyze_execution_overhead(self, execution_results: Dict) -> Dict[str, Any]:
        """
        Analyze overhead from execution results.
        
        Args:
            execution_results: Execution results dictionary
            
        Returns:
            Overhead analysis results
        """
        results = execution_results.get('results', [])
        
        if not results:
            return self._empty_overhead()
        
        # Organize by model
        model_data = defaultdict(list)
        for result in results:
            model = result.get('model', 'unknown')
            model_data[model].append(result)
        
        analysis = {
            'overall_overhead': self._calculate_overhead_stats(results),
            'model_overhead': {},
            'total_time': sum(r.get('latency', 0) for r in results),
            'total_tokens': sum(r.get('output_tokens', 0) for r in results)
        }
        
        for model, data in model_data.items():
            analysis['model_overhead'][model] = self._calculate_overhead_stats(data)
        
        return analysis
    
    def _empty_overhead(self) -> Dict[str, Any]:
        """Return empty overhead when no results available."""
        return {
            'overall_overhead': {
                'total_executions': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'total_tokens': 0,
                'avg_tokens': 0.0,
                'tokens_per_second': 0.0
            },
            'model_overhead': {},
            'total_time': 0.0,
            'total_tokens': 0
        }
    
    def _calculate_overhead_stats(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate overhead statistics for a set of results."""
        if not results:
            return {
                'total_executions': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'total_tokens': 0,
                'avg_tokens': 0.0,
                'tokens_per_second': 0.0
            }
        
        total_time = sum(r.get('latency', 0) for r in results)
        total_tokens = sum(r.get('output_tokens', 0) for r in results)
        total_executions = len(results)
        
        avg_time = total_time / total_executions if total_executions > 0 else 0
        avg_tokens = total_tokens / total_executions if total_executions > 0 else 0
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        return {
            'total_executions': total_executions,
            'total_time': total_time,
            'avg_time': avg_time,
            'total_tokens': total_tokens,
            'avg_tokens': avg_tokens,
            'tokens_per_second': tokens_per_second
        }
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """
        Measure current memory usage.
        
        Returns:
            Memory usage statistics
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
                gpu_memory[f'gpu_{i}'] = {
                    'allocated_gb': allocated,
                    'reserved_gb': reserved
                }
        
        return {
            'cpu_memory_gb': memory_info.rss / (1024 ** 3),
            'cpu_memory_mb': memory_info.rss / (1024 ** 2),
            'gpu_memory': gpu_memory
        }
    
    def calculate_resource_efficiency(self, execution_results: Dict) -> Dict[str, Any]:
        """
        Calculate resource efficiency metrics.
        
        Args:
            execution_results: Execution results dictionary
            
        Returns:
            Resource efficiency metrics
        """
        results = execution_results.get('results', [])
        
        if not results:
            return {
                'executions_per_second': 0.0,
                'tokens_per_second': 0.0,
                'avg_time_per_execution': 0.0
            }
        
        total_time = sum(r.get('latency', 0) for r in results)
        total_tokens = sum(r.get('output_tokens', 0) for r in results)
        total_executions = len(results)
        
        return {
            'executions_per_second': total_executions / total_time if total_time > 0 else 0,
            'tokens_per_second': total_tokens / total_time if total_time > 0 else 0,
            'avg_time_per_execution': total_time / total_executions if total_executions > 0 else 0
        }