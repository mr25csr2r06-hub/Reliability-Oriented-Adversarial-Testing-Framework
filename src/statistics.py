"""
Statistical analysis for LLM testing results.
"""

import numpy as np
from typing import Dict, List, Any
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, wilcoxon
import pandas as pd

from src.logger import get_logger


class StatisticalAnalysis:
    """Perform statistical analysis on experimental results."""
    
    def __init__(self):
        """Initialize statistical analyzer."""
        self.logger = get_logger('metrics')
    
    def analyze_model_comparison(self, model_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Perform statistical comparison between models.
        
        Args:
            model_metrics: Dictionary of metrics per model
            
        Returns:
            Statistical analysis results
        """
        analysis = {
            'asr_comparison': {},
            'rr_comparison': {},
            'overall_significance': {},
            'note': 'Insufficient data for statistical comparison'
        }
        
        # Check if we have multiple models for comparison
        if len(model_metrics) < 2:
            analysis['note'] = 'Need at least 2 models for comparison'
            return analysis
        
        # Extract ASR values
        asr_values = {model: metrics.get('ASR', 0) for model, metrics in model_metrics.items()}
        rr_values = {model: metrics.get('RR', 0) for model, metrics in model_metrics.items()}
        
        # Perform pairwise comparisons
        models = list(model_metrics.keys())
        
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                # Simple comparison for now (would need sample-level data for proper tests)
                asr_diff = abs(asr_values[model1] - asr_values[model2])
                rr_diff = abs(rr_values[model1] - rr_values[model2])
                
                analysis['asr_comparison'][f"{model1}_vs_{model2}"] = {
                    'difference': asr_diff,
                    'model1_asr': asr_values[model1],
                    'model2_asr': asr_values[model2]
                }
                
                analysis['rr_comparison'][f"{model1}_vs_{model2}"] = {
                    'difference': rr_diff,
                    'model1_rr': rr_values[model1],
                    'model2_rr': rr_values[model2]
                }
        
        if analysis['asr_comparison']:
            analysis['note'] = 'Comparison completed'
        
        return analysis
    
    def analyze_mutation_impact(self, execution_results: Dict) -> Dict[str, Any]:
        """
        Analyze impact of different mutation types.
        
        Args:
            execution_results: Execution results dictionary
            
        Returns:
            Mutation impact analysis
        """
        results = execution_results.get('results', [])
        
        # Group by mutation type
        mutation_data = {}
        for result in results:
            mutation = result.get('mutation_type', 'unknown')
            if mutation not in mutation_data:
                mutation_data[mutation] = []
            mutation_data[mutation].append(result)
        
        analysis = {
            'mutation_stats': {},
            'mutation_comparison': {}
        }
        
        # Calculate statistics per mutation
        for mutation, data in mutation_data.items():
            if data:
                unsafe_count = sum(1 for r in data if 'response' in r and r['response'])
                total = len(data)
                analysis['mutation_stats'][mutation] = {
                    'total_executions': total,
                    'unsafe_rate': unsafe_count / total if total > 0 else 0
                }
        
        return analysis
    
    def calculate_confidence_intervals(self, values: List[float], confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate confidence intervals for a set of values.
        
        Args:
            values: List of numerical values
            confidence: Confidence level (default 0.95)
            
        Returns:
            Dictionary with confidence interval statistics
        """
        if not values or len(values) < 2:
            return {
                'mean': 0.0,
                'std': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'sample_size': len(values)
            }
        
        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array, ddof=1)
        n = len(values_array)
        
        # Calculate confidence interval
        from scipy.stats import t
        t_critical = t.ppf((1 + confidence) / 2, n - 1)
        margin_of_error = t_critical * (std / np.sqrt(n))
        
        return {
            'mean': mean,
            'std': std,
            'ci_lower': mean - margin_of_error,
            'ci_upper': mean + margin_of_error,
            'sample_size': n,
            'confidence_level': confidence
        }
    
    def perform_hypothesis_test(self, group1: List[float], group2: List[float], 
                                 test_type: str = 'mannwhitney') -> Dict[str, Any]:
        """
        Perform hypothesis test between two groups.
        
        Args:
            group1: First group of values
            group2: Second group of values
            test_type: Type of test ('mannwhitney', 'ttest', 'wilcoxon')
            
        Returns:
            Test results
        """
        if not group1 or not group2:
            return {
                'test': test_type,
                'p_value': 1.0,
                'statistic': 0.0,
                'significant': False,
                'error': 'Insufficient data'
            }
        
        try:
            if test_type == 'mannwhitney':
                statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            elif test_type == 'ttest':
                statistic, p_value = stats.ttest_ind(group1, group2)
            elif test_type == 'wilcoxon':
                statistic, p_value = wilcoxon(group1, group2)
            else:
                return {
                    'test': test_type,
                    'p_value': 1.0,
                    'statistic': 0.0,
                    'significant': False,
                    'error': 'Unknown test type'
                }
            
            return {
                'test': test_type,
                'p_value': p_value,
                'statistic': statistic,
                'significant': p_value < 0.05,
                'alpha': 0.05
            }
        except Exception as e:
            return {
                'test': test_type,
                'p_value': 1.0,
                'statistic': 0.0,
                'significant': False,
                'error': str(e)
            }
    
    def calculate_effect_size(self, group1: List[float], group2: List[float]) -> Dict[str, float]:
        """
        Calculate effect size (Cohen's d) between two groups.
        
        Args:
            group1: First group of values
            group2: Second group of values
            
        Returns:
            Effect size statistics
        """
        if not group1 or not group2:
            return {
                'cohens_d': 0.0,
                'interpretation': 'none',
                'error': 'Insufficient data'
            }
        
        try:
            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            
            # Cohen's d
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            
            # Interpret effect size
            if abs(cohens_d) < 0.2:
                interpretation = 'small'
            elif abs(cohens_d) < 0.5:
                interpretation = 'medium'
            elif abs(cohens_d) < 0.8:
                interpretation = 'large'
            else:
                interpretation = 'very large'
            
            return {
                'cohens_d': cohens_d,
                'interpretation': interpretation,
                'mean1': mean1,
                'mean2': mean2,
                'pooled_std': pooled_std
            }
        except Exception as e:
            return {
                'cohens_d': 0.0,
                'interpretation': 'none',
                'error': str(e)
            }