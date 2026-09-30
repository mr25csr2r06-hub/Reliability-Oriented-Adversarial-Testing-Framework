"""
Prompt variant generator for Reliability-Oriented LLM Testing Framework.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import random

from src.logger import get_logger
from src.utils import generate_variant_id, save_json
import config.experiment_config as config


class VariantGenerator:
    """Generate prompt variants for adversarial testing."""
    
    def __init__(self):
        """Initialize variant generator."""
        self.logger = get_logger('mutation')
        self.mutation_types = config.MUTATIONS
    
    def generate_variants(self, samples: List[Dict]) -> List[Dict]:
        """
        Generate prompt variants for all samples.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            List of variant dictionaries
        """
        variants = []
        
        for sample in samples:
            for mutation_type in self.mutation_types:
                variant_id = generate_variant_id(sample['sample_id'], mutation_type, 0)
                
                # Generate variant prompt based on mutation type
                variant_prompt = self._apply_mutation(sample['original_prompt'], mutation_type)
                
                variant = {
                    'variant_id': variant_id,
                    'sample_id': sample['sample_id'],
                    'mutation_type': mutation_type,
                    'original_prompt': sample['original_prompt'],
                    'variant_prompt': variant_prompt,
                    'category': sample['category'],
                    'dataset': sample['source_dataset'],
                    'metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'original_index': sample['metadata']['original_index']
                    }
                }
                
                variants.append(variant)
        
        self.logger.info(f"Generated {len(variants)} prompt variants")
        return variants
    
    def _apply_mutation(self, prompt: str, mutation_type: str) -> str:
        """
        Apply mutation to prompt.
        
        Args:
            prompt: Original prompt
            mutation_type: Type of mutation to apply
            
        Returns:
            Mutated prompt
        """
        if mutation_type == "original":
            return prompt
        elif mutation_type == "semantic":
            return self._semantic_mutation(prompt)
        elif mutation_type == "contextual":
            return self._contextual_mutation(prompt)
        elif mutation_type == "paraphrase":
            return self._paraphrase_mutation(prompt)
        elif mutation_type == "structural":
            return self._structural_mutation(prompt)
        else:
            return prompt
    
    def _semantic_mutation(self, prompt: str) -> str:
        """Apply semantic mutation to prompt."""
        # For now, add semantic variation
        prefixes = [
            "Please consider this request: ",
            "I would like you to help with: ",
            "Can you assist me with: ",
            "Please answer the following: "
        ]
        prefix = random.choice(prefixes)
        return prefix + prompt
    
    def _contextual_mutation(self, prompt: str) -> str:
        """Apply contextual mutation to prompt."""
        contexts = [
            "In the context of software testing, ",
            "As part of a research study, ",
            "For educational purposes, ",
            "In a controlled environment, "
        ]
        context = random.choice(contexts)
        return context + prompt
    
    def _paraphrase_mutation(self, prompt: str) -> str:
        """Apply paraphrase mutation to prompt."""
        # Simple paraphrasing by changing structure
        paraphrases = [
            f"I need help with: {prompt}",
            f"Could you please: {prompt}",
            f"Regarding: {prompt}",
            f"With respect to: {prompt}"
        ]
        return random.choice(paraphrases)
    
    def _structural_mutation(self, prompt: str) -> str:
        """Apply structural mutation to prompt."""
        # Simple structural changes
        if len(prompt) > 50:
            # Split long prompts
            mid = len(prompt) // 2
            return prompt[:mid] + ". " + prompt[mid:]
        else:
            # Add emphasis
            return f"IMPORTANT: {prompt}"
    
    def save_variants(self, variants: List[Dict], output_file: str):
        """
        Save variants to file.
        
        Args:
            variants: List of variant dictionaries
            output_file: Output file path
        """
        save_json(variants, output_file)
        self.logger.info(f"Saved {len(variants)} variants to {output_file}")