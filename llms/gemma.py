"""
Gemma LLM implementation using Hugging Face.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, Any
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMResponse:
    """Response from LLM."""
    text: str
    latency: float
    input_tokens: int
    output_tokens: int
    model: str


class Gemma:
    """Gemma model wrapper."""
    
    def __init__(self, model_name: str = "google/gemma-7b"):
        """
        Initialize Gemma model.
        
        Args:
            model_name: Hugging Face model identifier.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = os.getenv("HF_TOKEN")
        
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.hf_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            token=self.hf_token
        )
        # Explicitly move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> LLMResponse:
        """
        Generate response for prompt.
        
        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
        
        Returns:
            LLMResponse object.
        """
        import time
        start_time = time.time()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Handle temperature=0 for greedy decoding
        if temperature == 0.0:
            do_sample = False
            temperature = 1.0  # Set to default for greedy decoding
        else:
            do_sample = True
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Move outputs to CPU for decoding to avoid device issues
        outputs = outputs.to('cpu')
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = generated_text[len(prompt):].strip()
        
        latency = time.time() - start_time
        input_tokens = inputs['input_ids'].shape[1]
        output_tokens = outputs.shape[1] - input_tokens
        
        return LLMResponse(
            text=response_text,
            latency=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model_name
        )
