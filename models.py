from typing import Set
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from prompts import get_specialized_context

class LLMModel:
    """Simple wrapper for transformer models"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class SpecializedAgent:
    def __init__(self, model: LLMModel, harm_types: Set[str]):
        self.model = model
        self.harm_types = harm_types
        
    def get_response(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0) -> str:
        specialized_prompt = self.get_specialized_prompt(prompt)
        return self.model.generate(specialized_prompt, max_new_tokens, temperature)
        
    def get_specialized_prompt(self, base_prompt: str) -> str:
        specialized_context = get_specialized_context(self.harm_types)
        return f"{specialized_context}\nTASK: {base_prompt}" 