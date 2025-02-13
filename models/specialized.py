from typing import Set
from enum import Enum, auto
from .llm import LLMModel
from ..prompts.harm_prompts import get_specialized_context

class HarmType(Enum):
    # Representational Harms
    DEROGATORY = auto()
    DISPARATE_PERFORMANCE = auto()
    ERASURE = auto()
    EXCLUSIONARY = auto()
    MISREPRESENTATION = auto()
    STEREOTYPING = auto()
    TOXICITY = auto()
    # Allocational Harms
    DIRECT_DISCRIMINATION = auto()
    INDIRECT_DISCRIMINATION = auto()

class SpecializedAgent:
    def __init__(self, model: LLMModel, harm_types: Set[HarmType]):
        self.model = model
        self.harm_types = harm_types
        
    def get_response(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0) -> str:
        specialized_prompt = self.get_specialized_prompt(prompt)
        return self.model.generate(specialized_prompt, max_new_tokens, temperature)
        
    def get_specialized_prompt(self, base_prompt: str) -> str:
        harm_types_str = [harm.name for harm in self.harm_types]
        specialized_context = get_specialized_context(harm_types_str)
        return f"{specialized_context}\nTASK: {base_prompt}" 