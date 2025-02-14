from typing import Set, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from prompts import get_specialized_context

class LLMModel:
    """Simple wrapper for transformer models with chat template support"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_8bit=True)},
            device_map="auto"
        )

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 64, temperature: float = 0.0) -> str:
        # Apply chat template
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True, 
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            tokenized_chat,
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
        messages = self.get_specialized_messages(prompt)
        return self.model.generate(messages, max_new_tokens, temperature)
        
    def get_specialized_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        specialized_context = get_specialized_context(self.harm_types)
        return [
            {
                "role": "system",
                "content": specialized_context
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ] 