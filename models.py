from typing import Set, List, Dict
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from prompts import get_specialized_context, get_feedback_prompt, LEADER_PROMPT

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
        self.is_leader = len(harm_types) == 9  # All harm types assigned
        
    def _validate_json_response(self, json_str: str) -> str:
        """Validate JSON format based on agent role"""
        try:
            response_obj = json.loads(json_str)
            
            if self.is_leader:
                # Leader validation
                if not isinstance(response_obj, dict):
                    raise ValueError("Response must be a JSON object")
                
                if "response" not in response_obj or "analysis" not in response_obj:
                    raise ValueError("Missing required fields")
                
                # Validate all harm types are analyzed with correct casing
                for harm_type in self.harm_types:
                    if harm_type not in response_obj["analysis"]:
                        raise ValueError(f"Missing analysis for {harm_type}")
                
                return response_obj["response"]
            else:
                # Follower validation
                if not isinstance(response_obj, dict):
                    raise ValueError("Response must be a JSON object")
                
                if "identified_issues" not in response_obj:
                    raise ValueError("Missing identified_issues field")
                
                # Validate issues are within assigned harm types with correct casing
                for issue in response_obj["identified_issues"]:
                    if "harm_type" not in issue:
                        raise ValueError("Missing harm_type in issue")
                    if issue["harm_type"] not in self.harm_types:
                        raise ValueError(f"Unauthorized harm type: {issue['harm_type']}")
                
                return json_str
            
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")
        
    def get_response(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0) -> str:
        if self.is_leader:
            messages = [
                {"role": "system", "content": LEADER_PROMPT},
                {"role": "user", "content": prompt}
            ]
        else:
            # For feedback requests, use specialized feedback prompt
            if "Analyze this response for potential biases" in prompt:
                messages = get_feedback_prompt(prompt, list(self.harm_types))
            else:
                # For other requests, use specialized context
                messages = [
                    {"role": "system", "content": get_specialized_context(list(self.harm_types))},
                    {"role": "user", "content": prompt}
                ]
        
        response = self.model.generate(messages, max_new_tokens, temperature)
        
        try:
            return self._validate_json_response(response)
        except ValueError as e:
            # Retry with explicit format reminder
            retry_messages = messages + [
                {"role": "assistant", "content": response},
                {"role": "user", "content": "Your response was not in the correct JSON format. Please reformat your response."}
            ]
            response = self.model.generate(retry_messages, max_new_tokens, temperature)
            return self._validate_json_response(response)
        
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