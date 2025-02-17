from typing import Set, List, Dict
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from prompts import get_specialized_context, get_feedback_prompt, get_leader_integration_prompt
from utils.auth import setup_hf_auth
import re  # Add this import at the top
from prompts import HARM_DESCRIPTIONS

class LLMModel:
    """Simple wrapper for transformer models with chat template support"""
    def __init__(self, model_name: str):
        # Setup HF auth before loading model
        if not setup_hf_auth():
            raise RuntimeError("Failed to authenticate with Hugging Face")
        
        #quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        self.model_name = model_name  
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype='auto',
            trust_remote_code=True,
            #quantization_config = quantization_config
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

        if temperature > 0.0:
            outputs = self.model.generate(
                tokenized_chat,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        else:
            outputs = self.model.generate(
                tokenized_chat,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        # Ignore the generation prompt
        return self.tokenizer.decode(outputs[0][len(tokenized_chat[0]):], skip_special_tokens=True)
        

class SpecializedAgent:
    def __init__(self, model: LLMModel, harm_types: Set[str]):
        self.model = model
        self.harm_types = set(HARM_DESCRIPTIONS.keys())
        self.is_leader = len(harm_types) == 0  # No harm types assigned (leader)
        
    def _validate_json_response(self, response: str) -> str:
        """Extract and validate JSON from model response that may contain markdown formatting"""
        # First try to find JSON within triple backticks
        match = re.search(r'```(?:json)?\n(.*?)\n```', response, re.DOTALL)
        
        # If found within backticks, use that, otherwise try the full response
        json_str = match.group(1).strip() if match else response.strip()
        
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
                
                if "analysis" not in response_obj or "recommendations" not in response_obj:
                    raise ValueError("Missing required fields")
                
                # Validate issues are within assigned harm types with correct casing
                for harm_type in self.harm_types:
                    if harm_type not in response_obj["analysis"]:
                        raise ValueError(f"Missing analysis for {harm_type}")
                
                return json.dumps(response_obj)  # Return formatted JSON string
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        
    def get_response(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0, feedback_messages: List[Dict[str, str]] = None) -> str:
        if self.is_leader:
            messages = get_leader_integration_prompt(prompt, feedback_messages)
        else:
            # For feedback requests, use specialized feedback prompt
            if feedback_messages is None:
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
            print(f'Invalid JSON response: {response}')
            retry_messages = messages + [
                {"role": "assistant", "content": response},
                {"role": "user", "content": "Make sure to follow the correct JSON format and use the exact same harm type keys in UPPERCASE as provided in the input list. Please reformat your response."}
            ]
            response = self.model.generate(retry_messages, max_new_tokens, temperature)
            return self._validate_json_response(response)
        