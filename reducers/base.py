from typing import List, Dict
from ..models.specialized import SpecializedAgent
from ..prompts.harm_prompts import get_feedback_prompt

class BiasReducer:
    """Base class for different debiasing strategies"""
    def __init__(self, specialized_agents: List[SpecializedAgent], config: Dict):
        self.specialized_agents = specialized_agents
        self.config = config

    def _get_feedback(self, agent: SpecializedAgent, response: str) -> str:
        feedback_prompt = get_feedback_prompt(response)
        return agent.get_response(
            feedback_prompt,
            max_new_tokens=self.config['feedback_tokens'],
            temperature=self.config['temperature']
        )

    def reduce_bias(self, query: str) -> str:
        raise NotImplementedError 