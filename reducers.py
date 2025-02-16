from typing import List, Dict, Union, Tuple, Optional
from models import SpecializedAgent
from prompts import get_feedback_prompt, LEADER_PROMPT
from dataclasses import dataclass

@dataclass
class ReducerOutput:
    """Output class for BiasReducer results"""
    final_response: str
    lineage: Optional[List[str]] = None
    feedback: Optional[List[List[str]]] = None

class BiasReducer:
    """Base class for different debiasing strategies"""
    def __init__(self, specialized_agents: List[SpecializedAgent], config: Dict):
        self.specialized_agents = specialized_agents
        self.config = config

    def _get_feedback(self, agent: SpecializedAgent, response: str) -> str:
        
        return agent.get_response(
            response,
            max_new_tokens=self.config['feedback_tokens'],
            temperature=self.config['temperature'],
        )

    def reduce_bias(self, query: str) -> str:
        raise NotImplementedError

class CentralizedReducer(BiasReducer):
    """Implements leader-follower debiasing approach"""
    def reduce_bias(
        self, 
        query: str, 
        return_lineage: bool = False, 
        return_feedback: bool = False
    ) -> Union[str, ReducerOutput]:
        leader = self.specialized_agents[0]
        followers = self.specialized_agents[1:]
        
        lineage = [] if return_lineage else None
        feedback = [] if return_feedback else None
    
        for _ in range(self.config['max_rounds']):
            feedback_messages = [self._get_feedback(f, query) for f in followers]
            
            if return_lineage:
                lineage.append(query)
            if return_feedback:
                feedback.append(feedback_messages)
                
            new_response = leader.get_response(
                query,
                max_new_tokens=self.config['max_new_tokens'],
                temperature=self.config['temperature'],
                feedback_messages=feedback_messages
            )
            
            if new_response == query:
                break
            query = new_response

        return ReducerOutput(
                final_response=query,
                lineage=lineage,
                feedback=feedback
            )

class DecentralizedReducer(BiasReducer):
    def reduce_bias(self, query: str) -> str:
        responses = []
        for agent in self.specialized_agents:
            response = agent.get_response(
                query,
                max_new_tokens=self.config['max_new_tokens'],
                temperature=self.config['temperature']
            )
            responses.append(response)

        for _ in range(self.config['max_rounds']):
            new_responses = []
            for i, agent in enumerate(self.specialized_agents):
                other_responses = responses[:i] + responses[i+1:]
                feedback = [self._get_feedback(agent, r) for r in other_responses]
                feedback_summary = " | ".join(feedback)
                
                consensus_prompt = (
                    f"Original query: {query}\n"
                    f"Other responses: {other_responses}\n"
                    f"Feedback on responses: {feedback_summary}\n"
                    "Generate improved response considering others' perspectives:"
                )
                
                new_response = agent.get_response(
                    consensus_prompt,
                    max_new_tokens=self.config['max_new_tokens'],
                    temperature=self.config['temperature']
                )
                new_responses.append(new_response)
            
            if new_responses == responses:
                break
            responses = new_responses

        return max(set(responses), key=responses.count) 