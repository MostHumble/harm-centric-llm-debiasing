from typing import List, Dict, Union, Tuple, Optional
from models import SpecializedAgent
from prompts import get_feedback_prompt, LEADER_PROMPT
from dataclasses import dataclass
import random

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
            max_new_tokens=self.config['max_new_tokens'],
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
                feedback.append(feedback_messages[:])
                
            random.shuffle(feedback_messages)

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
    """Implements decentralized debiasing approach where all agents collaborate equally"""
    def reduce_bias(
        self, 
        query: str,
        return_lineage: bool = False,
        return_feedback: bool = False
    ) -> Union[str, ReducerOutput]:
        # Track progress if requested
        lineage = [] if return_lineage else None
        all_feedback = [] if return_feedback else None
        
        # Initial responses from all agents
        responses = []
        for agent in self.specialized_agents:
            
            temp = []
            response = agent.get_response(
                query,
                max_new_tokens=self.config['max_new_tokens'],
                temperature=self.config['temperature']
            )
            temp.append(response)

        responses.append(temp)
        if return_lineage:
            lineage.extend(responses)

        # Refinement rounds
        for _ in range(self.config['max_rounds']):
            # Collect feedback from each agent on others' responses
            round_feedback = []
            for i, agent in enumerate(self.specialized_agents):
                other_responses = responses[:i] + responses[i+1:]
                agent_feedback = []
                
                for resp in other_responses:
                    feedback = agent.get_response(
                        resp,
                        max_new_tokens=self.config['max_new_tokens'],
                        temperature=self.config['temperature'],
                        feedback_messages=[]  # Empty list indicates feedback request
                    )
                    agent_feedback.append(feedback)
                round_feedback.append(agent_feedback)
            
            if return_feedback:
                all_feedback.append(round_feedback)
            
            # Generate new responses based on feedback
            new_responses = []
            for i, agent in enumerate(self.specialized_agents):
                # Get feedback received for this agent's last response
                received_feedback = []
                for j, agent_feedback in enumerate(round_feedback):
                    if j != i:  # Skip self-feedback
                        # Calculate index in agent_feedback for this agent's response
                        resp_idx = i if i < j else i - 1
                        received_feedback.append(agent_feedback[resp_idx])
                
                # Generate new response considering feedback
                new_response = agent.get_response(
                    query,
                    max_new_tokens=self.config['max_new_tokens'],
                    temperature=self.config['temperature'],
                    feedback_messages=received_feedback
                )
                new_responses.append(new_response)
            
            # Check for convergence
            if new_responses == responses:
                break
                
            responses = new_responses
            if return_lineage:
                lineage.extend(new_responses)
        
        # Select final response (most common among final responses)
        final_response = max(set(responses), key=responses.count)
        
        return ReducerOutput(
            final_response=final_response,
            lineage=lineage,
            feedback=all_feedback
        ) 