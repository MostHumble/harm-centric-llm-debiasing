from .base import BiasReducer
from ..prompts.harm_prompts import LEADER_PROMPT

class CentralizedReducer(BiasReducer):
    """Implements leader-follower debiasing approach"""
    def reduce_bias(self, query: str) -> str:
        # Leader should be aware of all harm types
        leader = self.specialized_agents[0]
        followers = self.specialized_agents[1:]

        # Initial response with comprehensive harm awareness
        leader_prompt = (
            f"{LEADER_PROMPT}\n\n"
            f"Query: {query}\n"
            "Generate a response that proactively addresses all potential harms:"
        )
        
        current_response = leader.get_response(
            leader_prompt,
            max_new_tokens=self.config['max_new_tokens'],
            temperature=self.config['temperature']
        )

        for _ in range(self.config['max_rounds']):
            # Get specialized feedback from followers
            all_feedback = [
                self._get_feedback(follower, current_response)
                for follower in followers
            ]

            feedback_summary = " | ".join(all_feedback)
            refinement_prompt = (
                f"{LEADER_PROMPT}\n\n"
                f"Original query: {query}\n"
                f"Previous response: {current_response}\n"
                f"Specialized feedback received: {feedback_summary}\n"
                "Generate an improved response that addresses the feedback while "
                "maintaining awareness of all potential harms:"
            )
            
            new_response = leader.get_response(
                refinement_prompt,
                max_new_tokens=self.config['max_new_tokens'],
                temperature=self.config['temperature']
            )
            
            if new_response == current_response:  # Convergence check
                break
            current_response = new_response

        return current_response 