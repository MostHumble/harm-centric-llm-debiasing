from .base import BiasReducer

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
                feedback = [
                    self._get_feedback(agent, resp)
                    for resp in other_responses
                ]
                
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
            
            if new_responses == responses:  # Convergence check
                break
            responses = new_responses

        return max(set(responses), key=responses.count) 