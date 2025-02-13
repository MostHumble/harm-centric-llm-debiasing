from typing import List, Dict
import argparse
import json
from models.llm import LLMModel
from models.specialized import SpecializedAgent, HarmType
from reducers.centralized import CentralizedReducer
from reducers.decentralized import DecentralizedReducer

class MultiLLMDebiasing:
    def __init__(self, model_names: List[str], harm_assignments: Dict[str, List[str]], 
                 config: Dict, strategy: str = "centralized"):
        if strategy not in ["centralized", "decentralized"]:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        # Create specialized agents
        self.specialized_agents = []
        
        for i, model_name in enumerate(model_names):
            model = LLMModel(model_name)
            
            # In centralized mode, first model (leader) gets all harm types
            if strategy == "centralized" and i == 0:
                harm_types = {harm_type for harm_type in HarmType}
            else:
                harm_types = {HarmType[harm.upper()] 
                            for harm in harm_assignments.get(model_name, [])}
                
            self.specialized_agents.append(SpecializedAgent(model, harm_types))
        
        # Initialize strategy
        if strategy == "centralized":
            self.reducer = CentralizedReducer(self.specialized_agents, config)
        else:
            self.reducer = DecentralizedReducer(self.specialized_agents, config)

    def get_debiased_response(self, query: str) -> str:
        """Get debiased response using the initialized strategy"""
        return self.reducer.reduce_bias(query)

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-LLM Debiasing Framework')
    
    parser.add_argument('--models', nargs='+', required=True,
                       help='List of model names')
    parser.add_argument('--harm-assignments', type=str, required=True,
                       help='JSON file mapping model names to lists of harm types')
    parser.add_argument('--strategy', choices=['centralized', 'decentralized'], 
                       default='centralized',
                       help='Debiasing strategy to use')
    parser.add_argument('--max-rounds', type=int, default=3,
                       help='Maximum number of refinement rounds')
    parser.add_argument('--max-new-tokens', type=int, default=64,
                       help='Maximum number of new tokens for response generation')
    parser.add_argument('--feedback-tokens', type=int, default=128,
                       help='Maximum number of tokens for feedback generation')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for response generation')
    parser.add_argument('--query', type=str, required=True,
                       help='Input query to debias')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load harm assignments from JSON file
    with open(args.harm_assignments) as f:
        harm_assignments = json.load(f)
    
    config = {
        'max_rounds': args.max_rounds,
        'max_new_tokens': args.max_new_tokens,
        'feedback_tokens': args.feedback_tokens,
        'temperature': args.temperature
    }
    
    debiasing = MultiLLMDebiasing(
        model_names=args.models,
        harm_assignments=harm_assignments,
        config=config,
        strategy=args.strategy
    )
    
    response = debiasing.get_debiased_response(args.query)
    print(f"\nStrategy: {args.strategy}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main() 