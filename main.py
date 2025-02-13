from typing import List, Dict, Tuple
import argparse
import yaml
from models import LLMModel, SpecializedAgent
from reducers import CentralizedReducer, DecentralizedReducer
from prompts import HARM_DESCRIPTIONS

class MultiLLMDebiasing:
    def __init__(self, harm_assignments: Dict[str, List[str]], config: Dict, strategy: str = "centralized"):
        # Create specialized agents
        self.specialized_agents = []
        
        for i, model_name in enumerate(harm_assignments.keys()):
            model = LLMModel(model_name)
            
            # In centralized mode, first model (leader) gets all harm types
            if strategy == "centralized" and i == 0:
                harm_types = set(HARM_DESCRIPTIONS.keys())
            else:
                harm_types = set(harm_assignments.get(model_name, []))
                
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
    
    parser.add_argument('--harm-assignments', type=str, required=True,
                       help='YAML file defining models and their harm types')
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
    
    args = parser.parse_args()
    
    return args

def process_harm_assignments(config_path: str) -> Tuple[Dict[str, List[str]], str]:
    """
    Process and validate the harm assignments YAML file.
    
    Returns:
        Tuple containing:
        - Dictionary of harm assignments
        - Strategy ('centralized' or 'decentralized')
    
    Raises:
        ValueError: If harm assignments are invalid or don't cover all harm types
    """
    # Load YAML config
    with open(config_path) as f:
        harm_config = yaml.safe_load(f)
    
    harm_assignments = {
        model: config['harm_types']
        for model, config in harm_config.items()
    }
    
    # Determine strategy
    empty_count = sum(1 for harms in harm_assignments.values() if not harms)
    
    if empty_count > 1:
        raise ValueError("Invalid configuration: At most one model can have empty harm assignments")
    
    if empty_count == 1:
        strategy = 'centralized'
    else:
        strategy = 'decentralized'
    
    # Validate that all harm types are covered
    all_assigned_harms = set()
    for harms in harm_assignments.values():
        all_assigned_harms.update(harms)
    
    all_possible_harms = set(HARM_DESCRIPTIONS.keys())
    
    uncovered_harms = all_possible_harms - all_assigned_harms
    if uncovered_harms:
        raise ValueError(f"The following harm types are not covered by any model: {uncovered_harms}")
    
    unknown_harms = all_assigned_harms - all_possible_harms
    if unknown_harms:
        raise ValueError(f"The following assigned harm types are not recognized: {unknown_harms}")
    
    return harm_assignments, strategy

def main():
    args = parse_args()
    
    harm_assignments, strategy = process_harm_assignments(args.harm_assignments)
    
    config = {
        'max_rounds': args.max_rounds,
        'max_new_tokens': args.max_new_tokens,
        'feedback_tokens': args.feedback_tokens,
        'temperature': args.temperature
    }
    
    debiasing = MultiLLMDebiasing(
        harm_assignments=harm_assignments,
        config=config,
        strategy=strategy
    )
    
    response = debiasing.get_debiased_response(args.query)
    print(f"\nStrategy: {strategy}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main() 