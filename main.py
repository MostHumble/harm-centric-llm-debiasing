from typing import List, Dict, Tuple
import argparse
import yaml
from tqdm import tqdm 
from models import LLMModel, SpecializedAgent
from reducers import CentralizedReducer, DecentralizedReducer
from prompts import HARM_DESCRIPTIONS
from utils.io_utils import IOHandler, DebiasedOutput

class MultiLLMDebiasing:
    def __init__(self, harm_assignments: Dict[str, List[str]], config: Dict, strategy: str = "centralized"):
        # Create specialized agents
        self.specialized_agents = []
        
        for i, model_name in enumerate(harm_assignments.keys()):
            model = LLMModel(model_name)
            
            harm_types = set(harm_assignments.get(model_name, []))
                
            self.specialized_agents.append(SpecializedAgent(model, harm_types))
        
        # Initialize strategy
        if strategy == "centralized":
            self.reducer = CentralizedReducer(self.specialized_agents, config)
        else:
            self.reducer = DecentralizedReducer(self.specialized_agents, config)

    def get_debiased_response(self, query: str, return_lineage: bool = False, return_feedback: bool = False) -> str:
        """Get debiased response using the initialized strategy"""
        return self.reducer.reduce_bias(query, return_lineage, return_feedback)


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-LLM Debiasing Framework')
    
    parser.add_argument('--harm-assignments', type=str, required=True,
                       help='YAML file defining models and their harm types')
    parser.add_argument('--input-file', type=str, required=True,
                       help='Input file containing queries to debias')
    parser.add_argument('--output-file', type=str, required=True,
                       help='Output file to save debiased responses')
    parser.add_argument('--max-rounds', type=int, default=3,
                       help='Maximum number of refinement rounds')
    parser.add_argument('--max-new-tokens', type=int, default=512,
                       help='Maximum number of new tokens for response generation')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for response generation')
    parser.add_argument('--return-lineage', action='store_true',
                       help='Return lineage of debiasing steps')
    parser.add_argument('--return-feedback', action='store_true',
                       help='Return feedback from followers')
    parser.add_argument('--include-metadata', action='store_true',
                       help='Include metadata in output')
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    # Process harm assignments
    harm_assignments, strategy = IOHandler.process_harm_assignments(args.harm_assignments)

    # Load queries
    queries = IOHandler.load_queries(args.input_file)
    
    config = {
        'max_rounds': args.max_rounds,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature
    }

    debiasing = MultiLLMDebiasing(
        harm_assignments=harm_assignments,
        config=config,
        strategy=strategy
    )
    
    # Process queries and collect outputs
    outputs = []
    for i, query in tqdm(enumerate(queries), desc="Processing queries", total=len(queries)):
        
        try:
            result = debiasing.get_debiased_response(
                query, 
                args.return_lineage, 
                args.return_feedback
            )
            
        except Exception as e:
            print(f"Error processing query {i}: {e}")
            continue    

        if args.include_metadata:
            metadata = {"query_index": i}
        else:
            metadata = None

        output = DebiasedOutput(
            original_query=query,
            debiased_response=result.final_response,
            lineage=result.lineage,
            feedback=result.feedback,
            metadata=metadata
        )
        outputs.append(output)
    
    # Save results
    IOHandler.save_outputs(outputs, args.output_file)

if __name__ == "__main__":
    main() 