import json
import csv
import pickle
import yaml
from pathlib import Path
from typing import List, Union, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from prompts import HARM_DESCRIPTIONS

@dataclass
class DebiasedOutput:
    """Class for storing debiased outputs with metadata"""
    original_query: str
    debiased_response: str
    lineage: List[str] = None
    feedback: List[List[str]] = None
    metadata: Dict[str, Any] = None

class IOHandler:
    """Handles input/output operations for the debiasing framework"""
    
    @staticmethod
    def load_queries(input_file: Union[str, Path]) -> List[str]:
        """
        Load queries from various file formats.
        
        Args:
            input_file: Path to input file (supports .json, .csv, .pkl, .txt)
            
        Returns:
            List of query strings
            
        Raises:
            ValueError: If file format is invalid or unsupported
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        if input_path.suffix == '.json':
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "queries" in data:
                    return data["queries"]
                else:
                    raise ValueError("Invalid JSON format: Expected a list or dict with 'queries' key")
                    
        elif input_path.suffix == '.csv':
            with open(input_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                return [row[0] for row in reader if row]  # Assumes queries in first column
                
        elif input_path.suffix == '.pkl':
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    return data
                else:
                    raise ValueError("Invalid pickle format: Expected a list of queries")
                    
        elif input_path.suffix == '.txt':
            with open(input_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
                
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

    @staticmethod
    def save_outputs(
        outputs: List[DebiasedOutput],
        output_file: Union[str, Path],
        include_metadata: bool = True
    ) -> None:
        """
        Save debiased outputs to file.
        
        Args:
            outputs: List of DebiasedOutput objects
            output_file: Path to output file (supports .json, .csv, .pkl)
            include_metadata: Whether to include metadata in output
            
        Raises:
            ValueError: If file format is unsupported
        """
        output_path = Path(output_file)
        
        # Convert outputs to dicts
        output_dicts = [asdict(output) for output in outputs]
        if not include_metadata:
            for d in output_dicts:
                d.pop('metadata', None)
                
        if output_path.suffix == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_dicts, f, indent=2)
                
        elif output_path.suffix == '.csv':
            if not outputs:
                return
                
            fieldnames = ['original_query', 'debiased_response']
            if any(o.lineage for o in outputs):
                fieldnames.append('lineage')
            if any(o.feedback for o in outputs):
                fieldnames.append('feedback')
            if include_metadata and any(o.metadata for o in outputs):
                fieldnames.append('metadata')
                
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for output_dict in output_dicts:
                    # Convert lists to strings for CSV
                    if 'lineage' in output_dict:
                        output_dict['lineage'] = json.dumps(output_dict['lineage'])
                    if 'feedback' in output_dict:
                        output_dict['feedback'] = json.dumps(output_dict['feedback'])
                    if 'metadata' in output_dict:
                        output_dict['metadata'] = json.dumps(output_dict['metadata'])
                    writer.writerow(output_dict)
                    
        elif output_path.suffix == '.pkl':
            with open(output_path, 'wb') as f:
                pickle.dump(outputs, f)
                
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")

    @staticmethod
    def process_harm_assignments(config_path: Union[str, Path]) -> Tuple[Dict[str, List[str]], str]:
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

        config_path = Path(config_path)
            
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        if config_path.suffix != '.yaml' and config_path.suffix != '.yml':
            raise ValueError("Config file must be YAML format")

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