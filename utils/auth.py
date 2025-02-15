from typing import Optional
import os
from huggingface_hub import login, HfApi, HfFolder
import yaml

def setup_hf_auth(token: Optional[str] = None, token_path: str = "config/hf_token.yml") -> bool:
    """Setup Hugging Face authentication using token
    
    Args:
        token: HF access token string. If None, will try to load from token_path
        token_path: Path to YAML file containing token
    
    Returns:
        bool: True if authentication successful
    """
    try:
        # Try to get token from param first
        hf_token = token
        
        # If no token provided, try to load from file
        if not hf_token:
            if os.path.exists(token_path):
                with open(token_path, 'r') as f:
                    config = yaml.safe_load(f)
                    hf_token = config.get('hf_token')
        
        # If still no token, try environment variable
        if not hf_token:
            hf_token = os.environ.get('HF_TOKEN')
            
        if not hf_token:
            raise ValueError(
                "No HF token found. Please either:\n"
                "1. Pass token directly\n"
                "2. Create config/hf_token.yml with hf_token: YOUR_TOKEN\n" 
                "3. Set HF_TOKEN environment variable"
            )
            
        # Login to HF
        login(token=hf_token)
        
        # Verify token works by trying to get user info
        api = HfApi()
        api.whoami()
        
        # Save token for transformers library
        HfFolder.save_token(hf_token)
        
        return True
        
    except Exception as e:
        print(f"Error setting up HF authentication: {str(e)}")
        return False 