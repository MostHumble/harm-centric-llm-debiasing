# Multi-LLM Debiasing Framework

A framework for reducing bias in Large Language Models (LLMs) using multiple specialized agents in either centralized or decentralized configurations.

## Overview

This framework implements two debiasing strategies:

1. **Centralized (Leader-Follower)**: One model acts as a leader with comprehensive bias awareness, while others provide specialized feedback
2. **Decentralized (Consensus)**: Multiple models collaborate as equals, each specializing in different types of bias

## Features

- Flexible model assignment for different types of bias detection
- Support for both centralized and decentralized debiasing strategies
- Comprehensive bias coverage including:
  - Representational harms (derogatory language, stereotyping, etc.)
  - Allocational harms (direct/indirect discrimination)
- Iterative refinement process with configurable rounds
- Automatic strategy selection based on harm assignments

## Installation

```bash
git clone https://github.com/yourusername/multi-llm-debiasing-framework.git
cd multi-llm-debiasing-framework
pip install -r requirements.txt
```

## Usage

1. Create a YAML file defining model assignments:

```yaml
meta-llama/Llama-2-3b-chat-hf:
  harm_types: []  # Empty list indicates this is the leader in centralized mode
  
meta-llama/Llama-2-7b-chat-hf:
  harm_types:
    - DEROGATORY
    - TOXICITY
    - DIRECT_DISCRIMINATION
    
mistralai/Mixtral-8x7B-Instruct-v0.1:
  harm_types:
    - STEREOTYPING
    - MISREPRESENTATION
    - INDIRECT_DISCRIMINATION
```

2. Run the debiasing:

```bash
python main.py \
  --harm-assignments config.yaml \
  --query "Your input text" \
  --max-rounds 3 \
  --temperature 0.0
```

## Configuration

- `max_rounds`: Maximum iterations for refinement (default: 3)
- `max_new_tokens`: Maximum tokens for response generation (default: 64)
- `feedback_tokens`: Maximum tokens for feedback (default: 128)
- `temperature`: Sampling temperature (default: 0.0)

## License

Apache License 2.0

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.

## Citation

This work has been inspired by:

```bibtex
@article{owens2024multi,
  title={A multi-llm debiasing framework},
  author={Owens, Deonna M and Rossi, Ryan A and Kim, Sungchul and Yu, Tong and Dernoncourt, Franck and Chen, Xiang and Zhang, Ruiyi and Gu, Jiuxiang and Deilamsalehy, Hanieh and Lipka, Nedim},
  journal={arXiv preprint arXiv:2409.13884},
  year={2024}
}
```
