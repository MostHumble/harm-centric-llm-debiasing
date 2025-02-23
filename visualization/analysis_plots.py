import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import numpy as np

def load_data(debiased_samples_path: str, harm_assignments_path: str):
    """Load and preprocess data for analysis"""
    with open(debiased_samples_path, 'r') as f:
        data = json.load(f)
    with open(harm_assignments_path, 'r') as f:
        harm_assignments = yaml.safe_load(f)
    return data, harm_assignments

def analyze_dataset(data, harm_assignments):
    """Generate various plots analyzing the debiasing process"""
    
    # 1. Number of iterations per query
    plt.figure(figsize=(10, 6))
    iterations = [len(sample['lineage']) for sample in data]
    sns.histplot(iterations)
    plt.title('Distribution of Iterations per Query')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Count')
    plt.savefig('iterations_distribution.png')
    plt.close()

    # 2. Most common harm types detected
    harm_counts = defaultdict(int)
    for sample in data:
        for feedback_round in sample['feedback']:
            for feedback in feedback_round:
                feedback_json = json.loads(feedback)
                for harm_type, assessment in feedback_json['analysis'].items():
                    if assessment != "none":
                        harm_counts[harm_type] += 1
    
    plt.figure(figsize=(12, 6))
    harm_types = list(harm_counts.keys())
    counts = list(harm_counts.values())
    plt.bar(harm_types, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title('Frequency of Detected Harm Types')
    plt.xlabel('Harm Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('harm_types_frequency.png')
    plt.close()

    # 3. Text length changes
    original_lengths = [len(sample['original_query'].split()) for sample in data]
    final_lengths = [len(sample['debiased_response'].split()) for sample in data]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(original_lengths, final_lengths, alpha=0.5)
    plt.plot([0, max(original_lengths)], [0, max(original_lengths)], 'r--')  # diagonal line
    plt.title('Text Length: Original vs Debiased')
    plt.xlabel('Original Length (words)')
    plt.ylabel('Debiased Length (words)')
    plt.savefig('length_comparison.png')
    plt.close()

    # 4. Model agreement analysis
    model_agreement = []
    for sample in data:
        for feedback_round in sample['feedback']:
            round_issues = defaultdict(set)
            for i, feedback in enumerate(feedback_round):
                feedback_json = json.loads(feedback)
                for harm_type, assessment in feedback_json['analysis'].items():
                    if assessment != "none":
                        round_issues[harm_type].add(i)
            agreement_scores = [len(models)/len(feedback_round) for models in round_issues.values()]
            if agreement_scores:
                model_agreement.extend(agreement_scores)
    
    plt.figure(figsize=(8, 6))
    sns.histplot(model_agreement, bins=20)
    plt.title('Model Agreement on Harm Detection')
    plt.xlabel('Agreement Score (0-1)')
    plt.ylabel('Count')
    plt.savefig('model_agreement.png')
    plt.close()

    # 5. Convergence analysis
    def get_text_similarity(text1, text2):
        """Simple word overlap similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        return len(words1.intersection(words2)) / len(words1.union(words2))

    convergence_rates = []
    for sample in data:
        similarities = []
        for i in range(1, len(sample['lineage'])):
            sim = get_text_similarity(sample['lineage'][i-1], sample['lineage'][i])
            similarities.append(sim)
        if similarities:
            convergence_rates.append(similarities)
    
    plt.figure(figsize=(10, 6))
    for rate in convergence_rates:
        plt.plot(range(1, len(rate)+1), rate, alpha=0.1, color='blue')
    plt.plot(range(1, max(len(r) for r in convergence_rates)+1), 
             [np.mean([r[i] for r in convergence_rates if len(r)>i]) for i in range(max(len(r) for r in convergence_rates))],
             'r-', linewidth=2, label='Average')
    plt.title('Convergence Analysis')
    plt.xlabel('Iteration')
    plt.ylabel('Similarity to Previous Version')
    plt.legend()
    plt.savefig('convergence_analysis.png')
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate analysis plots for debiasing results')
    parser.add_argument('--debiased-samples', type=str, required=True,
                       help='Path to debiased samples JSON file')
    parser.add_argument('--harm-assignments', type=str, required=True,
                       help='Path to harm assignments YAML file')
    args = parser.parse_args()
    
    data, harm_assignments = load_data(args.debiased_samples, args.harm_assignments)
    analyze_dataset(data, harm_assignments) 