import json
import yaml
from pathlib import Path
import argparse
import streamlit as st
import difflib
from html import escape

def parse_args():
    parser = argparse.ArgumentParser(description='Debiasing Visualization Tool')
    parser.add_argument('--debiased-samples', type=str, required=True,
                       help='Path to debiased samples JSON file')
    parser.add_argument('--harm-assignments', type=str, required=True,
                       help='Path to harm assignments YAML file')
    return parser.parse_args()

def load_data(debiased_samples_path: str, harm_assignments_path: str):
    """Load data from specified paths"""
    try:
        # Load JSON data
        with open(debiased_samples_path, 'r') as f:
            data = json.load(f)
        
        # Load harm assignments
        with open(harm_assignments_path, 'r') as f:
            harm_assignments = yaml.safe_load(f)
        
        return data, harm_assignments
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info(f"""Please check the paths to your files:
        - Debiased samples: {debiased_samples_path}
        - Harm assignments: {harm_assignments_path}
        """)
        raise

def get_leader_model(harm_assignments):
    """Identify the leader model (one with empty harm_types)"""
    for model, config in harm_assignments.items():
        if not config.get('harm_types'):
            return model
    return "Unknown Leader"

def get_follower_models(harm_assignments):
    """Get the list of follower models"""
    return [model for model, config in harm_assignments.items() if config.get('harm_types')]

def generate_diff_html(text1, text2):
    """Generate HTML diff between two texts"""
    diff = difflib.ndiff(text1.split(), text2.split())
    
    html_parts = []
    for line in diff:
        if line.startswith('+ '):
            html_parts.append(f'<span style="background-color: rgba(0, 255, 0, 0.3); padding: 2px 4px; border-radius: 3px;">{escape(line[2:])}</span>')
        elif line.startswith('- '):
            html_parts.append(f'<span style="background-color: rgba(255, 0, 0, 0.3); padding: 2px 4px; border-radius: 3px;">{escape(line[2:])}</span>')
        elif line.startswith('  '):
            html_parts.append(escape(line[2:]))
    
    return " ".join(html_parts)

def analyze_feedback(feedback_list, follower_models):
    """Extract and organize feedback by model and harm type"""
    organized_feedback = {}
    
    for iteration_idx, iteration_feedback in enumerate(feedback_list):
        for model_idx, model_feedback in enumerate(iteration_feedback):
            if model_idx < len(follower_models):
                model_name = follower_models[model_idx]
                feedback_json = json.loads(model_feedback)
                
                # Initialize model entry if it doesn't exist
                if model_name not in organized_feedback:
                    organized_feedback[model_name] = {}
                
                # Organize feedback by harm type
                for harm_type, assessment in feedback_json['analysis'].items():
                    if assessment != "none":
                        if harm_type not in organized_feedback[model_name]:
                            organized_feedback[model_name][harm_type] = []
                        
                        # Add feedback with iteration info
                        organized_feedback[model_name][harm_type].append({
                            "iteration": iteration_idx + 1,
                            "assessment": assessment,
                            "recommendations": [rec for rec in feedback_json['recommendations'] if harm_type in rec]
                        })
    
    return organized_feedback

def main():
    # Parse command line arguments
    args = parse_args()
    
    st.set_page_config(layout="wide", page_title="Debiasing Visualization Tool")
    
    st.title("Debiasing Process Visualization")
    st.write("This tool visualizes how queries evolve through a centralized debiasing approach.")
    
    # Load data with paths from arguments
    try:
        data, harm_assignments = load_data(args.debiased_samples, args.harm_assignments)
        leader_model = get_leader_model(harm_assignments)
        follower_models = get_follower_models(harm_assignments)
        
        st.success(f"Data loaded successfully. Leader model: {leader_model}")
        
        # Select example to visualize
        example_indices = [f"Example {i+1}: {entry['original_query'][:50]}..." for i, entry in enumerate(data)]
        selected_example = st.selectbox("Select example to analyze:", example_indices)
        example_idx = int(selected_example.split(':')[0].replace('Example ', '')) - 1
        
        # Get selected example data
        example = data[example_idx]
        
        # Display before and after with diff highlighting
        st.header("Before & After Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Query")
            st.write(example['original_query'])
        
        with col2:
            st.subheader("Debiased Result")
            st.write(example['debiased_response'])
        
        # Visualize the diff
        st.header("Changes Visualization")
        diff_html = generate_diff_html(example['original_query'], example['debiased_response'])
        st.markdown(f'<div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px;">{diff_html}</div>', unsafe_allow_html=True)
        
        # Analyze feedback from different models
        organized_feedback = analyze_feedback(example['feedback'], follower_models)
        
        # Display lineage (evolution of query)
        st.header("Query Evolution")
        tabs = st.tabs([f"Iteration {i+1}" for i in range(len(example['lineage']))])
        
        for i, (tab, query_version) in enumerate(zip(tabs, example['lineage'])):
            with tab:
                if i > 0:
                    st.markdown("#### Changes from previous version:")
                    diff_html = generate_diff_html(example['lineage'][i-1], query_version)
                    st.markdown(f'<div style="padding: 10px; border: 1px solid #eee; border-radius: 5px;">{diff_html}</div>', unsafe_allow_html=True)
                
                st.markdown("#### Complete text in this iteration:")
                st.write(query_version)
                
                # Show feedback that led to this iteration
                if i < len(example['feedback']):
                    st.markdown("#### Feedback provided after this version:")
                    feedback_cols = st.columns(len(example['feedback'][i]))
                    
                    for j, (col, feedback_item) in enumerate(zip(feedback_cols, example['feedback'][i])):
                        with col:
                            model_name = follower_models[j] if j < len(follower_models) else f"Model {j+1}"
                            st.markdown(f"**Feedback from {model_name}:**")
                            
                            feedback_json = json.loads(feedback_item)
                            
                            # Display harm analysis
                            st.markdown("**Detected issues:**")
                            detected_issues = [harm for harm, assessment in feedback_json['analysis'].items() if assessment != "none"]
                            
                            if detected_issues:
                                for issue in detected_issues:
                                    st.markdown(f"- {issue}: {feedback_json['analysis'][issue]}")
                            else:
                                st.markdown("- No issues detected")
                            
                            # Display recommendations
                            st.markdown("**Recommendations:**")
                            for rec in feedback_json['recommendations']:
                                st.markdown(f"- {rec}")
        
        # Summary of Harm Types Addressed
        st.header("Summary of Biases Addressed")
        
        if organized_feedback:
            for model_name, harm_types in organized_feedback.items():
                st.subheader(f"Feedback from {model_name}")
                
                for harm_type, feedback_items in harm_types.items():
                    with st.expander(f"{harm_type} ({len(feedback_items)} mentions)"):
                        for item in feedback_items:
                            st.markdown(f"**Iteration {item['iteration']}:**")
                            st.markdown(f"*Assessment:* {item['assessment']}")
                            
                            if item['recommendations']:
                                st.markdown("*Recommendations:*")
                                for rec in item['recommendations']:
                                    st.markdown(f"- {rec}")
                            st.divider()
        else:
            st.write("No biases were detected and addressed in this example.")
            
    except Exception as e:
        st.error(f"Error in visualization process: {str(e)}")
        st.info("""
        Usage:
        streamlit run streamlit_app.py -- --debiased-samples path/to/samples.json --harm-assignments path/to/config.yaml
        """)

if __name__ == "__main__":
    main() 