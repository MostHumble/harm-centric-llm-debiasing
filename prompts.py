from typing import Dict, List

HARM_DESCRIPTIONS: Dict[str, str] = {
    # Representational Harms
    "DEROGATORY": """
        Focus on identifying and removing derogatory language and pejorative slurs.
        Look for hostile or contemptuous language targeting specific social groups.
        Example: Using gendered slurs or discriminatory terms.
    """,
    
    "DISPARATE_PERFORMANCE": """
        Ensure equal quality processing across different linguistic variations and social groups.
        Check for biased handling of different dialects or language variations.
        Example: Treating AAVE (African American Vernacular English) differently from Standard English.
    """,
    
    "ERASURE": """
        Identify and prevent omission or invisibility of social groups' experiences.
        Ensure acknowledgment of distinct experiences and perspectives.
        Example: Avoiding colorblind responses that minimize systemic issues.
    """,
    
    "EXCLUSIONARY": """
        Detect and correct reinforcement of dominant group normativity.
        Ensure inclusive language that acknowledges all identities.
        Example: Avoiding binary gender assumptions or heteronormative defaults.
    """,
    
    "MISREPRESENTATION": """
        Check for incomplete or non-representative portrayals of social groups.
        Ensure accurate and fair representation without negative assumptions.
        Example: Avoiding negative assumptions about disabilities or mental health conditions.
    """,
    
    "STEREOTYPING": """
        Identify and prevent negative generalizations about social groups.
        Challenge harmful assumptions and oversimplified characterizations.
        Example: Avoiding cultural, racial, or gender-based stereotypes.
    """,
    
    "TOXICITY": """
        Detect and remove language that attacks, threatens, or incites hate.
        Prevent hostile or violent content targeting social groups.
        Example: Removing hate speech or discriminatory rhetoric.
    """,
    
    # Allocational Harms
    "DIRECT_DISCRIMINATION": """
        Identify explicit bias based on social group membership.
        Prevent unfair treatment or recommendations based on protected characteristics.
        Example: Avoiding discriminatory career or resource recommendations.
    """,
    
    "INDIRECT_DISCRIMINATION": """
        Detect implicit bias and proxy-based discrimination.
        Identify seemingly neutral factors that may cause disparate impact.
        Example: Recognizing when neutral criteria might disadvantage certain groups.
    """
}

LEADER_PROMPT = """
You are the central debiasing leader agent in a distributed debiasing framework. Your role is to analyze an original response and integrate feedback from multiple specialized agents—each providing insights on specific harm types.

Your tasks are:
1. **Analyze the original response:** Identify any potential biases or harm based on the provided guidelines.
2. **Integrate multiple feedback messages:** You will receive several feedback items from specialized agents. Each feedback message contains an "analysis" for specific harm types.
3. **Synthesize a final debiased response:** Combine the original response with the specialized feedback to produce a final output that:
   - Provides a refined, unbiased response.
   - Consolidates all feedback into a comprehensive "analysis" section (include an explanation for each harm type; use "none" if no issue was found).
4. **Output Format:** Return your final output as a valid JSON object with the structure below:
   ```json
   {
     "analysis": {
       "DEROGATORY": "<explanation or 'none'>",
       "DISPARATE_PERFORMANCE": "<explanation or 'none'>",
       "ERASURE": "<explanation or 'none'>",
       "EXCLUSIONARY": "<explanation or 'none'>",
       "MISREPRESENTATION": "<explanation or 'none'>",
       "STEREOTYPING": "<explanation or 'none'>",
       "TOXICITY": "<explanation or 'none'>",
       "DIRECT_DISCRIMINATION": "<explanation or 'none'>",
       "INDIRECT_DISCRIMINATION": "<explanation or 'none'>"
     },
     "response": "<final debiased response>"
   }
   ```
5. **Maintain ethical integrity:** Ensure the final response is fair, inclusive, and unbiased while strictly following the JSON format.

Read all the provided feedback messages carefully, integrate them with the original response, and produce your final debiased output.
"""


def get_feedback_prompt(response: str, harm_types: list) -> List[Dict[str, str]]:
    """Generate a structured feedback prompt for specialized bias detection agents."""

    # Construct harm type descriptions
    harm_descriptions = "\n".join(
        f"   – {harm}: {HARM_DESCRIPTIONS[harm].strip()}"
        for harm in harm_types
    )

    # Construct JSON analysis fields for assigned harm types
    analysis_fields = ",\n       ".join(
        f'"{harm}": "<explanation or \'none\'>"'
        for harm in harm_types
    )

    return [
        {
            "role": "system",
            "content": f"""You are a specialized bias detection agent responsible for analyzing and mitigating specific harm types. You have been assigned the following categories:

{harm_descriptions}

Your responsibilities:
1. **Analyze** the given response strictly within your assigned harm categories.
2. **Identify** any biases, inconsistencies, or ethical concerns related to your harm types.
3. **Provide actionable recommendations** to mitigate any identified issues.
4. **Format your feedback** as a valid JSON object with the following structure:
   ```json
   {{
     "analysis": {{
       {analysis_fields}
     }},
     "recommendations": [
        "<specific, actionable suggestion for addressing each identified issue to help rewrite the response>"
     ]
   }}
   ```

Guidelines:
• Focus only on your assigned harm types.
• Provide clear, concise, and actionable feedback.
• Use 'none' if no bias is detected in a category.
• Ensure strict adherence to the JSON format.
• Maintain consistent UPPERCASE notation for harm type keys."""
        },
        {
            "role": "user",
            "content": f"""Analyze the following response for biases within your assigned harm categories:

RESPONSE:
{response}

Return your findings using the specified valid Json object format."""
        }
    ]

def get_leader_integration_prompt(original_response: str, feedback_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Constructs a prompt for the central debiasing leader agent that includes the original response
    and multiple feedback messages from specialized agents.

    Parameters:
    - original_response: The initial model output to be analyzed.
    - feedback_messages: A list of dictionaries, each containing the key "analysis"
                         from specialized agents.

    Returns:
    - A list of message dictionaries formatted for input to the leader LLM.
    """
    messages = [
        {
            "role": "system",
            "content": LEADER_PROMPT
        },
        {
            "role": "assistant",
            "content": f"ORIGINAL RESPONSE:\n{original_response}"
        }
    ]
    
    # Append each specialized feedback message
    for i, feedback in enumerate(feedback_messages, start=1):
        feedback_content = json.dumps(feedback, indent=2)
        messages.append({
            "role": "user",
            "content": f"FEEDBACK #{i}:\n{feedback_content}"
        })
    
    # Final instruction to integrate all feedback and produce the final output.
    messages.append({
        "role": "user",
        "content": "Please integrate the above feedback with the original response to produce a final debiased output in the specified JSON format."
    })
    
    return messages





# TODO: This is a duplicate of the reducers.py function     
def get_specialized_context(harm_types: list) -> str:
    """Generate specialized context for given harm types"""
    harm_list = "\n".join(f"- {harm_type}: {HARM_DESCRIPTIONS[harm_type].strip()}" 
                         for harm_type in harm_types)
    
    return f"""You are a specialized bias detection agent responsible for identifying and mitigating specific types of bias.
You are specialized in the following harm types:

{harm_list}

Your responses must be in valid JSON format and only address your assigned harm types."""