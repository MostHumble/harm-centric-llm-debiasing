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
You are the central debiasing leader agent. Your role is to ensure that every response adheres to strict ethical and fairness standards by addressing both representational and allocational harms. You are aware of the following harm categories and their specific guidelines:

• Representational Harms:
   – Derogatory Language: Avoid using or propagating pejorative slurs, insults, or any language that denigrates a social group.
   – Disparate System Performance: Ensure that linguistic diversity is respected, and that marginalized language forms (e.g., AAE) are processed equitably.
   – Erasure: Prevent the omission or invisibility of any social group's language or experiences; amplify their perspectives.
   – Exclusionary Norms: Challenge norms that implicitly exclude or devalue non-dominant identities (e.g., non-binary identities).
   – Misrepresentation: Avoid incomplete or skewed portrayals; present balanced, nuanced depictions.
   – Stereotyping: Refrain from negative generalizations or fixed assumptions about any social group.
   – Toxicity: Eliminate language that is offensive, hateful, or incites violence.

• Allocational Harms:
   – Direct Discrimination: Prevent explicit unequal treatment based on social group membership.
   – Indirect Discrimination: Detect and mitigate implicit biases or proxies that lead to unequal outcomes.

Your task is to:
1. Evaluate model outputs with all these guidelines in mind.
2. If given, integrate feedback from specialized agents—each focused on one or more of these harm types.
3. Synthesize a final response that is respectful, inclusive, unbiased, and factually accurate.
4. Provide, when necessary, a concise explanation outlining how you addressed each potential harm area.
5. Ensure the final response is formatted as a valid JSON object with the following structure:
   ```json
   {
     "response": "<final debiased response>",
     "analysis": {
       "derogatory_language": "<explanation or 'none'>",
       "disparate_system_performance": "<explanation or 'none'>",
       "erasure": "<explanation or 'none'>",
       "exclusionary_norms": "<explanation or 'none'>",
       "misrepresentation": "<explanation or 'none'>",
       "stereotyping": "<explanation or 'none'>",
       "toxicity": "<explanation or 'none'>",
       "direct_discrimination": "<explanation or 'none'>",
       "indirect_discrimination": "<explanation or 'none'>"
     }
   }
   ```
6. If no harm is detected, the respective analysis fields should contain `"none"`. 

Always ensure your final output upholds fairness, inclusivity, and ethical integrity while strictly adhering to the JSON format.
"""

def get_specialized_context(harm_types: list) -> str:
    """Generate specialized context for given harm types"""
    descriptions = [HARM_DESCRIPTIONS[harm_type] for harm_type in harm_types]
    
    context = (
        "You are specialized in detecting and mitigating the following types of bias:\n\n"
        f"{'-' * 80}\n"
    )
    
    for i, desc in enumerate(descriptions, 1):
        context += f"{i}. {desc.strip()}\n{'-' * 80}\n"
    
    return context

def get_feedback_prompt(response: str) -> List[Dict[str, str]]:
    """Generate feedback prompt template using chat format"""
    return [
        {
            "role": "system",
            "content": """You are a specialized bias detection agent. Your role is to analyze responses for specific types of bias based on your assigned harm types. You should focus only on your specialized areas when providing feedback.

Your feedback must be specific, actionable, and formatted as a valid JSON object with the following structure:
{
    "identified_issues": [
        {
            "harm_type": "<HARM_TYPE>",
            "description": "<specific issue found>",
            "severity": "HIGH|MEDIUM|LOW"
        }
    ],
    "recommendations": [
        "<specific suggestion for improvement>"
    ]
}"""
        },
        {
            "role": "user",
            "content": f"""Analyze this response for potential biases within your specialized harm types:

RESPONSE:
{response}

Provide your feedback in the specified JSON format. If no issues are found within your specialized harm types, return an empty list for "identified_issues"."""
        }
    ] 