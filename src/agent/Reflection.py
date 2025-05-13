from typing import Dict, Any, List, Optional
import json
from ollama import chat
from pydantic import BaseModel
import re

from src.agent.Core import AgentState
from src.models.LoadLLM import llm
from src.agent.Roles import get_role_prompt

class EvaluationMetrics(BaseModel):
    """Structured metrics for evaluating diagnosis quality"""
    factual_correctness: float  # 0-1 score
    completeness: float  # 0-1 score
    evidence_basis: float  # 0-1 score
    logical_coherence: float  # 0-1 score
    alternative_considerations: float  # 0-1 score
    overall_score: float  # 0-1 score
    improvements: List[str]  # Specific improvement suggestions
    
    def passed_threshold(self, threshold: float = 0.7) -> bool:
        """Check if the overall score passes a quality threshold"""
        return self.overall_score >= threshold


def create_critique_prompt(diagnosis: str) -> str:
    """Create a prompt for the critic agent to evaluate a diagnosis"""
    return f"""
Please evaluate the following medical diagnosis:

{diagnosis}

Analyze the diagnosis for:
1. Factual correctness (0-1 score)
2. Completeness (0-1 score)
3. Evidence basis (0-1 score)
4. Logical coherence (0-1 score)
5. Alternative considerations (0-1 score)

Respond with a JSON object containing:
- Scores for each category (as floats between 0-1)
- An overall_score (weighted average)
- An array of specific improvement suggestions

Format your response as valid JSON with these exact keys. DONT include any other text or explanations.:

{{
  "factual_correctness": float,
  "completeness": float,
  "evidence_basis": float,
  "logical_coherence": float,
  "alternative_considerations": float,
  "overall_score": float,
  "improvements": [list of strings]
}}
"""


def execute_reflection(state: AgentState) -> Dict:
    """Generate a reflection on the current diagnosis output"""
    print("\n🔍 Evaluating diagnosis quality...")
    
    # Extract the diagnosis from state
    diagnosis = state.get("output", {}).get("answer", "No diagnosis available")
    
    critique_prompt = create_critique_prompt(diagnosis)
    
    # Set up critic agent with proper system prompt
    system_prompt = get_role_prompt("critic")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": critique_prompt}
    ]
    
    try:
        response = llm._call(messages)
        content = re.sub(r'^```json\s*|\s*```$', '', response.strip(), flags=re.DOTALL)
        # print("Critique response:", content)
        # Parse the evaluation metrics
        try:
            
            metrics_data = json.loads(content)
            
            # Create evaluation metrics object
            metrics = EvaluationMetrics(
                factual_correctness=metrics_data.get("factual_correctness", 0.0),
                completeness=metrics_data.get("completeness", 0.0),
                evidence_basis=metrics_data.get("evidence_basis", 0.0),
                logical_coherence=metrics_data.get("logical_coherence", 0.0),
                alternative_considerations=metrics_data.get("alternative_considerations", 0.0),
                overall_score=metrics_data.get("overall_score", 0.0),
                improvements=metrics_data.get("improvements", ["No specific improvements provided"])
            )
            
            # Print evaluation for visibility
            print("\n📊 Evaluation Results:")
            print(f"✓ Factual Correctness: {metrics.factual_correctness:.2f}")
            print(f"✓ Completeness: {metrics.completeness:.2f}")
            print(f"✓ Evidence Basis: {metrics.evidence_basis:.2f}")
            print(f"✓ Logical Coherence: {metrics.logical_coherence:.2f}")
            print(f"✓ Alternative Considerations: {metrics.alternative_considerations:.2f}")
            print(f"✓ Overall Score: {metrics.overall_score:.2f}")
            print("\n🔧 Suggested Improvements:")
            for i, improvement in enumerate(metrics.improvements):
                print(f"{i+1}. {improvement}")
            
            # Update the state with reflection
            return {
                "reflection": {
                    "metrics": metrics.dict(),
                    "critique": content,
                    "needs_refinement": not metrics.passed_threshold(0.7)
                },
                "agent_role": "refiner" if not metrics.passed_threshold(0.7) else "executor",
                "chat_history": state["chat_history"] + [
                    {"role": "user", "content": critique_prompt},
                    {"role": "assistant", "content": content}
                ]
            }
            
        except json.JSONDecodeError as e:
            print(f"Error parsing critique response: {e}")
            # Create a fallback evaluation
            return create_fallback_reflection(state)
            
    except Exception as e:
        print(f"Error during reflection: {e}")
        return create_fallback_reflection(state)


def create_fallback_reflection(state: AgentState) -> Dict:
    """Create a fallback reflection when evaluation fails"""
    # Default fallback evaluation
    fallback_metrics = EvaluationMetrics(
        factual_correctness=0.5,
        completeness=0.5,
        evidence_basis=0.5,
        logical_coherence=0.5,
        alternative_considerations=0.5,
        overall_score=0.5,
        improvements=["Improve diagnosis with more detailed evidence.",
                      "Consider alternative diagnoses more thoroughly."]
    )
    
    print("\n⚠️ Using fallback evaluation:")
    print(f"Overall Score: {fallback_metrics.overall_score} (default)")
    
    return {
        "reflection": {
            "metrics": fallback_metrics.dict(),
            "critique": "Automated evaluation failed. Using default metrics.",
            "needs_refinement": True  # Default to refinement on failure
        },
        "agent_role": "refiner",  # Switch to refiner role
        "chat_history": state["chat_history"]  # Keep the existing chat history
    }


def execute_refinement(state: AgentState) -> Dict:
    """Refine the diagnosis based on reflection feedback"""
    print("\n🔄 Refining diagnosis based on feedback...")
    
    # Extract the original diagnosis and reflection
    diagnosis = state.get("output", {}).get("answer", "No diagnosis available")
    reflection = state.get("reflection", {})
    
    # Extract improvements from reflection
    improvements = reflection.get("metrics", {}).get("improvements", [])
    improvements_text = "\n".join([f"- {imp}" for imp in improvements])
    
    refinement_prompt = f"""
Please refine the following medical diagnosis based on the critique provided:

ORIGINAL DIAGNOSIS:
{diagnosis}

IMPROVEMENT AREAS:
{improvements_text}

Provide an improved diagnosis that addresses these critique points while maintaining accuracy.
The refined diagnosis should be complete, evidence-based, and logically coherent.
"""
    
    # Set up refiner agent with proper system prompt
    system_prompt = get_role_prompt("refiner")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": refinement_prompt}
    ]
    
    try:
        response = llm._call(messages)
        refined_diagnosis = response.strip()
        
        print("\n✍️ Refined Diagnosis Created")
        
        # Update the state with the refined output
        return {
            "output": {"answer": refined_diagnosis},
            "agent_role": "critic",  # Switch back to critic for re-evaluation
            "chat_history": state["chat_history"] + [
                {"role": "user", "content": refinement_prompt},
                {"role": "assistant", "content": refined_diagnosis}
            ]
        }
            
    except Exception as e:
        print(f"Error during refinement: {e}")
        # Keep the original output if refinement fails
        return {
            "agent_role": "executor",  # Switch back to executor
            "chat_history": state["chat_history"]  # Keep the existing chat history
        }