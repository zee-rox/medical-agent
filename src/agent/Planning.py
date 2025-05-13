from typing import Dict, List, Optional
import json
from ollama import chat
from pydantic import BaseModel
import re

from src.agent.Models import Plan, AgentState
from src.models.LoadLLM import llm
from src.agent.Roles import get_role_prompt

class PlanningOutput(BaseModel):
    """Output of the planning process"""
    plan: Plan
    reasoning: str

def create_planning_prompt(query: str) -> str:
    """Create a planning prompt from the user's query"""
    return f"""
Please create a medical diagnosis plan for the following request:

{query}

Respond with a JSON object that includes:
1. "steps" (array): A step-by-step plan for diagnosing this medical case
2. "reasoning" (string): Your reasoning for this plan structure

The output format MUST be. DONT include any other text or explanations.
{{
  "steps": [
    "Step 1: ...",
    "Step 2: ..."
  ],
  "reasoning": "This plan is structured to ensure a comprehensive diagnosis."
}}

Format your response as valid JSON with these exact keys.
"""

def execute_planning(state: AgentState) -> Dict:
    """Create an execution plan for the agent"""
    print("\n🧠 Creating diagnosis plan...")
    
    planning_prompt = create_planning_prompt(state["input"])
    
    # Set up planning agent with proper system prompt
    system_prompt = get_role_prompt("planner")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": planning_prompt}
    ]
    
    try:
        response = llm._call(messages)
        content = re.sub(r'^```json\s*|\s*```$', '', response.strip(), flags=re.DOTALL)
        print("Diagnosis plan response:", content)
        
        # Parse the plan from the response
        try:
            plan_data = json.loads(content)
            
            # Create a Plan object
            plan = Plan(
                steps=plan_data.get("steps", []),
                reasoning=plan_data.get("reasoning", "No reasoning provided"),
                current_step_index=0
            )
            
            # Print the plan for visibility
            print("\n📋 Diagnosis Plan:")
            for i, step in enumerate(plan.steps):
                print(f"{i+1}. {step}")
            print(f"\nReasoning: {plan.reasoning}")
            
            # Update the state with the new plan
            return {
                "plan": plan,
                "agent_role": "executor",  # Switch to executor role after planning
                "chat_history": state["chat_history"] + [
                    {"role": "user", "content": planning_prompt},
                    {"role": "assistant", "content": content}
                ]
            }
            
        except json.JSONDecodeError as e:
            print(f"Error parsing planning response: {e}")
            # Create a fallback plan
            return create_fallback_plan(state)
            
    except Exception as e:
        print(f"Error during planning: {e}")
        return create_fallback_plan(state)

def create_fallback_plan(state: AgentState) -> Dict:
    """Create a fallback plan when planning fails"""
    # Default fallback plan
    fallback_plan = Plan(
        steps=[
            "Understand the patient's symptoms and medical history",
            "Search for relevant medical information",
            "Analyze the available data",
            "Generate a diagnosis based on the evidence",
            "Provide a final diagnostic report"
        ],
        reasoning="This is a default diagnostic process when detailed planning failed.",
        current_step_index=0
    )
    
    print("\n⚠️ Using fallback diagnosis plan:")
    for i, step in enumerate(fallback_plan.steps):
        print(f"{i+1}. {step}")
    
    return {
        "plan": fallback_plan,
        "agent_role": "executor",  # Switch to executor role
        "chat_history": state["chat_history"]  # Keep the existing chat history
    }