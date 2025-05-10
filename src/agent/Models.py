from typing import List, TypedDict, Union, Dict, Any, Optional
from pydantic import BaseModel

class AgentAction(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: Union[str, None] = None

    @classmethod
    def from_ollama(cls, ollama_response: dict):
        try:
            import json
            # content = ollama_response.get("message", {}).get("content", "").strip()
            content = ollama_response
            try:
                output = json.loads(content)
                if "name" in output and "parameters" in output and output["parameters"].get("answer", "").strip():
                    return cls(
                        tool_name=output["name"],
                        tool_input=output["parameters"],
                        tool_output=output["parameters"].get("answer")
                    )
                else:
                    return cls(
                        tool_name="final_answer",
                        tool_input={"answer": content},
                        tool_output=content
                    )
            except Exception as e:
                return cls(
                    tool_name="final_answer",
                    tool_input={"answer": content},
                    tool_output=content
                )
        except Exception as e:
            print(f"Error parsing ollama response:\n{ollama_response}\n")
            raise e

    def __str__(self):
        text = f"Tool: {self.tool_name}\nInput: {self.tool_input}"
        if self.tool_output is not None:
            text += f"\nOutput: {self.tool_output}"
        return text


def action_to_message(action: AgentAction):
    import json
    assistant_content = json.dumps({"name": action.tool_name, "parameters": action.tool_input})
    assistant_message = {"role": "assistant", "content": assistant_content, "role_tag": "executor"}
    user_message = {"role": "user", "content": action.tool_output if action.tool_output is not None else "", "role_tag": "system"}
    return [assistant_message, user_message]


def create_scratchpad(intermediate_steps: List[AgentAction]):
    scratch_pad_messages = []
    for action in intermediate_steps:
        if action.tool_output is not None:
            scratch_pad_messages.extend(action_to_message(action))
    return scratch_pad_messages


class Plan(BaseModel):
    """Represents a plan created by the agent"""
    steps: List[str]
    reasoning: str
    current_step_index: int = 0
    
    def get_current_step(self) -> Optional[str]:
        """Get the current step in the plan"""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    def advance(self) -> bool:
        """Advance to the next step in the plan. Returns True if successful, False if at the end."""
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            return True
        return False
    
    def is_complete(self) -> bool:
        """Check if the plan is complete"""
        return self.current_step_index >= len(self.steps) - 1


class AgentState(TypedDict):
    """Enhanced agent state that tracks all components needed for the agent workflow"""
    input: str  # The original user query
    chat_history: List[dict]  # Full conversation history
    intermediate_steps: List[AgentAction]  # All actions taken during execution
    output: dict  # The final output of the agent
    plan: Optional[Plan]  # The current execution plan
    reflection: Optional[Dict[str, Any]]  # Reflections on actions and outputs
    agent_role: str  # Current active agent role
    agent_outcome: Optional[Dict[str, Any]]  # Outcome evaluation