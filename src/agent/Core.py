from typing import Optional
from langgraph.graph import StateGraph, END
import re
import os
import datetime
import base64
import io
import requests
from PIL import Image as im

from src.agent.Models import AgentAction, Plan, AgentState, create_scratchpad
from src.agent.Planning import execute_planning
from src.schema.Tools import search_schema, final_answer_schema, xray_detection_schema, get_system_tools_prompt
from src.imaging.DetectXRAY import detect_chest_xray
from src.models.LoadLLM import llm
from src.agent.Roles import get_role_prompt, get_role_name
from src.agent.Reflection import execute_reflection, execute_refinement

from src.retrieval.Search import search, final_answer

# Helper functions for role-based prompts
def get_system_prompt_with_plan(agent_role: str, plan: Optional[Plan] = None) -> str:
    """Get system prompt for an agent role, including plan context if available"""
    base_prompt = get_role_prompt(agent_role)
    
    if plan and agent_role == "executor":
        current_step = plan.get_current_step()
        step_context = f"\nCURRENT PLAN STEP: {current_step}\n"
        plan_context = "\nPLAN STEPS:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan.steps)])
        return f"{base_prompt}\n{plan_context}\n{step_context}"
    
    return base_prompt


# Graph node implementations

def call_llm_with_history(state: AgentState) -> AgentAction:
    """Call LLM with appropriate context and tools based on agent role"""
    user_input = state["input"]
    chat_history = state["chat_history"]
    intermediate_steps = state["intermediate_steps"]
    agent_role = state.get("agent_role", "executor")
    plan = state.get("plan")
    
    scratchpad = create_scratchpad(intermediate_steps)
    
    # Create a role-appropriate continuation message
    if scratchpad:
        if plan and not plan.is_complete():
            current_step = plan.get_current_step()
            step_context = f"You are on step: \"{current_step}\". "
            scratchpad.append({
                "role": "user",
                "content": f"{step_context}Please continue. My original query was: '{user_input}'. Use the appropriate tools to complete this step."
            })
        else:
            scratchpad.append({
                "role": "user",
                "content": f"Please continue. My original query was: '{user_input}'. Use all the provided information and proceed with the analysis."
            })
    
    # Choose tools based on agent role
    tools = [search_schema, final_answer_schema, xray_detection_schema]
    
    # Get role-appropriate system prompt
    role_system_prompt = get_system_prompt_with_plan(agent_role, plan)
    
    messages = [
        {"role": "system", "content": get_system_tools_prompt(role_system_prompt, tools), "role_tag": "system"},
        *chat_history,
        {"role": "user", "content": user_input, "role_tag": "user"},
        *scratchpad,
    ]
    
    print(f"\n📝 LLM Invocation - Role: {get_role_name(agent_role)}")
    
    res = llm._call([{k: v for k, v in m.items() if k != "role_tag"} for m in messages])
    content = re.sub(r'^```json\s*|\s*```$', '', res.strip(), flags=re.DOTALL)
    # print("LLM response:", content)
    
    return AgentAction.from_ollama(content)


def handle_tool_error(state: AgentState) -> dict:
    error_message = "Error encountered during tool execution. Please check the input and try again."
    fallback_action = AgentAction(
        tool_name="final_answer",
        tool_input={"answer": error_message},
        tool_output=error_message
    )
    return {"intermediate_steps": state["intermediate_steps"] + [fallback_action]}


# LangGraph node functions

def run_planner(state: AgentState) -> dict:
    """Planning node that creates the execution plan"""
    return execute_planning(state)


def run_oracle(state: AgentState) -> dict:
    """Oracle node that performs the core agent action"""
    print(f"run_oracle with role: {state.get('agent_role', 'executor')}")
    try:
        action = call_llm_with_history(state)
        
        # If we have a plan and just completed a step, advance the plan
        plan = state.get("plan")
        if plan and action.tool_name == "final_answer" and not plan.is_complete():
            plan.advance()
            return {
                "intermediate_steps": state["intermediate_steps"] + [action],
                "plan": plan
            }
        
        return {"intermediate_steps": state["intermediate_steps"] + [action]}
    except Exception as e:
        print(f"Error in oracle: {str(e)}")
        return handle_tool_error(state)


def run_critic(state: AgentState) -> dict:
    """Critic node that evaluates the agent's output"""
    return execute_reflection(state)


def run_refiner(state: AgentState) -> dict:
    """Refiner node that improves the agent's output based on feedback"""
    return execute_refinement(state)


def process_input(state: AgentState) -> dict:
    """Process the input and set up initial state"""
    # Initialize any missing state components
    update = {}
    
    if "agent_role" not in state:
        update["agent_role"] = "planner"
        
    if "intermediate_steps" not in state or not state["intermediate_steps"]:
        update["intermediate_steps"] = []
        
    if "chat_history" not in state:
        update["chat_history"] = []
        
    if "output" not in state:
        update["output"] = {}
        
    if "reflection" not in state:
        update["reflection"] = None
        
    if "agent_outcome" not in state:
        update["agent_outcome"] = None
        
    if "plan" not in state:
        update["plan"] = None
        
    return update


def router(state: AgentState) -> str:
    """Enhanced router that determines next node based on agent state"""
    print("router")
    
    # If we don't have a plan yet, go to planning
    if state.get("plan") is None:
        return "planner"
    
    # If we have a final answer in intermediate steps
    if state["intermediate_steps"]:
        last_action = state["intermediate_steps"][-1]
        
        # If we just got a final answer and haven't evaluated it yet
        if last_action.tool_name == "final_answer" and state.get("reflection") is None:
            if last_action.tool_output and last_action.tool_output.strip():
                return "critic"  # Evaluate the answer
            else:
                return "oracle"  # Try again
                
        # Route based on agent role after evaluation/refinement
        agent_role = state.get("agent_role", "executor")
        
        if agent_role == "refiner":
            return "refiner"
        elif agent_role == "critic":
            return "critic"
        elif last_action.tool_name != "final_answer":
            # Continue with relevant tool
            return last_action.tool_name
        else:
            # We're complete
            return "final_answer"
    
    # Default to oracle for next action
    return "oracle"


tool_str_to_func = {
    "search": search,
    "final_answer": final_answer,
    "detect_chest_xray": detect_chest_xray
}


def run_tool(state: AgentState) -> dict:
    """Run a tool based on agent action"""
    try:
        tool_name = state["intermediate_steps"][-1].tool_name
        tool_args = state["intermediate_steps"][-1].tool_input
        print(f"run_tool | {tool_name}.invoke(input={tool_args})")
        
        out = tool_str_to_func[tool_name](**tool_args)
        
        action_out = AgentAction(
            tool_name=tool_name,
            tool_input=tool_args,
            tool_output=str(out)
        )
        
        if tool_name == "final_answer":
            return {"output": out}
        else:
            return {"intermediate_steps": state["intermediate_steps"] + [action_out]}
    except Exception as e:
        print(f"Tool execution error: {str(e)}")
        return handle_tool_error(state)


# Build the agent graph
def build_graph() -> StateGraph:
    """Build the enhanced agent graph with all components"""
    graph = StateGraph(AgentState)
    
    # Add all nodes
    graph.add_node("process_input", process_input)
    graph.add_node("planner", run_planner)
    graph.add_node("oracle", run_oracle)
    graph.add_node("critic", run_critic)
    graph.add_node("refiner", run_refiner)
    graph.add_node("search", run_tool)
    graph.add_node("final_answer", run_tool)
    graph.add_node("detect_chest_xray", run_tool)
    
    # Set entry point
    graph.set_entry_point("process_input")
    
    # Define path maps for conditional routing
    # Each string returned by router maps to a target node
    conditional_path_map = {
        "planner": "planner",
        "oracle": "oracle",
        "critic": "critic",
        "refiner": "refiner", 
        "search": "search",
        "final_answer": "final_answer",
        "detect_chest_xray": "detect_chest_xray"
    }
    
    # Add conditional edges based on router
    graph.add_edge("process_input", "planner")
    graph.add_conditional_edges("planner", router, conditional_path_map)
    graph.add_conditional_edges("oracle", router, conditional_path_map)
    graph.add_conditional_edges("critic", router, conditional_path_map)
    graph.add_conditional_edges("refiner", router, conditional_path_map)
    
    # Tool routing
    for tool_obj in [search_schema, final_answer_schema, xray_detection_schema]:
        tool_name = tool_obj["function"]["name"]
        if tool_name != "final_answer":
            graph.add_conditional_edges(tool_name, router, conditional_path_map)
    
    # Final edge
    graph.add_edge("final_answer", END)
    
    return graph

def ConvertMermaidStringtoGraph(graph, output_path):
    graphbytes = graph.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    url = 'https://mermaid.ink/img/' + base64_string

    response = requests.get(url)

    if 'image' not in response.headers.get('Content-Type', ''):
        raise ValueError(f"Expected image, got {response.headers.get('Content-Type')}. Response content: {response.text[:200]}")

    img = im.open(io.BytesIO(response.content))
    img.save(output_path, dpi=(300, 300))
    
def save_graph_visualization(graph, output_dir: str = f"{os.getcwd()}/data/visualizations") -> None:
    """Save the graph visualization to a file"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"graph_visualization_{timestamp}.png")
        mm_string = graph.get_graph().draw_mermaid()
        mm_string = re.sub(r'^---[\s\S]+?---\s*', '', mm_string, flags=re.DOTALL)
        # print(mm_string)
        graph = ConvertMermaidStringtoGraph(mm_string, output_path)
        print(f"Graph visualization saved to {output_path}")
    except Exception as e:
        print(f"Failed to save graph visualization: {str(e)}")