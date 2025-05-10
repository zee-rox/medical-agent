search_schema = {
    "function": {
        "name": "search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
}

final_answer_schema = {
    "function": {
        "name": "final_answer",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"]
        }
    }
}

xray_detection_schema = {
    "function": {
        "name": "detect_chest_xray",
        "parameters": {
            "type": "object",
            "properties": {"image": {"type": "string"}},
            "required": ["image"]
        }
    }
}

def get_system_tools_prompt(system_prompt: str, tools: list) -> str:
    tools_str = "\n".join([str(tool) for tool in tools])
    return f"{system_prompt}\nTools:\n{tools_str}"

system_prompt = (
    "You are the oracle, the great AI clinical diagnosis decision maker. "
    "Given the clinical context, patient history data, and chest X‑ray findings (if provided), "
    "generate a concise, evidence‑based final diagnosis report. "
    "Return your answer as a JSON object with a single key: 'answer' (required, non‑empty). "
    "When using a tool, output the tool name and its parameters in JSON format as follows:\n"
    '{ "name": "<tool_name>", "parameters": { "<key>": <value>, ... } }'
)