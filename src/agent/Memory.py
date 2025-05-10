import json
from typing import Union
import os

class ConfigSpec:
    def __init__(self, id: str, description: str = ""):
        self.id = id
        self.description = description

class CustomMemorySaver:
    """
    A custom memory saver that persists the agent's state to a JSON file.
    """
    config_specs = [ConfigSpec(id="filename", description="Path to the agent memory file.")]
    
    def __init__(self, filename="agent_memory.json"):
        self.filename = filename

    @property
    def config(self):
        return {"filename": self.filename}

    def save(self, state: dict) -> None:
        with open(self.filename, "w") as f:
            json.dump(state, f)

    def load(self) -> Union[dict, None]:
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                return json.load(f)
        return None