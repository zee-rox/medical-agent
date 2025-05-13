from langchain.llms.base import LLM
from ollama import chat
from openai import OpenAI
import os
import torch

# os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
class OllamaLLM():
    def __init__(self, model: str = "llama3.2:3b"):
        self.client = OpenAI(

        )
        self.model = "deepseek/deepseek-chat-v3-0324:free"
        # self.model = model
    
    def _call(self, prompt):
        """Make a direct call to the Ollama API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
        )
        
        # response = chat(
        #     model=self.model,
        #     messages=prompt
        # )
        
        # try:
        #     return response.get("message", {}).get("content", "❌ No response generated.")
        
        # except (KeyError, IndexError, TypeError):
        #     return "❌ No response generated."
        try:
            return response.choices[0].message.content
        except (KeyError, IndexError, TypeError):
            return "No response generated."
    
    @property
    def _llm_type(self) -> str:
        return "ollama_llama3.2"
    
# Initialize LLM
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔄 Using device: {device}")

llm = OllamaLLM(model="llama3.2:3b")