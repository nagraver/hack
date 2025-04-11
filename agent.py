from PIL.Image import Image
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()

login(token=os.getenv("TOKEN"))

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())
agent.run("how many people are on the planet in 2025?")
