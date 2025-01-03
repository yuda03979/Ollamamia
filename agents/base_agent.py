from datetime import datetime
from typing import Literal, Any

from pydantic import Field





class BaseAgent:

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.agent_task: str
        self.message: str

    def predict(self, query: Any):
        """ overwrite """
        pass





