"""eegagent: an LLM agent that runs EEG source-imaging analyses through typed tools.

from eegagent import Agent, Session, get_provider
agent = Agent(get_provider("anthropic"), Session())
result = agent.run("Load the simulated recording, filter 1-30 Hz, epoch and localise the peak")
print(result.text)
"""

from .agent import Agent, AgentResult
from .providers import get_provider
from .session import Session

__all__ = ["Agent", "AgentResult", "Session", "get_provider"]
__version__ = "0.2.0"
