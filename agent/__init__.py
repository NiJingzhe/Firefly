from .BaseAgent import BaseAgent
from .SimpleAgent import SimpleAgent
from .AgentRegister import (
    AgentRegistry,
    get_agent_registry,
    register_agent,
    get_agent,
    list_available_models
)

from .Firefly import FireflyAgent

__all__ = [
    'BaseAgent',
    'SimpleAgent',
    'AgentRegistry',
    'get_agent_registry',
    'register_agent',
    'get_agent',
    'list_available_models',
]

register_agent(
    "simple-agent-v1",
    SimpleAgent
)

register_agent(
    "firefly-agent-v1",
    FireflyAgent
)

