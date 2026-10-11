from typing import Any, Dict, List, Optional, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

_APPROVAL_SIGNALS = ("approve", "proceed", "yes", "go ahead", "confirmed", "ok, go", "ok go")

class BaseAgent:
    """
    Base class for all agents.
    """
    def __init__(self, model, checkpointer=None):
        self.model = model
        self.checkpointer = checkpointer or MemorySaver()
        self._compiled_graph = None

    def _make_compiled_graph(self):
        """
        To be implemented by subclasses to build the state graph.
        """
        raise NotImplementedError

    @property
    def graph(self):
        """Exposes the compiled graph."""
        if not self._compiled_graph:
            self._compiled_graph = self._make_compiled_graph()
        return self._compiled_graph

    def invoke(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invokes the agent's graph.
        """
        if not self._compiled_graph:
            self._compiled_graph = self._make_compiled_graph()
        
        return self._compiled_graph.invoke(inputs, config=config)

    @staticmethod
    def _get_intent(messages: Sequence[BaseMessage]) -> str:
        """Return the last non-approval HumanMessage content.

        When a HITL approval message ('Approve and proceed') is the last
        message, the coding agents must still act on the *original* user
        request, not on the approval string.
        """
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                low = msg.content.lower()
                if not any(sig in low for sig in _APPROVAL_SIGNALS):
                    return msg.content
        # Fallback: first human message or empty string
        for msg in messages:
            if isinstance(msg, HumanMessage):
                return msg.content
        return messages[-1].content if messages else ""

    def ainvoke(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Asynchronously invokes the agent's graph.
        """
        if not self._compiled_graph:
            self._compiled_graph = self._make_compiled_graph()
            
        return self._compiled_graph.ainvoke(inputs, config=config)
