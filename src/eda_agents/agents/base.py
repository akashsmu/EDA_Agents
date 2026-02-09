from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

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

    def invoke(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invokes the agent's graph.
        """
        if not self._compiled_graph:
            self._compiled_graph = self._make_compiled_graph()
        
        return self._compiled_graph.invoke(inputs, config=config)

    def ainvoke(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Asynchronously invokes the agent's graph.
        """
        if not self._compiled_graph:
            self._compiled_graph = self._make_compiled_graph()
            
        return self._compiled_graph.ainvoke(inputs, config=config)
