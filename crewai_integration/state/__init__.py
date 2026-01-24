"""
State Management Module for CrewAI
==================================

Provides persistent state management for:
- Agent decision history
- Conversation context
- Learning outcomes
- Inter-agent communication
- Flow checkpoints
- Shared knowledge base
"""

from .manager import StateManager
from .schemas import AgentState, FlowState, KnowledgeEntry

__all__ = [
    "StateManager",
    "AgentState",
    "FlowState",
    "KnowledgeEntry",
]
