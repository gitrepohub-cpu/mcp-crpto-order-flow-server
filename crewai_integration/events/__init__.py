"""
Event Communication Module for CrewAI Integration
=================================================

Provides event-driven communication between:
- MCP system components
- CrewAI agents
- External triggers
"""

from .bus import EventBus, Event, EventHandler

__all__ = [
    "EventBus",
    "Event",
    "EventHandler",
]
