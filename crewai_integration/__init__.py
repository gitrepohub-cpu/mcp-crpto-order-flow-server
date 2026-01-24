"""
CrewAI Integration for MCP Crypto Order Flow Server
====================================================

This module provides AI agent orchestration for the MCP crypto trading system.
It wraps existing MCP tools and exposes them to CrewAI agents for autonomous
operation while maintaining complete separation from the core MCP system.

Directory Structure:
-------------------
crewai_integration/
├── __init__.py              # This file - main entry point
├── crews/                   # Crew definitions (Data, Analytics, etc.)
├── agents/                  # Individual agent configurations
├── tasks/                   # Task definitions and workflows
├── tools/                   # MCP tool wrappers for CrewAI
├── flows/                   # Event-driven workflow definitions
├── config/                  # YAML configurations
├── state/                   # Persistent state management
├── events/                  # Event bus and communication
└── tests/                   # Testing framework

Integration Principles:
----------------------
1. NON-INVASIVE: No modifications to existing MCP tools or schemas
2. GRACEFUL DEGRADATION: MCP system operates normally if CrewAI is disabled
3. PERMISSION-BASED: Agents only access tools they're authorized to use
4. AUDITABLE: All agent actions are logged for review
5. RATE-LIMITED: Respects exchange rate limits

Version: 1.0.0 (Phase 1 Foundation)
"""

__version__ = "1.0.0"
__phase__ = "foundation"

from .core.controller import CrewAIController
from .core.registry import ToolRegistry, AgentRegistry
from .state.manager import StateManager
from .events.bus import EventBus

__all__ = [
    "CrewAIController",
    "ToolRegistry",
    "AgentRegistry",
    "StateManager",
    "EventBus",
]
