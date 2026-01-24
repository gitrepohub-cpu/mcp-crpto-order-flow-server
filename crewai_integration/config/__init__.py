"""
Configuration Module for CrewAI Integration
===========================================

Provides YAML-based configuration management for:
- Agent configurations
- Task definitions
- System parameters
- Crew configurations
"""

from .loader import ConfigLoader
from .schemas import AgentConfig, TaskConfig, SystemConfig, CrewConfig

__all__ = [
    "ConfigLoader",
    "AgentConfig",
    "TaskConfig",
    "SystemConfig",
    "CrewConfig",
]
