"""
CrewAI Tools Module
===================

This module provides tool wrappers that expose MCP tools to CrewAI agents.
Each wrapper includes:
- Input validation
- Output parsing
- Error handling
- Rate limiting
- Permission checking
- Audit logging
"""

from .base import ToolWrapper, tool_wrapper
from .wrappers import (
    ExchangeDataTools,
    ForecastingTools,
    AnalyticsTools,
    StreamingTools,
    FeatureTools,
    VisualizationTools,
)

__all__ = [
    "ToolWrapper",
    "tool_wrapper",
    "ExchangeDataTools",
    "ForecastingTools",
    "AnalyticsTools",
    "StreamingTools",
    "FeatureTools",
    "VisualizationTools",
]
