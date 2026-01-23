"""
Sibyl Integration Package
=========================

Streamlit frontend components for visualizing MCP Crypto Order Flow Server.

This package contains:
- mcp_client.py: HTTP client to call MCP tools
- frontend/: Streamlit pages and components
"""

from .mcp_client import (
    MCPClient,
    MCPResponse,
    SyncMCPClient,
    get_mcp_client,
    get_sync_client,
)

__all__ = [
    "MCPClient",
    "MCPResponse",
    "SyncMCPClient",
    "get_mcp_client",
    "get_sync_client",
]
