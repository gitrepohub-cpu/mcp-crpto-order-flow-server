"""
Tool Registry for CrewAI Integration
=====================================

Central registry that catalogs all 248+ MCP tools and provides:
- Tool categorization
- Metadata management
- Access level requirements
- Rate limit configurations
- Tool discovery for agents
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import functools

from .permissions import ToolCategory, AccessLevel

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration for a tool."""
    calls_per_second: float = 1.0
    calls_per_minute: int = 30
    burst_limit: int = 5
    cooldown_seconds: float = 0.5


@dataclass
class ToolMetadata:
    """Metadata for a registered MCP tool."""
    name: str
    category: ToolCategory
    description: str
    access_level: AccessLevel
    function: Optional[Callable] = None
    is_async: bool = True
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    parameters: Dict[str, Any] = field(default_factory=dict)
    returns: str = ""
    exchange: Optional[str] = None  # For exchange-specific tools
    examples: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    registered_at: datetime = field(default_factory=datetime.utcnow)
    invocation_count: int = 0
    last_invoked: Optional[datetime] = None
    avg_latency_ms: float = 0.0


class ToolRegistry:
    """
    Central registry for all MCP tools available to CrewAI agents.
    
    This registry wraps existing MCP tools and exposes them to CrewAI
    with proper metadata, rate limiting, and access control.
    """
    
    # Exchange-specific rate limits (seconds between calls)
    EXCHANGE_RATE_LIMITS = {
        "binance": RateLimitConfig(calls_per_second=0.5, calls_per_minute=30, burst_limit=3),
        "bybit": RateLimitConfig(calls_per_second=0.33, calls_per_minute=20, burst_limit=2),
        "okx": RateLimitConfig(calls_per_second=0.5, calls_per_minute=30, burst_limit=3),
        "hyperliquid": RateLimitConfig(calls_per_second=1.0, calls_per_minute=60, burst_limit=5),
        "gateio": RateLimitConfig(calls_per_second=0.5, calls_per_minute=30, burst_limit=3),
        "kraken": RateLimitConfig(calls_per_second=0.5, calls_per_minute=30, burst_limit=3),
        "deribit": RateLimitConfig(calls_per_second=0.5, calls_per_minute=30, burst_limit=3),
        "default": RateLimitConfig(calls_per_second=1.0, calls_per_minute=60, burst_limit=5),
    }
    
    def __init__(self):
        self._tools: Dict[str, ToolMetadata] = {}
        self._categories: Dict[ToolCategory, List[str]] = {cat: [] for cat in ToolCategory}
        self._exchanges: Dict[str, List[str]] = {}
        self._rate_limiters: Dict[str, asyncio.Semaphore] = {}
        self._last_call_times: Dict[str, datetime] = {}
        self._initialized = False
        
    def register(
        self,
        name: str,
        function: Callable,
        category: ToolCategory,
        description: str,
        access_level: AccessLevel = AccessLevel.READ_ONLY,
        exchange: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        returns: str = "",
        examples: Optional[List[str]] = None,
        tags: Optional[Set[str]] = None,
        rate_limit: Optional[RateLimitConfig] = None
    ) -> ToolMetadata:
        """
        Register an MCP tool for CrewAI access.
        
        Args:
            name: Unique tool name
            function: The actual MCP tool function
            category: Tool category for permissions
            description: Human-readable description
            access_level: Minimum access level required
            exchange: Exchange this tool belongs to (if applicable)
            parameters: Parameter schema
            returns: Return type description
            examples: Usage examples
            tags: Searchable tags
            rate_limit: Custom rate limit (uses exchange default if not provided)
            
        Returns:
            ToolMetadata for the registered tool
        """
        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered, updating...")
        
        # Determine rate limit
        if rate_limit is None:
            if exchange:
                rate_limit = self.EXCHANGE_RATE_LIMITS.get(
                    exchange.lower(),
                    self.EXCHANGE_RATE_LIMITS["default"]
                )
            else:
                rate_limit = self.EXCHANGE_RATE_LIMITS["default"]
        
        # Check if function is async
        is_async = asyncio.iscoroutinefunction(function)
        
        metadata = ToolMetadata(
            name=name,
            function=function,
            category=category,
            description=description,
            access_level=access_level,
            is_async=is_async,
            rate_limit=rate_limit,
            parameters=parameters or {},
            returns=returns,
            exchange=exchange,
            examples=examples or [],
            tags=tags or set()
        )
        
        self._tools[name] = metadata
        self._categories[category].append(name)
        
        if exchange:
            if exchange not in self._exchanges:
                self._exchanges[exchange] = []
            self._exchanges[exchange].append(name)
        
        logger.debug(f"Registered tool '{name}' in category '{category.value}'")
        return metadata
    
    def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[ToolMetadata]:
        """Get all tools in a category."""
        return [self._tools[name] for name in self._categories.get(category, [])]
    
    def get_tools_by_exchange(self, exchange: str) -> List[ToolMetadata]:
        """Get all tools for a specific exchange."""
        return [self._tools[name] for name in self._exchanges.get(exchange, [])]
    
    def search_tools(
        self,
        query: Optional[str] = None,
        category: Optional[ToolCategory] = None,
        exchange: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        max_access_level: Optional[AccessLevel] = None
    ) -> List[ToolMetadata]:
        """
        Search for tools matching criteria.
        
        Args:
            query: Text search in name/description
            category: Filter by category
            exchange: Filter by exchange
            tags: Filter by tags (any match)
            max_access_level: Filter by maximum access level
            
        Returns:
            List of matching tools
        """
        results = list(self._tools.values())
        
        if category:
            results = [t for t in results if t.category == category]
        
        if exchange:
            results = [t for t in results if t.exchange == exchange]
        
        if tags:
            results = [t for t in results if t.tags & tags]
        
        if max_access_level:
            results = [t for t in results if t.access_level <= max_access_level]
        
        if query:
            query_lower = query.lower()
            results = [
                t for t in results
                if query_lower in t.name.lower() or query_lower in t.description.lower()
            ]
        
        return results
    
    def list_all_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_category_map(self) -> Dict[str, ToolCategory]:
        """Get mapping of tool names to categories."""
        return {name: meta.category for name, meta in self._tools.items()}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_tools": len(self._tools),
            "by_category": {
                cat.value: len(tools) for cat, tools in self._categories.items()
            },
            "by_exchange": {
                ex: len(tools) for ex, tools in self._exchanges.items()
            },
            "by_access_level": {
                level.name: len([t for t in self._tools.values() if t.access_level == level])
                for level in AccessLevel
            }
        }
    
    async def invoke_tool(
        self,
        name: str,
        **kwargs
    ) -> Any:
        """
        Invoke a registered tool with rate limiting.
        
        This method should be called through the wrapper layer,
        not directly by agents.
        
        Args:
            name: Tool name
            **kwargs: Tool parameters
            
        Returns:
            Tool result
            
        Raises:
            KeyError: If tool not found
            Exception: From tool execution
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        
        tool = self._tools[name]
        
        # Apply rate limiting
        await self._apply_rate_limit(name, tool.rate_limit)
        
        # Track invocation
        start_time = datetime.utcnow()
        tool.invocation_count += 1
        tool.last_invoked = start_time
        
        try:
            # Invoke the tool
            if tool.is_async:
                result = await tool.function(**kwargs)
            else:
                result = tool.function(**kwargs)
            
            # Update latency tracking
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            tool.avg_latency_ms = (
                (tool.avg_latency_ms * (tool.invocation_count - 1) + latency_ms)
                / tool.invocation_count
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Tool '{name}' execution failed: {e}")
            raise
    
    async def _apply_rate_limit(self, tool_name: str, config: RateLimitConfig):
        """Apply rate limiting before tool invocation."""
        # Simple time-based rate limiting
        last_call = self._last_call_times.get(tool_name)
        
        if last_call:
            elapsed = (datetime.utcnow() - last_call).total_seconds()
            min_interval = 1.0 / config.calls_per_second
            
            if elapsed < min_interval:
                wait_time = min_interval - elapsed
                logger.debug(f"Rate limiting '{tool_name}': waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        self._last_call_times[tool_name] = datetime.utcnow()


class AgentRegistry:
    """
    Registry for CrewAI agents.
    
    Tracks all registered agents and their configurations.
    """
    
    def __init__(self):
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._crews: Dict[str, List[str]] = {}
        
    def register(
        self,
        agent_id: str,
        crew: str,
        role: str,
        goal: str,
        backstory: str,
        tools: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Register a CrewAI agent.
        
        Args:
            agent_id: Unique identifier
            crew: Crew membership
            role: Agent's role
            goal: Agent's goal
            backstory: Agent's backstory for context
            tools: List of tool names this agent can use
            config: Additional configuration
            
        Returns:
            Agent registration data
        """
        agent_data = {
            "id": agent_id,
            "crew": crew,
            "role": role,
            "goal": goal,
            "backstory": backstory,
            "tools": tools,
            "config": config or {},
            "registered_at": datetime.utcnow().isoformat(),
            "status": "registered"
        }
        
        self._agents[agent_id] = agent_data
        
        if crew not in self._crews:
            self._crews[crew] = []
        self._crews[crew].append(agent_id)
        
        logger.info(f"Registered agent '{agent_id}' in crew '{crew}'")
        return agent_data
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent by ID."""
        return self._agents.get(agent_id)
    
    def get_crew_agents(self, crew: str) -> List[Dict[str, Any]]:
        """Get all agents in a crew."""
        agent_ids = self._crews.get(crew, [])
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]
    
    def list_crews(self) -> List[str]:
        """List all registered crews."""
        return list(self._crews.keys())
    
    def list_agents(self) -> List[str]:
        """List all agent IDs."""
        return list(self._agents.keys())
    
    def update_status(self, agent_id: str, status: str) -> bool:
        """Update agent status."""
        if agent_id in self._agents:
            self._agents[agent_id]["status"] = status
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_agents": len(self._agents),
            "by_crew": {crew: len(agents) for crew, agents in self._crews.items()},
            "by_status": {
                status: len([a for a in self._agents.values() if a.get("status") == status])
                for status in ["registered", "active", "idle", "error"]
            }
        }
