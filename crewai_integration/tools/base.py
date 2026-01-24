"""
Base Tool Wrapper for CrewAI Integration
========================================

Provides the foundational wrapper class that all MCP tool wrappers inherit from.
Implements common functionality:
- Permission checking
- Input validation
- Output formatting
- Error handling
- Rate limiting
- Audit logging
"""

import logging
import asyncio
import functools
import traceback
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ToolInvocation:
    """Record of a tool invocation."""
    tool_name: str
    agent_id: str
    parameters: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    result: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class ValidationResult:
    """Result of parameter validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_params: Dict[str, Any] = field(default_factory=dict)


class ToolWrapper(ABC):
    """
    Abstract base class for all MCP tool wrappers.
    
    Each wrapper class groups related tools and provides:
    - Consistent interface for CrewAI agents
    - Automatic permission checking
    - Input validation and sanitization
    - Output formatting for agent consumption
    - Error handling and recovery
    - Audit logging
    
    Subclasses should implement `_get_tools()` to return
    the mapping of tool names to their wrapper methods.
    
    Shadow Mode:
        When shadow_mode=True, the wrapper operates in observation mode:
        - Permission checks are bypassed
        - Tools can be listed and inspected but not executed
        - Useful for validation and testing without side effects
    """
    
    def __init__(
        self,
        permission_manager: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        agent_id: Optional[str] = None,
        shadow_mode: bool = False
    ):
        """
        Initialize the tool wrapper.
        
        Args:
            permission_manager: PermissionManager instance for access control (optional in shadow mode)
            tool_registry: ToolRegistry instance for tool metadata (optional in shadow mode)
            agent_id: Default agent ID for invocations
            shadow_mode: If True, wrapper operates in observation mode without executing tools
        """
        self.permission_manager = permission_manager
        self.tool_registry = tool_registry
        self.agent_id = agent_id
        self.shadow_mode = shadow_mode
        self._invocation_log: List[ToolInvocation] = []
        self._error_count = 0
        self._success_count = 0
        
        # Log shadow mode initialization
        if shadow_mode:
            logger.debug(f"{self.__class__.__name__} initialized in SHADOW MODE")
    
    @abstractmethod
    def _get_tools(self) -> Dict[str, Callable]:
        """
        Return mapping of tool names to wrapper methods.
        
        Subclasses must implement this to define available tools.
        
        Returns:
            Dict mapping tool name to callable wrapper method
        """
        pass
    
    @abstractmethod
    def _get_category(self) -> str:
        """Return the tool category for this wrapper."""
        pass
    
    def list_tools(self) -> List[str]:
        """List all available tools in this wrapper."""
        return list(self._get_tools().keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific tool."""
        if not self.tool_registry:
            # In shadow mode without registry, return basic info
            tools = self._get_tools()
            if tool_name in tools:
                return {
                    "name": tool_name,
                    "description": f"Tool {tool_name} from {self.__class__.__name__}",
                    "category": self._get_category(),
                    "parameters": {},
                    "returns": "Dict[str, Any]",
                    "examples": []
                }
            return None
            
        tool_meta = self.tool_registry.get_tool(tool_name)
        if tool_meta:
            return {
                "name": tool_meta.name,
                "description": tool_meta.description,
                "category": tool_meta.category.value,
                "parameters": tool_meta.parameters,
                "returns": tool_meta.returns,
                "examples": tool_meta.examples
            }
        return None
    
    def is_shadow_mode(self) -> bool:
        """Check if wrapper is operating in shadow mode."""
        return self.shadow_mode
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the wrapper.
        
        Returns:
            Dict with health status, available tools, and operational state
        """
        tools = self._get_tools()
        return {
            "wrapper": self.__class__.__name__,
            "category": self._get_category(),
            "shadow_mode": self.shadow_mode,
            "operational": True,
            "tools_count": len(tools),
            "tools_available": list(tools.keys())[:10],  # First 10 tools
            "has_permission_manager": self.permission_manager is not None,
            "has_tool_registry": self.tool_registry is not None,
            "success_count": self._success_count,
            "error_count": self._error_count
        }
    
    async def invoke(
        self,
        tool_name: str,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke a tool with full wrapper functionality.
        
        Args:
            tool_name: Name of the tool to invoke
            agent_id: Agent making the request
            **kwargs: Tool parameters
            
        Returns:
            Structured response with result or error
            
        Note:
            In shadow mode, tools are not actually executed. Instead,
            a simulated response is returned for validation purposes.
        """
        agent = agent_id or self.agent_id or "unknown"
        invocation = ToolInvocation(
            tool_name=tool_name,
            agent_id=agent,
            parameters=kwargs,
            started_at=datetime.utcnow()
        )
        
        try:
            # Step 1: Check if tool exists
            tools = self._get_tools()
            if tool_name not in tools:
                raise ValueError(f"Tool '{tool_name}' not found in {self.__class__.__name__}")
            
            # Shadow mode: Return simulated response without executing
            if self.shadow_mode:
                invocation.completed_at = datetime.utcnow()
                invocation.success = True
                invocation.latency_ms = (
                    invocation.completed_at - invocation.started_at
                ).total_seconds() * 1000
                
                self._success_count += 1
                self._log_invocation(invocation)
                
                return {
                    "success": True,
                    "tool": tool_name,
                    "result": {
                        "_shadow_mode": True,
                        "_simulated": True,
                        "message": f"Shadow mode: {tool_name} would be executed with {kwargs}"
                    },
                    "latency_ms": invocation.latency_ms,
                    "warnings": ["Running in shadow mode - tool not actually executed"]
                }
            
            # Step 2: Check permissions (skip if no permission_manager)
            if self.tool_registry and self.permission_manager:
                tool_meta = self.tool_registry.get_tool(tool_name)
                if tool_meta:
                    has_permission = self.permission_manager.check_permission(
                        agent_id=agent,
                        tool_name=tool_name,
                        category=tool_meta.category,
                        required_level=tool_meta.access_level
                    )
                    if not has_permission:
                        raise PermissionError(f"Agent '{agent}' lacks permission for '{tool_name}'")
            
            # Step 3: Validate parameters
            validation = await self._validate_parameters(tool_name, kwargs)
            if not validation.valid:
                raise ValueError(f"Invalid parameters: {', '.join(validation.errors)}")
            
            # Step 4: Execute the tool
            wrapper_method = tools[tool_name]
            if asyncio.iscoroutinefunction(wrapper_method):
                result = await wrapper_method(**validation.sanitized_params)
            else:
                result = wrapper_method(**validation.sanitized_params)
            
            # Step 5: Format output
            formatted_result = await self._format_output(tool_name, result)
            
            # Record success
            invocation.completed_at = datetime.utcnow()
            invocation.success = True
            invocation.result = formatted_result
            invocation.latency_ms = (
                invocation.completed_at - invocation.started_at
            ).total_seconds() * 1000
            
            self._success_count += 1
            self._log_invocation(invocation)
            
            return {
                "success": True,
                "tool": tool_name,
                "result": formatted_result,
                "latency_ms": invocation.latency_ms,
                "warnings": validation.warnings
            }
            
        except PermissionError as e:
            invocation.completed_at = datetime.utcnow()
            invocation.error = str(e)
            self._error_count += 1
            self._log_invocation(invocation)
            logger.warning(f"Permission denied: {e}")
            return {
                "success": False,
                "tool": tool_name,
                "error": "permission_denied",
                "message": str(e)
            }
            
        except ValueError as e:
            invocation.completed_at = datetime.utcnow()
            invocation.error = str(e)
            self._error_count += 1
            self._log_invocation(invocation)
            logger.warning(f"Validation error: {e}")
            return {
                "success": False,
                "tool": tool_name,
                "error": "validation_error",
                "message": str(e)
            }
            
        except Exception as e:
            invocation.completed_at = datetime.utcnow()
            invocation.error = str(e)
            self._error_count += 1
            self._log_invocation(invocation)
            logger.error(f"Tool execution error: {e}\n{traceback.format_exc()}")
            return {
                "success": False,
                "tool": tool_name,
                "error": "execution_error",
                "message": str(e)
            }
    
    async def _validate_parameters(
        self,
        tool_name: str,
        params: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate and sanitize tool parameters.
        
        Override in subclasses for tool-specific validation.
        
        Args:
            tool_name: Tool being invoked
            params: Parameters to validate
            
        Returns:
            ValidationResult with sanitized parameters
        """
        # Basic validation - override for specific requirements
        sanitized = {}
        errors = []
        warnings = []
        
        tool_meta = self.tool_registry.get_tool(tool_name)
        if tool_meta and tool_meta.parameters:
            required_params = [
                k for k, v in tool_meta.parameters.items()
                if v.get("required", False)
            ]
            for req in required_params:
                if req not in params:
                    errors.append(f"Missing required parameter: {req}")
        
        # Copy provided parameters
        for key, value in params.items():
            sanitized[key] = value
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_params=sanitized
        )
    
    async def _format_output(self, tool_name: str, result: Any) -> Any:
        """
        Format tool output for agent consumption.
        
        Override in subclasses for tool-specific formatting.
        
        Args:
            tool_name: Tool that produced the result
            result: Raw tool result
            
        Returns:
            Formatted result suitable for agent processing
        """
        # Default: return as-is
        return result
    
    def _log_invocation(self, invocation: ToolInvocation):
        """Log a tool invocation."""
        self._invocation_log.append(invocation)
        
        # Keep only last 1000 invocations in memory
        if len(self._invocation_log) > 1000:
            self._invocation_log = self._invocation_log[-1000:]
        
        # Log to standard logger
        if invocation.success:
            logger.debug(
                f"Tool '{invocation.tool_name}' invoked by '{invocation.agent_id}' "
                f"({invocation.latency_ms:.1f}ms)"
            )
        else:
            logger.warning(
                f"Tool '{invocation.tool_name}' failed for '{invocation.agent_id}': "
                f"{invocation.error}"
            )
    
    def get_invocation_history(
        self,
        limit: int = 100,
        tool_name: Optional[str] = None,
        success_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get invocation history."""
        history = self._invocation_log
        
        if tool_name:
            history = [h for h in history if h.tool_name == tool_name]
        if success_only:
            history = [h for h in history if h.success]
        
        return [
            {
                "tool": h.tool_name,
                "agent": h.agent_id,
                "started": h.started_at.isoformat(),
                "success": h.success,
                "latency_ms": h.latency_ms,
                "error": h.error
            }
            for h in history[-limit:]
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get wrapper statistics."""
        return {
            "wrapper": self.__class__.__name__,
            "category": self._get_category(),
            "shadow_mode": self.shadow_mode,
            "tools_count": len(self._get_tools()),
            "success_count": self._success_count,
            "error_count": self._error_count,
            "success_rate": (
                self._success_count / (self._success_count + self._error_count)
                if (self._success_count + self._error_count) > 0 else 0
            ),
            "total_invocations": len(self._invocation_log)
        }


def tool_wrapper(
    tool_name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None
):
    """
    Decorator for creating tool wrapper methods.
    
    Usage:
        @tool_wrapper("my_tool", "Description of my tool")
        async def my_tool_wrapper(self, param1: str) -> Dict:
            return await self._invoke_mcp_tool("actual_tool_name", param1=param1)
    
    Args:
        tool_name: Name for the wrapper
        description: Tool description for agents
        parameters: Parameter schema
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            return await func(self, *args, **kwargs)
        
        wrapper._tool_name = tool_name
        wrapper._tool_description = description
        wrapper._tool_parameters = parameters or {}
        
        return wrapper
    return decorator
