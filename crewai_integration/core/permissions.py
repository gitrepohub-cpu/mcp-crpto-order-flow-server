"""
Permission Management System for CrewAI Agents
==============================================

Implements a role-based access control (RBAC) system that governs
which agents can access which MCP tools.

Access Levels:
-------------
- READ_ONLY: Can query data but not modify anything
- READ_WRITE: Can read data and write to agent-specific tables
- WRITE: Can write to operational tables (data collection)
- ADMIN: Full access (system configuration)

Categories:
-----------
Each tool belongs to a category that determines its default access level.
Agents are assigned permissions based on their crew membership.
"""

import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Access levels for tool permissions."""
    NONE = auto()       # No access
    READ_ONLY = auto()  # Can read/query data
    READ_WRITE = auto() # Can read and write to own tables
    WRITE = auto()      # Can write to operational tables
    ADMIN = auto()      # Full system access
    
    def __ge__(self, other):
        if not isinstance(other, AccessLevel):
            return NotImplemented
        return self.value >= other.value
    
    def __le__(self, other):
        if not isinstance(other, AccessLevel):
            return NotImplemented
        return self.value <= other.value
    
    def __gt__(self, other):
        if not isinstance(other, AccessLevel):
            return NotImplemented
        return self.value > other.value
    
    def __lt__(self, other):
        if not isinstance(other, AccessLevel):
            return NotImplemented
        return self.value < other.value


class ToolCategory(Enum):
    """Categories of MCP tools with associated access requirements."""
    # Exchange Data Fetchers (60+ tools) - Read-only, all agents
    EXCHANGE_DATA = "exchange_data"
    
    # Streaming Controllers (8 tools) - Write, Operations crew only
    STREAMING_CONTROL = "streaming_control"
    
    # Forecasting Tools (38+ tools) - Read-only, Analytics crew only
    FORECASTING = "forecasting"
    
    # Feature Calculators (35 tools) - Read-only, Intelligence crew only
    FEATURE_CALCULATOR = "feature_calculator"
    
    # Historical Queries (40+ tools) - Read-only, all agents
    HISTORICAL_QUERY = "historical_query"
    
    # Database Writers (15 tools) - Write, Data crew only
    DATABASE_WRITE = "database_write"
    
    # Visualization Tools (5 tools) - Read-only, Research crew only
    VISUALIZATION = "visualization"
    
    # System Configuration (10 tools) - Admin, Operations crew only
    SYSTEM_CONFIG = "system_config"
    
    # Analytics Tools (20+ tools) - Read-only, Analytics crew
    ANALYTICS = "analytics"
    
    # Composite Intelligence (10 tools) - Read-only, Intelligence crew
    COMPOSITE_INTELLIGENCE = "composite_intelligence"


@dataclass
class Permission:
    """Represents a permission grant."""
    tool_name: str
    access_level: AccessLevel
    granted_by: str = "system"
    granted_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if permission is still valid."""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True


@dataclass
class AgentPermissions:
    """Permissions assigned to a specific agent."""
    agent_id: str
    crew: str
    role: str
    permissions: Dict[str, Permission] = field(default_factory=dict)
    category_access: Dict[ToolCategory, AccessLevel] = field(default_factory=dict)
    
    def can_access(self, tool_name: str, required_level: AccessLevel) -> bool:
        """Check if agent can access a tool at the required level."""
        # Check specific permission first
        if tool_name in self.permissions:
            perm = self.permissions[tool_name]
            if perm.is_valid() and perm.access_level >= required_level:
                return True
        return False
    
    def can_access_category(self, category: ToolCategory, required_level: AccessLevel) -> bool:
        """Check if agent has category-level access."""
        if category in self.category_access:
            return self.category_access[category] >= required_level
        return False


class PermissionManager:
    """
    Manages permissions for all CrewAI agents.
    
    Implements RBAC with the following hierarchy:
    - System-level defaults
    - Crew-level permissions
    - Agent-specific overrides
    """
    
    # Default category access by crew
    CREW_DEFAULT_ACCESS: Dict[str, Dict[ToolCategory, AccessLevel]] = {
        "data_crew": {
            ToolCategory.EXCHANGE_DATA: AccessLevel.READ_ONLY,
            ToolCategory.DATABASE_WRITE: AccessLevel.WRITE,
            ToolCategory.HISTORICAL_QUERY: AccessLevel.READ_ONLY,
            ToolCategory.STREAMING_CONTROL: AccessLevel.NONE,
            ToolCategory.FORECASTING: AccessLevel.NONE,
            ToolCategory.FEATURE_CALCULATOR: AccessLevel.NONE,
            ToolCategory.VISUALIZATION: AccessLevel.NONE,
            ToolCategory.SYSTEM_CONFIG: AccessLevel.NONE,
            ToolCategory.ANALYTICS: AccessLevel.NONE,
            ToolCategory.COMPOSITE_INTELLIGENCE: AccessLevel.NONE,
        },
        "analytics_crew": {
            ToolCategory.EXCHANGE_DATA: AccessLevel.READ_ONLY,
            ToolCategory.FORECASTING: AccessLevel.READ_ONLY,
            ToolCategory.HISTORICAL_QUERY: AccessLevel.READ_ONLY,
            ToolCategory.ANALYTICS: AccessLevel.READ_ONLY,
            ToolCategory.DATABASE_WRITE: AccessLevel.NONE,
            ToolCategory.STREAMING_CONTROL: AccessLevel.NONE,
            ToolCategory.FEATURE_CALCULATOR: AccessLevel.NONE,
            ToolCategory.VISUALIZATION: AccessLevel.READ_ONLY,
            ToolCategory.SYSTEM_CONFIG: AccessLevel.NONE,
            ToolCategory.COMPOSITE_INTELLIGENCE: AccessLevel.NONE,
        },
        "intelligence_crew": {
            ToolCategory.EXCHANGE_DATA: AccessLevel.READ_ONLY,
            ToolCategory.FEATURE_CALCULATOR: AccessLevel.READ_ONLY,
            ToolCategory.COMPOSITE_INTELLIGENCE: AccessLevel.READ_ONLY,
            ToolCategory.HISTORICAL_QUERY: AccessLevel.READ_ONLY,
            ToolCategory.ANALYTICS: AccessLevel.READ_ONLY,
            ToolCategory.DATABASE_WRITE: AccessLevel.NONE,
            ToolCategory.STREAMING_CONTROL: AccessLevel.NONE,
            ToolCategory.FORECASTING: AccessLevel.READ_ONLY,
            ToolCategory.VISUALIZATION: AccessLevel.READ_ONLY,
            ToolCategory.SYSTEM_CONFIG: AccessLevel.NONE,
        },
        "operations_crew": {
            ToolCategory.STREAMING_CONTROL: AccessLevel.WRITE,
            ToolCategory.SYSTEM_CONFIG: AccessLevel.ADMIN,
            ToolCategory.EXCHANGE_DATA: AccessLevel.READ_ONLY,
            ToolCategory.HISTORICAL_QUERY: AccessLevel.READ_ONLY,
            ToolCategory.DATABASE_WRITE: AccessLevel.READ_ONLY,
            ToolCategory.FORECASTING: AccessLevel.READ_ONLY,
            ToolCategory.FEATURE_CALCULATOR: AccessLevel.READ_ONLY,
            ToolCategory.VISUALIZATION: AccessLevel.READ_ONLY,
            ToolCategory.ANALYTICS: AccessLevel.READ_ONLY,
            ToolCategory.COMPOSITE_INTELLIGENCE: AccessLevel.READ_ONLY,
        },
        "research_crew": {
            ToolCategory.EXCHANGE_DATA: AccessLevel.READ_ONLY,
            ToolCategory.VISUALIZATION: AccessLevel.READ_ONLY,
            ToolCategory.HISTORICAL_QUERY: AccessLevel.READ_ONLY,
            ToolCategory.ANALYTICS: AccessLevel.READ_ONLY,
            ToolCategory.FORECASTING: AccessLevel.READ_ONLY,
            ToolCategory.FEATURE_CALCULATOR: AccessLevel.READ_ONLY,
            ToolCategory.COMPOSITE_INTELLIGENCE: AccessLevel.READ_ONLY,
            ToolCategory.DATABASE_WRITE: AccessLevel.NONE,
            ToolCategory.STREAMING_CONTROL: AccessLevel.NONE,
            ToolCategory.SYSTEM_CONFIG: AccessLevel.NONE,
        },
    }
    
    def __init__(self):
        self._agent_permissions: Dict[str, AgentPermissions] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self._enabled = True
        
    def register_agent(
        self,
        agent_id: str,
        crew: str,
        role: str,
        custom_permissions: Optional[Dict[str, Permission]] = None
    ) -> AgentPermissions:
        """
        Register an agent with default crew-based permissions.
        
        Args:
            agent_id: Unique identifier for the agent
            crew: Crew the agent belongs to
            role: Role within the crew
            custom_permissions: Optional specific permission overrides
            
        Returns:
            AgentPermissions object for the registered agent
        """
        # Get default category access for this crew
        category_access = self.CREW_DEFAULT_ACCESS.get(crew, {}).copy()
        
        # Create agent permissions
        permissions = AgentPermissions(
            agent_id=agent_id,
            crew=crew,
            role=role,
            category_access=category_access,
            permissions=custom_permissions or {}
        )
        
        self._agent_permissions[agent_id] = permissions
        
        self._log_audit(
            action="register_agent",
            agent_id=agent_id,
            details={"crew": crew, "role": role}
        )
        
        logger.info(f"Registered agent '{agent_id}' in crew '{crew}' with role '{role}'")
        return permissions
    
    def check_permission(
        self,
        agent_id: str,
        tool_name: str,
        category: ToolCategory,
        required_level: AccessLevel = AccessLevel.READ_ONLY
    ) -> bool:
        """
        Check if an agent has permission to use a tool.
        
        Args:
            agent_id: The agent requesting access
            tool_name: The MCP tool being accessed
            category: Tool category for default permissions
            required_level: Minimum access level required
            
        Returns:
            True if access is granted, False otherwise
        """
        if not self._enabled:
            return True  # Bypass when disabled (testing)
        
        if agent_id not in self._agent_permissions:
            logger.warning(f"Unknown agent '{agent_id}' attempted to access '{tool_name}'")
            self._log_audit(
                action="access_denied",
                agent_id=agent_id,
                details={"tool": tool_name, "reason": "unknown_agent"}
            )
            return False
        
        perms = self._agent_permissions[agent_id]
        
        # Check specific tool permission first
        if perms.can_access(tool_name, required_level):
            self._log_audit(
                action="access_granted",
                agent_id=agent_id,
                details={"tool": tool_name, "via": "specific_permission"}
            )
            return True
        
        # Check category-level permission
        if perms.can_access_category(category, required_level):
            self._log_audit(
                action="access_granted",
                agent_id=agent_id,
                details={"tool": tool_name, "via": "category_permission", "category": category.value}
            )
            return True
        
        logger.warning(
            f"Agent '{agent_id}' denied access to '{tool_name}' "
            f"(required: {required_level.name}, category: {category.value})"
        )
        self._log_audit(
            action="access_denied",
            agent_id=agent_id,
            details={
                "tool": tool_name,
                "required_level": required_level.name,
                "category": category.value
            }
        )
        return False
    
    def grant_permission(
        self,
        agent_id: str,
        tool_name: str,
        access_level: AccessLevel,
        granted_by: str = "system",
        expires_at: Optional[datetime] = None
    ) -> bool:
        """
        Grant specific permission to an agent.
        
        Args:
            agent_id: The agent to grant permission to
            tool_name: The tool to grant access to
            access_level: Level of access to grant
            granted_by: Who/what is granting the permission
            expires_at: Optional expiration time
            
        Returns:
            True if permission was granted successfully
        """
        if agent_id not in self._agent_permissions:
            logger.error(f"Cannot grant permission to unknown agent '{agent_id}'")
            return False
        
        permission = Permission(
            tool_name=tool_name,
            access_level=access_level,
            granted_by=granted_by,
            expires_at=expires_at
        )
        
        self._agent_permissions[agent_id].permissions[tool_name] = permission
        
        self._log_audit(
            action="grant_permission",
            agent_id=agent_id,
            details={
                "tool": tool_name,
                "level": access_level.name,
                "granted_by": granted_by
            }
        )
        
        logger.info(f"Granted {access_level.name} access to '{tool_name}' for agent '{agent_id}'")
        return True
    
    def revoke_permission(self, agent_id: str, tool_name: str) -> bool:
        """Revoke a specific permission from an agent."""
        if agent_id not in self._agent_permissions:
            return False
        
        if tool_name in self._agent_permissions[agent_id].permissions:
            del self._agent_permissions[agent_id].permissions[tool_name]
            self._log_audit(
                action="revoke_permission",
                agent_id=agent_id,
                details={"tool": tool_name}
            )
            logger.info(f"Revoked permission for '{tool_name}' from agent '{agent_id}'")
            return True
        return False
    
    def get_agent_permissions(self, agent_id: str) -> Optional[AgentPermissions]:
        """Get all permissions for an agent."""
        return self._agent_permissions.get(agent_id)
    
    def list_accessible_tools(
        self,
        agent_id: str,
        tool_categories: Dict[str, ToolCategory]
    ) -> List[str]:
        """
        List all tools accessible by an agent.
        
        Args:
            agent_id: The agent to check
            tool_categories: Dict mapping tool names to categories
            
        Returns:
            List of accessible tool names
        """
        if agent_id not in self._agent_permissions:
            return []
        
        perms = self._agent_permissions[agent_id]
        accessible = []
        
        for tool_name, category in tool_categories.items():
            if perms.can_access(tool_name, AccessLevel.READ_ONLY):
                accessible.append(tool_name)
            elif perms.can_access_category(category, AccessLevel.READ_ONLY):
                accessible.append(tool_name)
        
        return accessible
    
    def _log_audit(self, action: str, agent_id: str, details: Dict[str, Any]):
        """Log an audit entry."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "agent_id": agent_id,
            "details": details
        }
        self._audit_log.append(entry)
        
        # Keep only last 10000 entries in memory
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-10000:]
    
    def get_audit_log(
        self,
        agent_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get audit log entries.
        
        Args:
            agent_id: Filter by agent
            action: Filter by action type
            limit: Maximum entries to return
            
        Returns:
            List of audit log entries
        """
        entries = self._audit_log
        
        if agent_id:
            entries = [e for e in entries if e["agent_id"] == agent_id]
        if action:
            entries = [e for e in entries if e["action"] == action]
        
        return entries[-limit:]
    
    def disable(self):
        """Disable permission checking (for testing)."""
        self._enabled = False
        logger.warning("Permission checking DISABLED")
    
    def enable(self):
        """Enable permission checking."""
        self._enabled = True
        logger.info("Permission checking enabled")
