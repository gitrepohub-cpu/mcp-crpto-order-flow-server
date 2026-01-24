"""
State Schemas for CrewAI Integration
====================================

Data classes representing state objects stored in the state manager.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class AgentState:
    """State for an individual agent."""
    agent_id: str
    crew: str
    role: str
    status: str = "idle"
    current_task: Optional[str] = None
    memory: Dict[str, Any] = field(default_factory=dict)
    decision_count: int = 0
    success_count: int = 0
    error_count: int = 0
    last_active: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "crew": self.crew,
            "role": self.role,
            "status": self.status,
            "current_task": self.current_task,
            "decision_count": self.decision_count,
            "success_rate": self.success_rate,
            "last_active": self.last_active.isoformat() if self.last_active else None
        }


@dataclass
class FlowState:
    """State for a multi-step flow."""
    flow_id: str
    flow_name: str
    current_step: str
    total_steps: int
    status: str = "running"  # running, paused, completed, failed
    started_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    state_data: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    @property
    def progress(self) -> float:
        """Calculate flow progress (0-1)."""
        if self.status == "completed":
            return 1.0
        # This would need step tracking to be accurate
        return 0.5 if self.current_step else 0.0
    
    def add_checkpoint(self, step: str, data: Dict[str, Any]):
        """Add a checkpoint."""
        self.checkpoints.append({
            "step": step,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        })
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "flow_id": self.flow_id,
            "flow_name": self.flow_name,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "status": self.status,
            "progress": self.progress,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "checkpoint_count": len(self.checkpoints),
            "error": self.error
        }


@dataclass
class KnowledgeEntry:
    """An entry in the shared knowledge base."""
    id: int
    discovered_by: str
    category: str
    title: str
    content: str
    confidence: float = 0.5
    verified: bool = False
    verified_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    usefulness_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "discovered_by": self.discovered_by,
            "category": self.category,
            "title": self.title,
            "content": self.content,
            "confidence": self.confidence,
            "verified": self.verified,
            "created_at": self.created_at.isoformat(),
            "usefulness_score": self.usefulness_score,
            "tags": self.tags
        }


@dataclass
class AgentMessage:
    """Inter-agent communication message."""
    id: int
    from_agent: str
    to_agent: Optional[str]  # None for broadcasts
    message_type: str  # 'request', 'response', 'alert', 'info'
    content: Dict[str, Any]
    priority: int = 5  # 1-10
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "from": self.from_agent,
            "to": self.to_agent,
            "type": self.message_type,
            "content": self.content,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged
        }


@dataclass 
class DecisionRecord:
    """Record of an agent decision."""
    id: int
    agent_id: str
    crew: str
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    success: bool
    latency_ms: float
    reasoning: str
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "crew": self.crew,
            "tool": self.tool_name,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat()
        }
