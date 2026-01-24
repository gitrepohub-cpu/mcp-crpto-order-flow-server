"""
Configuration Schemas for CrewAI Integration
============================================

Data classes for typed configuration objects.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class RateLimitConfig:
    """Rate limit configuration for an exchange."""
    calls_per_second: float = 1.0
    burst: int = 5
    cooldown_seconds: float = 0.5


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: str = "logs/crewai.log"
    max_size_mb: int = 100
    backup_count: int = 5


@dataclass
class AlertConfig:
    """Alert configuration."""
    enabled: bool = True
    error_threshold: int = 5
    latency_threshold_ms: int = 5000


@dataclass
class HumanApprovalConfig:
    """Human approval configuration."""
    required_for: List[str] = field(default_factory=lambda: [
        "streaming_control", "system_config", "database_write"
    ])
    timeout_seconds: int = 300


@dataclass
class ShadowModeConfig:
    """Shadow mode configuration."""
    enabled: bool = True
    log_recommendations: bool = True
    compare_with_actual: bool = True


@dataclass
class SystemConfig:
    """Complete system configuration."""
    version: str = "1.0.0"
    phase: str = "foundation"
    llm: LLMConfig = field(default_factory=LLMConfig)
    rate_limits: Dict[str, RateLimitConfig] = field(default_factory=dict)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    human_approval: HumanApprovalConfig = field(default_factory=HumanApprovalConfig)
    shadow_mode: ShadowModeConfig = field(default_factory=ShadowModeConfig)


@dataclass
class GuardrailConfig:
    """Agent guardrail configuration."""
    max_api_calls_per_minute: int = 60
    require_approval_for: List[str] = field(default_factory=list)
    forbidden_actions: List[str] = field(default_factory=list)
    alert_threshold: float = 0.7


@dataclass
class AgentConfig:
    """Agent configuration."""
    id: str
    crew: str
    role: str
    goal: str
    backstory: str
    tools: List[str] = field(default_factory=list)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    verbose: bool = True
    allow_delegation: bool = False
    memory: bool = True
    max_iterations: int = 10
    max_rpm: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CrewAI."""
        return {
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "verbose": self.verbose,
            "allow_delegation": self.allow_delegation,
            "memory": self.memory,
            "max_iterations": self.max_iterations,
            "max_rpm": self.max_rpm
        }


@dataclass
class TaskConfig:
    """Task configuration."""
    id: str
    name: str
    description: str
    agent: str
    expected_output: str
    async_execution: bool = True
    timeout_seconds: int = 300
    retry_count: int = 3
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CrewAI."""
        return {
            "description": self.description,
            "expected_output": self.expected_output,
            "async_execution": self.async_execution
        }


@dataclass
class CrewConfig:
    """Crew configuration."""
    id: str
    name: str
    description: str
    agents: List[str]
    tasks: List[str]
    process: str = "sequential"  # or "parallel", "hierarchical"
    verbose: bool = True
    memory: bool = True
    embedder: Optional[Dict[str, Any]] = None
    manager_llm: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CrewAI."""
        return {
            "verbose": self.verbose,
            "memory": self.memory,
            "process": self.process
        }
