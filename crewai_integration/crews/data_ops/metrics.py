"""
Phase 2: Data Operations Crew - Metrics Collection
===================================================

This module provides metrics collection and export for Phase 2 agents.
Metrics can be consumed by the Sibyl Dashboard and monitoring systems.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import threading
import json

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Metrics for a single agent."""
    agent_id: str
    actions_taken: int = 0
    decisions_made: int = 0
    errors: int = 0
    last_action_time: Optional[datetime] = None
    avg_response_time_ms: float = 0.0
    status: str = "idle"
    current_task: Optional[str] = None


@dataclass
class DataQualityMetrics:
    """Data quality metrics."""
    records_validated: int = 0
    validation_passed: int = 0
    validation_failed: int = 0
    anomalies_detected: int = 0
    quarantined_records: int = 0
    pass_rate: float = 100.0


@dataclass
class StreamingMetrics:
    """Streaming health metrics."""
    exchanges_connected: int = 0
    exchanges_total: int = 8
    streams_active: int = 0
    records_per_minute: float = 0.0
    data_gaps_detected: int = 0
    reconnections_triggered: int = 0
    backfills_requested: int = 0


@dataclass
class CleaningMetrics:
    """Data cleaning metrics."""
    gaps_filled: int = 0
    records_interpolated: int = 0
    records_aggregated: int = 0
    records_normalized: int = 0


@dataclass
class EscalationMetrics:
    """Escalation metrics."""
    total_escalations: int = 0
    escalations_by_level: Dict[str, int] = field(default_factory=lambda: {
        "low": 0, "medium": 0, "high": 0, "critical": 0
    })
    pending_escalations: int = 0
    resolved_escalations: int = 0


@dataclass
class CrewPerformanceMetrics:
    """Overall crew performance metrics."""
    uptime_seconds: float = 0.0
    start_time: Optional[datetime] = None
    total_tasks_completed: int = 0
    tasks_in_progress: int = 0
    avg_task_duration_seconds: float = 0.0
    health_score: float = 100.0  # 0-100


class DataOpsMetricsCollector:
    """
    Collects and aggregates metrics from the Data Operations Crew.
    
    This collector:
    - Tracks agent-level metrics
    - Aggregates data quality metrics
    - Monitors streaming health
    - Tracks escalations
    - Provides API for dashboard consumption
    """
    
    def __init__(self):
        """Initialize the metrics collector."""
        self._lock = threading.RLock()
        
        # Agent metrics
        self.agent_metrics: Dict[str, AgentMetrics] = {
            "data_collector": AgentMetrics(agent_id="data_collector"),
            "data_validator": AgentMetrics(agent_id="data_validator"),
            "data_cleaner": AgentMetrics(agent_id="data_cleaner"),
            "schema_manager": AgentMetrics(agent_id="schema_manager")
        }
        
        # Aggregated metrics
        self.data_quality = DataQualityMetrics()
        self.streaming = StreamingMetrics()
        self.cleaning = CleaningMetrics()
        self.escalations = EscalationMetrics()
        self.performance = CrewPerformanceMetrics()
        
        # Time series data (for charts)
        self._time_series_buffer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._max_time_series_points = 1000
        
        logger.info("DataOpsMetricsCollector initialized")
    
    # === Agent Metrics ===
    
    def record_agent_action(
        self,
        agent_id: str,
        action: str,
        success: bool = True,
        duration_ms: float = 0.0
    ):
        """Record an agent action."""
        with self._lock:
            if agent_id not in self.agent_metrics:
                self.agent_metrics[agent_id] = AgentMetrics(agent_id=agent_id)
            
            metrics = self.agent_metrics[agent_id]
            metrics.actions_taken += 1
            metrics.last_action_time = datetime.now(timezone.utc)
            
            if not success:
                metrics.errors += 1
            
            # Update rolling average response time
            if duration_ms > 0:
                if metrics.avg_response_time_ms == 0:
                    metrics.avg_response_time_ms = duration_ms
                else:
                    # Exponential moving average
                    metrics.avg_response_time_ms = (
                        0.9 * metrics.avg_response_time_ms + 0.1 * duration_ms
                    )
    
    def set_agent_status(self, agent_id: str, status: str, task: Optional[str] = None):
        """Update agent status."""
        with self._lock:
            if agent_id in self.agent_metrics:
                self.agent_metrics[agent_id].status = status
                self.agent_metrics[agent_id].current_task = task
    
    def record_decision(self, agent_id: str):
        """Record an agent decision."""
        with self._lock:
            if agent_id in self.agent_metrics:
                self.agent_metrics[agent_id].decisions_made += 1
    
    # === Data Quality Metrics ===
    
    def record_validation(self, passed: bool, is_anomaly: bool = False):
        """Record a validation result."""
        with self._lock:
            self.data_quality.records_validated += 1
            if passed:
                self.data_quality.validation_passed += 1
            else:
                self.data_quality.validation_failed += 1
            
            if is_anomaly:
                self.data_quality.anomalies_detected += 1
            
            # Update pass rate
            if self.data_quality.records_validated > 0:
                self.data_quality.pass_rate = (
                    self.data_quality.validation_passed / 
                    self.data_quality.records_validated * 100
                )
    
    def record_quarantine(self, count: int = 1):
        """Record quarantined records."""
        with self._lock:
            self.data_quality.quarantined_records += count
    
    # === Streaming Metrics ===
    
    def update_streaming_status(
        self,
        exchanges_connected: int,
        streams_active: int,
        records_per_minute: float
    ):
        """Update streaming metrics."""
        with self._lock:
            self.streaming.exchanges_connected = exchanges_connected
            self.streaming.streams_active = streams_active
            self.streaming.records_per_minute = records_per_minute
            
            # Add to time series
            self._add_time_series_point("streaming_rpm", {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "value": records_per_minute
            })
    
    def record_data_gap(self):
        """Record a detected data gap."""
        with self._lock:
            self.streaming.data_gaps_detected += 1
    
    def record_reconnection(self):
        """Record a reconnection attempt."""
        with self._lock:
            self.streaming.reconnections_triggered += 1
    
    def record_backfill(self):
        """Record a backfill request."""
        with self._lock:
            self.streaming.backfills_requested += 1
    
    # === Cleaning Metrics ===
    
    def record_interpolation(self, records_count: int):
        """Record interpolated records."""
        with self._lock:
            self.cleaning.records_interpolated += records_count
            self.cleaning.gaps_filled += 1
    
    def record_aggregation(self, records_count: int):
        """Record aggregated records."""
        with self._lock:
            self.cleaning.records_aggregated += records_count
    
    def record_normalization(self, records_count: int):
        """Record normalized records."""
        with self._lock:
            self.cleaning.records_normalized += records_count
    
    # === Escalation Metrics ===
    
    def record_escalation(self, level: str, resolved: bool = False):
        """Record an escalation."""
        with self._lock:
            self.escalations.total_escalations += 1
            
            level_lower = level.lower()
            if level_lower in self.escalations.escalations_by_level:
                self.escalations.escalations_by_level[level_lower] += 1
            
            if resolved:
                self.escalations.resolved_escalations += 1
            else:
                self.escalations.pending_escalations += 1
    
    def resolve_escalation(self):
        """Mark an escalation as resolved."""
        with self._lock:
            if self.escalations.pending_escalations > 0:
                self.escalations.pending_escalations -= 1
                self.escalations.resolved_escalations += 1
    
    # === Performance Metrics ===
    
    def start_crew(self):
        """Record crew start."""
        with self._lock:
            self.performance.start_time = datetime.now(timezone.utc)
    
    def update_performance(self):
        """Update performance metrics."""
        with self._lock:
            if self.performance.start_time:
                self.performance.uptime_seconds = (
                    datetime.now(timezone.utc) - self.performance.start_time
                ).total_seconds()
            
            # Calculate health score based on various factors
            self._calculate_health_score()
    
    def record_task_completion(self, duration_seconds: float):
        """Record a completed task."""
        with self._lock:
            self.performance.total_tasks_completed += 1
            
            # Update rolling average duration
            if self.performance.avg_task_duration_seconds == 0:
                self.performance.avg_task_duration_seconds = duration_seconds
            else:
                # Exponential moving average
                self.performance.avg_task_duration_seconds = (
                    0.9 * self.performance.avg_task_duration_seconds + 0.1 * duration_seconds
                )
    
    def _calculate_health_score(self):
        """Calculate overall health score (0-100)."""
        score = 100.0
        
        # Deduct for disconnected exchanges
        exchange_health = (
            self.streaming.exchanges_connected / max(self.streaming.exchanges_total, 1)
        ) * 100
        score -= max(0, 100 - exchange_health) * 0.3  # 30% weight
        
        # Deduct for low validation pass rate
        score -= max(0, 100 - self.data_quality.pass_rate) * 0.2  # 20% weight
        
        # Deduct for pending escalations
        if self.escalations.pending_escalations > 0:
            score -= min(20, self.escalations.pending_escalations * 2)  # Max 20 points
        
        # Deduct for critical escalations
        critical = self.escalations.escalations_by_level.get("critical", 0)
        if critical > 0:
            score -= min(15, critical * 5)  # Max 15 points
        
        # Deduct for agent errors
        total_errors = sum(a.errors for a in self.agent_metrics.values())
        total_actions = sum(a.actions_taken for a in self.agent_metrics.values())
        if total_actions > 0:
            error_rate = total_errors / total_actions * 100
            score -= min(15, error_rate)  # Max 15 points
        
        self.performance.health_score = max(0, min(100, score))
    
    # === Time Series ===
    
    def _add_time_series_point(self, series_name: str, point: Dict[str, Any]):
        """Add a point to a time series."""
        self._time_series_buffer[series_name].append(point)
        
        # Trim if too large
        if len(self._time_series_buffer[series_name]) > self._max_time_series_points:
            self._time_series_buffer[series_name] = \
                self._time_series_buffer[series_name][-self._max_time_series_points:]
    
    def get_time_series(
        self,
        series_name: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get time series data."""
        with self._lock:
            return self._time_series_buffer.get(series_name, [])[-limit:]
    
    # === Export Methods ===
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics formatted for dashboard consumption.
        
        Returns:
            Dict with all metrics in a dashboard-friendly format
        """
        with self._lock:
            self.update_performance()
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agents": {
                    agent_id: {
                        "agent_id": m.agent_id,
                        "actions_taken": m.actions_taken,
                        "decisions_made": m.decisions_made,
                        "errors": m.errors,
                        "last_action": m.last_action_time.isoformat() if m.last_action_time else None,
                        "avg_response_ms": round(m.avg_response_time_ms, 2),
                        "status": m.status,
                        "current_task": m.current_task
                    }
                    for agent_id, m in self.agent_metrics.items()
                },
                "data_quality": asdict(self.data_quality),
                "streaming": asdict(self.streaming),
                "cleaning": asdict(self.cleaning),
                "escalations": asdict(self.escalations),
                "performance": {
                    "uptime_seconds": round(self.performance.uptime_seconds, 0),
                    "start_time": self.performance.start_time.isoformat() if self.performance.start_time else None,
                    "total_tasks_completed": self.performance.total_tasks_completed,
                    "tasks_in_progress": self.performance.tasks_in_progress,
                    "avg_task_duration_seconds": round(self.performance.avg_task_duration_seconds, 2),
                    "health_score": round(self.performance.health_score, 1)
                }
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a brief summary of crew status."""
        with self._lock:
            self.update_performance()
            
            return {
                "health_score": round(self.performance.health_score, 1),
                "uptime_seconds": round(self.performance.uptime_seconds, 0),
                "exchanges_connected": self.streaming.exchanges_connected,
                "validation_pass_rate": round(self.data_quality.pass_rate, 1),
                "pending_escalations": self.escalations.pending_escalations,
                "total_actions": sum(a.actions_taken for a in self.agent_metrics.values()),
                "total_errors": sum(a.errors for a in self.agent_metrics.values())
            }
    
    def to_json(self) -> str:
        """Export metrics as JSON string."""
        return json.dumps(self.get_dashboard_metrics(), indent=2)


# Global singleton instance
_metrics_collector: Optional[DataOpsMetricsCollector] = None


def get_metrics_collector() -> DataOpsMetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = DataOpsMetricsCollector()
    
    return _metrics_collector
