"""
Phase 2: Data Operations Crew - Autonomous Behaviors
=====================================================

This module defines autonomous behaviors and triggers for the Data Operations Crew.
These behaviors enable agents to respond to events without explicit human requests.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of triggers for autonomous behaviors."""
    THRESHOLD = "threshold"          # Value exceeds threshold
    PATTERN = "pattern"              # Pattern detected
    SCHEDULE = "schedule"            # Time-based trigger
    EVENT = "event"                  # Event-based trigger
    ABSENCE = "absence"              # Absence of expected data


@dataclass
class TriggerCondition:
    """Defines a condition that triggers autonomous behavior."""
    trigger_type: TriggerType
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    cooldown_seconds: int = 60
    last_triggered: Optional[datetime] = None
    
    def can_trigger(self) -> bool:
        """Check if trigger is allowed (cooldown passed)."""
        if self.last_triggered is None:
            return True
        elapsed = (datetime.utcnow() - self.last_triggered).total_seconds()
        return elapsed >= self.cooldown_seconds
    
    def mark_triggered(self):
        """Mark trigger as fired."""
        self.last_triggered = datetime.utcnow()


@dataclass
class AutonomousAction:
    """Defines an action to take when a trigger fires."""
    name: str
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False
    escalation_on_failure: bool = True


@dataclass
class Behavior:
    """Combines a trigger with an action."""
    name: str
    description: str
    trigger: TriggerCondition
    action: AutonomousAction
    enabled: bool = True
    priority: int = 5  # 1 = highest, 10 = lowest


class AutonomousBehaviorEngine:
    """
    Engine for managing and executing autonomous behaviors.
    
    This engine monitors conditions and executes actions when triggers fire.
    """
    
    def __init__(self, crew):
        """
        Initialize the behavior engine.
        
        Args:
            crew: The DataOperationsCrew instance
        """
        self.crew = crew
        self.behaviors: Dict[str, Behavior] = {}
        self._running = False
        self._evaluators: Dict[TriggerType, Callable] = {
            TriggerType.THRESHOLD: self._evaluate_threshold,
            TriggerType.PATTERN: self._evaluate_pattern,
            TriggerType.SCHEDULE: self._evaluate_schedule,
            TriggerType.EVENT: self._evaluate_event,
            TriggerType.ABSENCE: self._evaluate_absence
        }
        
        # Register default behaviors
        self._register_default_behaviors()
    
    def _register_default_behaviors(self):
        """Register the default autonomous behaviors from configuration."""
        
        # ===== Data Collector Behaviors =====
        
        # Reconnection on disconnect
        self.register_behavior(Behavior(
            name="auto_reconnect_on_disconnect",
            description="Automatically attempt reconnection when exchange disconnects",
            trigger=TriggerCondition(
                trigger_type=TriggerType.EVENT,
                name="exchange_disconnection",
                parameters={"event_type": "exchange_disconnection"},
                cooldown_seconds=30
            ),
            action=AutonomousAction(
                name="reconnect_exchange",
                action_type="tool_call",
                parameters={"tool": "reconnect_exchange", "max_retries": 3},
                requires_approval=False
            ),
            priority=1
        ))
        
        # Backfill on gap detection
        self.register_behavior(Behavior(
            name="auto_backfill_on_gap",
            description="Request backfill when data gap is detected",
            trigger=TriggerCondition(
                trigger_type=TriggerType.THRESHOLD,
                name="data_gap_duration",
                parameters={
                    "metric": "gap_duration_seconds",
                    "threshold": 300,  # 5 minutes
                    "operator": ">"
                },
                cooldown_seconds=600
            ),
            action=AutonomousAction(
                name="request_backfill",
                action_type="tool_call",
                parameters={"tool": "request_backfill"},
                requires_approval=False
            ),
            priority=2
        ))
        
        # ===== Data Validator Behaviors =====
        
        # Quarantine on severe anomaly
        self.register_behavior(Behavior(
            name="auto_quarantine_severe_anomaly",
            description="Quarantine data when severe anomaly detected",
            trigger=TriggerCondition(
                trigger_type=TriggerType.THRESHOLD,
                name="anomaly_severity",
                parameters={
                    "metric": "z_score",
                    "threshold": 4.0,
                    "operator": ">"
                },
                cooldown_seconds=60
            ),
            action=AutonomousAction(
                name="quarantine_data",
                action_type="tool_call",
                parameters={"tool": "quarantine_data"},
                requires_approval=False
            ),
            priority=1
        ))
        
        # Cross-exchange validation alert
        self.register_behavior(Behavior(
            name="alert_cross_exchange_inconsistency",
            description="Alert when price differs significantly across exchanges",
            trigger=TriggerCondition(
                trigger_type=TriggerType.THRESHOLD,
                name="cross_exchange_deviation",
                parameters={
                    "metric": "deviation_percent",
                    "threshold": 5.0,
                    "operator": ">"
                },
                cooldown_seconds=300
            ),
            action=AutonomousAction(
                name="log_quality_issue",
                action_type="log",
                parameters={"severity": "medium"},
                requires_approval=False,
                escalation_on_failure=True
            ),
            priority=2
        ))
        
        # ===== Data Cleaner Behaviors =====
        
        # Auto-interpolate small gaps
        self.register_behavior(Behavior(
            name="auto_interpolate_small_gaps",
            description="Automatically interpolate gaps under 5 minutes",
            trigger=TriggerCondition(
                trigger_type=TriggerType.THRESHOLD,
                name="gap_for_interpolation",
                parameters={
                    "metric": "gap_duration_seconds",
                    "min_threshold": 60,
                    "max_threshold": 300,
                    "data_types": ["price", "funding", "open_interest"]
                },
                cooldown_seconds=60
            ),
            action=AutonomousAction(
                name="interpolate_missing_data",
                action_type="tool_call",
                parameters={"tool": "interpolate_missing_data", "strategy": "linear"},
                requires_approval=False
            ),
            priority=3
        ))
        
        # Scheduled aggregation
        self.register_behavior(Behavior(
            name="scheduled_timeframe_aggregation",
            description="Aggregate data to higher timeframes on schedule",
            trigger=TriggerCondition(
                trigger_type=TriggerType.SCHEDULE,
                name="aggregation_schedule",
                parameters={
                    "interval_minutes": 5,
                    "at_minute": 0  # At minute 0, 5, 10, etc.
                },
                cooldown_seconds=300
            ),
            action=AutonomousAction(
                name="aggregate_to_timeframe",
                action_type="tool_call",
                parameters={
                    "tool": "aggregate_to_timeframe",
                    "source_timeframe": "1m",
                    "target_timeframe": "5m"
                },
                requires_approval=False
            ),
            priority=5
        ))
        
        # ===== Schema Manager Behaviors =====
        
        # Table health check
        self.register_behavior(Behavior(
            name="scheduled_health_check",
            description="Periodic health check of all tables",
            trigger=TriggerCondition(
                trigger_type=TriggerType.SCHEDULE,
                name="health_check_schedule",
                parameters={"interval_minutes": 5},
                cooldown_seconds=300
            ),
            action=AutonomousAction(
                name="get_all_tables_health",
                action_type="tool_call",
                parameters={"tool": "get_all_tables_health"},
                requires_approval=False
            ),
            priority=4
        ))
        
        # Auto-vacuum on high fragmentation
        self.register_behavior(Behavior(
            name="auto_vacuum_fragmented",
            description="Vacuum tables with high fragmentation",
            trigger=TriggerCondition(
                trigger_type=TriggerType.THRESHOLD,
                name="table_fragmentation",
                parameters={
                    "metric": "fragmentation_percent",
                    "threshold": 30,
                    "operator": ">"
                },
                cooldown_seconds=3600
            ),
            action=AutonomousAction(
                name="vacuum_table",
                action_type="tool_call",
                parameters={"tool": "vacuum_table"},
                requires_approval=False
            ),
            priority=6
        ))
        
        # Capacity alert
        self.register_behavior(Behavior(
            name="capacity_threshold_alert",
            description="Alert when database approaches capacity",
            trigger=TriggerCondition(
                trigger_type=TriggerType.THRESHOLD,
                name="capacity_usage",
                parameters={
                    "metric": "capacity_percent",
                    "threshold": 80,
                    "operator": ">"
                },
                cooldown_seconds=3600
            ),
            action=AutonomousAction(
                name="escalate_capacity",
                action_type="escalate",
                parameters={"level": "high"},
                requires_approval=False,
                escalation_on_failure=True
            ),
            priority=1
        ))
    
    def register_behavior(self, behavior: Behavior):
        """Register a behavior with the engine."""
        self.behaviors[behavior.name] = behavior
        logger.info(f"Registered behavior: {behavior.name}")
    
    def unregister_behavior(self, name: str):
        """Unregister a behavior."""
        if name in self.behaviors:
            del self.behaviors[name]
            logger.info(f"Unregistered behavior: {name}")
    
    def enable_behavior(self, name: str):
        """Enable a behavior."""
        if name in self.behaviors:
            self.behaviors[name].enabled = True
    
    def disable_behavior(self, name: str):
        """Disable a behavior."""
        if name in self.behaviors:
            self.behaviors[name].enabled = False
    
    async def start(self):
        """Start the behavior engine."""
        self._running = True
        logger.info("Autonomous behavior engine started")
        
        await self._run_loop()
    
    async def stop(self):
        """Stop the behavior engine."""
        self._running = False
        logger.info("Autonomous behavior engine stopped")
    
    async def _run_loop(self):
        """Main loop for evaluating behaviors."""
        while self._running:
            try:
                # Get all enabled behaviors sorted by priority
                active_behaviors = sorted(
                    [b for b in self.behaviors.values() if b.enabled],
                    key=lambda x: x.priority
                )
                
                for behavior in active_behaviors:
                    if behavior.trigger.can_trigger():
                        should_fire = await self._evaluate_trigger(behavior.trigger)
                        if should_fire:
                            await self._execute_action(behavior)
                            behavior.trigger.mark_triggered()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Behavior engine error: {e}")
                await asyncio.sleep(30)
    
    async def _evaluate_trigger(self, trigger: TriggerCondition) -> bool:
        """Evaluate if a trigger should fire."""
        evaluator = self._evaluators.get(trigger.trigger_type)
        if evaluator:
            return await evaluator(trigger)
        return False
    
    async def _evaluate_threshold(self, trigger: TriggerCondition) -> bool:
        """Evaluate threshold-based trigger."""
        params = trigger.parameters
        metric = params.get("metric")
        threshold = params.get("threshold")
        operator = params.get("operator", ">")
        
        # Get current metric value
        current_value = await self._get_metric_value(metric)
        if current_value is None:
            return False
        
        if operator == ">":
            return current_value > threshold
        elif operator == "<":
            return current_value < threshold
        elif operator == ">=":
            return current_value >= threshold
        elif operator == "<=":
            return current_value <= threshold
        elif operator == "==":
            return current_value == threshold
        
        return False
    
    async def _evaluate_pattern(self, trigger: TriggerCondition) -> bool:
        """Evaluate pattern-based trigger."""
        # Pattern detection would go here
        return False
    
    async def _evaluate_schedule(self, trigger: TriggerCondition) -> bool:
        """Evaluate schedule-based trigger."""
        params = trigger.parameters
        interval_minutes = params.get("interval_minutes", 5)
        at_minute = params.get("at_minute")
        
        now = datetime.utcnow()
        
        if at_minute is not None:
            # Check if current minute matches
            return now.minute % interval_minutes == at_minute
        else:
            # Just check interval
            return now.minute % interval_minutes == 0
    
    async def _evaluate_event(self, trigger: TriggerCondition) -> bool:
        """Evaluate event-based trigger."""
        # Event triggers are handled via the event queue, not polling
        return False
    
    async def _evaluate_absence(self, trigger: TriggerCondition) -> bool:
        """Evaluate absence-based trigger (missing data)."""
        params = trigger.parameters
        max_age_seconds = params.get("max_age_seconds", 60)
        
        # Check last data timestamp
        last_data_time = await self._get_last_data_time(params)
        if last_data_time is None:
            return True  # No data at all
        
        age = (datetime.utcnow() - last_data_time).total_seconds()
        return age > max_age_seconds
    
    async def _get_metric_value(self, metric: str) -> Optional[float]:
        """Get current value of a metric."""
        # This would fetch from the monitoring system
        return None
    
    async def _get_last_data_time(self, params: Dict) -> Optional[datetime]:
        """Get timestamp of last data received."""
        # This would check data freshness
        return None
    
    async def _execute_action(self, behavior: Behavior):
        """Execute the action for a behavior."""
        action = behavior.action
        
        logger.info(f"Executing behavior: {behavior.name}")
        
        if self.crew.shadow_mode:
            logger.info(f"[SHADOW] Would execute: {action.name}")
            return
        
        try:
            if action.action_type == "tool_call":
                tool_name = action.parameters.get("tool")
                # Execute via crew
                result = self.crew.execute_task(
                    task_name=f"execute_{tool_name}",
                    context=action.parameters
                )
                logger.info(f"Action result: {result}")
                
            elif action.action_type == "log":
                # Log the issue
                logger.info(f"Logging issue: {behavior.name}")
                
            elif action.action_type == "escalate":
                from crewai_integration.crews.data_ops.crew import EscalationEvent, EscalationLevel
                
                level_str = action.parameters.get("level", "medium")
                level = EscalationLevel(level_str)
                
                event = EscalationEvent(
                    level=level,
                    trigger=behavior.name,
                    description=behavior.description
                )
                self.crew.escalate(event)
                
        except Exception as e:
            logger.error(f"Failed to execute action {action.name}: {e}")
            
            if action.escalation_on_failure:
                from crewai_integration.crews.data_ops.crew import EscalationEvent, EscalationLevel
                
                event = EscalationEvent(
                    level=EscalationLevel.HIGH,
                    trigger=f"action_failure:{action.name}",
                    description=f"Autonomous action failed: {e}"
                )
                self.crew.escalate(event)
    
    def get_behavior_status(self) -> Dict[str, Any]:
        """Get status of all behaviors."""
        return {
            "total_behaviors": len(self.behaviors),
            "enabled_behaviors": sum(1 for b in self.behaviors.values() if b.enabled),
            "behaviors": {
                name: {
                    "enabled": b.enabled,
                    "priority": b.priority,
                    "trigger_type": b.trigger.trigger_type.value,
                    "last_triggered": b.trigger.last_triggered.isoformat() if b.trigger.last_triggered else None,
                    "cooldown_seconds": b.trigger.cooldown_seconds
                }
                for name, b in self.behaviors.items()
            }
        }


class EventDrivenBehavior:
    """
    Handles event-driven behaviors that respond to specific events.
    
    Unlike the polling-based AutonomousBehaviorEngine, this class
    listens for events and responds immediately.
    """
    
    def __init__(self, crew, behavior_engine: AutonomousBehaviorEngine):
        """
        Initialize event-driven behavior handler.
        
        Args:
            crew: The DataOperationsCrew instance
            behavior_engine: The autonomous behavior engine
        """
        self.crew = crew
        self.behavior_engine = behavior_engine
        self._event_handlers: Dict[str, List[Behavior]] = {}
        
        # Register event handlers from behaviors
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register behaviors as event handlers."""
        for behavior in self.behavior_engine.behaviors.values():
            if behavior.trigger.trigger_type == TriggerType.EVENT:
                event_type = behavior.trigger.parameters.get("event_type")
                if event_type:
                    if event_type not in self._event_handlers:
                        self._event_handlers[event_type] = []
                    self._event_handlers[event_type].append(behavior)
    
    async def handle_event(self, event: Dict[str, Any]):
        """
        Handle an incoming event.
        
        Args:
            event: The event to handle
        """
        event_type = event.get("type")
        handlers = self._event_handlers.get(event_type, [])
        
        for behavior in handlers:
            if behavior.enabled and behavior.trigger.can_trigger():
                await self.behavior_engine._execute_action(behavior)
                behavior.trigger.mark_triggered()
