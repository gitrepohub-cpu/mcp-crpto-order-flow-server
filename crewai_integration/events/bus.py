"""
Event Bus for CrewAI Integration
================================

A lightweight event system that enables communication between:
- The MCP streaming system (data events)
- CrewAI agents (processing events)
- External triggers (scheduled events)

This allows loose coupling between components while maintaining
real-time responsiveness.
"""

import logging
import asyncio
from typing import Dict, Any, List, Callable, Optional, Awaitable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events in the system."""
    # Data Events
    DATA_RECEIVED = "data.received"
    DATA_QUALITY_ALERT = "data.quality_alert"
    DATA_GAP_DETECTED = "data.gap_detected"
    
    # Streaming Events
    STREAMING_STARTED = "streaming.started"
    STREAMING_STOPPED = "streaming.stopped"
    STREAMING_ERROR = "streaming.error"
    STREAMING_HEALTH_CHANGE = "streaming.health_change"
    
    # Analysis Events
    FORECAST_GENERATED = "analysis.forecast_generated"
    REGIME_CHANGE_DETECTED = "analysis.regime_change"
    ANOMALY_DETECTED = "analysis.anomaly_detected"
    
    # Intelligence Events
    SMART_MONEY_SIGNAL = "intelligence.smart_money"
    SQUEEZE_ALERT = "intelligence.squeeze_alert"
    LIQUIDATION_RISK = "intelligence.liquidation_risk"
    
    # Agent Events
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    AGENT_ERROR = "agent.error"
    AGENT_DECISION = "agent.decision"
    AGENT_RECOMMENDATION = "agent.recommendation"
    
    # System Events
    SYSTEM_HEALTH_CHANGE = "system.health_change"
    CONFIG_CHANGED = "system.config_changed"
    HUMAN_APPROVAL_REQUIRED = "system.approval_required"
    
    # Custom Events
    CUSTOM = "custom"


@dataclass
class Event:
    """An event in the system."""
    type: EventType
    source: str
    data: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: int = 5  # 1-10, higher = more urgent
    correlation_id: Optional[str] = None  # For tracking related events
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata
        }


# Type alias for event handlers
EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """
    Asynchronous event bus for inter-component communication.
    
    Features:
    - Subscribe to specific event types or patterns
    - Async event processing
    - Event history for debugging
    - Priority-based processing
    - Correlation tracking
    
    Usage:
        bus = EventBus()
        
        async def my_handler(event: Event):
            print(f"Received: {event.type}")
        
        bus.subscribe(EventType.DATA_RECEIVED, my_handler)
        
        await bus.publish(Event(
            type=EventType.DATA_RECEIVED,
            source="binance_stream",
            data={"symbol": "BTCUSDT", "price": 50000}
        ))
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize the event bus.
        
        Args:
            max_history: Maximum number of events to keep in history
        """
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._wildcard_handlers: List[EventHandler] = []
        self._history: List[Event] = []
        self._max_history = max_history
        self._running = False
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
        self._event_count = 0
        self._error_count = 0
    
    def subscribe(
        self,
        event_type: Optional[EventType],
        handler: EventHandler
    ):
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to (None for all events)
            handler: Async function to handle events
        """
        if event_type is None:
            if handler not in self._wildcard_handlers:
                self._wildcard_handlers.append(handler)
                logger.debug(f"Added wildcard handler: {handler.__name__}")
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            if handler not in self._handlers[event_type]:
                self._handlers[event_type].append(handler)
                logger.debug(f"Subscribed handler to {event_type.value}: {handler.__name__}")
    
    def unsubscribe(
        self,
        event_type: Optional[EventType],
        handler: EventHandler
    ):
        """Unsubscribe a handler from events."""
        if event_type is None:
            if handler in self._wildcard_handlers:
                self._wildcard_handlers.remove(handler)
        else:
            if event_type in self._handlers and handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
    
    async def publish(self, event: Event):
        """
        Publish an event to the bus.
        
        Args:
            event: The event to publish
        """
        # Add to queue for async processing
        await self._event_queue.put(event)
        
        # Also process immediately if not using background processing
        if not self._running:
            await self._process_event(event)
    
    def publish_sync(self, event: Event):
        """
        Synchronously queue an event (for non-async contexts).
        
        Args:
            event: The event to publish
        """
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping event: {event.id}")
    
    async def _process_event(self, event: Event):
        """Process a single event by calling all relevant handlers."""
        self._event_count += 1
        
        # Add to history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        
        # Get handlers for this event type
        handlers: List[EventHandler] = []
        handlers.extend(self._wildcard_handlers)
        if event.type in self._handlers:
            handlers.extend(self._handlers[event.type])
        
        if not handlers:
            logger.debug(f"No handlers for event type: {event.type.value}")
            return
        
        # Call handlers (sorted by priority for the event)
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                self._error_count += 1
                logger.error(f"Handler {handler.__name__} failed for event {event.id}: {e}")
    
    async def start(self):
        """Start background event processing."""
        if self._running:
            return
        
        self._running = True
        self._processor_task = asyncio.create_task(self._process_loop())
        logger.info("Event bus started")
    
    async def stop(self):
        """Stop background event processing."""
        self._running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None
        
        logger.info(f"Event bus stopped (processed {self._event_count} events)")
    
    async def _process_loop(self):
        """Background loop for processing events."""
        while self._running:
            try:
                # Wait for event with timeout
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=1.0
                    )
                    await self._process_event(event)
                except asyncio.TimeoutError:
                    continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._error_count += 1
                logger.error(f"Error in event processing loop: {e}")
    
    def get_history(
        self,
        event_type: Optional[EventType] = None,
        source: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type
            source: Filter by source
            limit: Maximum events to return
            
        Returns:
            List of event dictionaries
        """
        events = self._history
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        if source:
            events = [e for e in events if e.source == source]
        
        return [e.to_dict() for e in events[-limit:]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        type_counts = {}
        for event in self._history:
            type_name = event.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            "running": self._running,
            "total_events": self._event_count,
            "error_count": self._error_count,
            "history_size": len(self._history),
            "queue_size": self._event_queue.qsize(),
            "handler_count": sum(len(h) for h in self._handlers.values()) + len(self._wildcard_handlers),
            "events_by_type": type_counts
        }
    
    def clear_history(self):
        """Clear event history."""
        self._history.clear()


def create_data_event(
    source: str,
    symbol: str,
    exchange: str,
    data_type: str,
    data: Dict[str, Any]
) -> Event:
    """Helper to create a data received event."""
    return Event(
        type=EventType.DATA_RECEIVED,
        source=source,
        data={
            "symbol": symbol,
            "exchange": exchange,
            "data_type": data_type,
            **data
        }
    )


def create_alert_event(
    source: str,
    alert_type: str,
    message: str,
    severity: str = "warning",
    data: Optional[Dict[str, Any]] = None
) -> Event:
    """Helper to create an alert event."""
    event_type_map = {
        "data_quality": EventType.DATA_QUALITY_ALERT,
        "data_gap": EventType.DATA_GAP_DETECTED,
        "anomaly": EventType.ANOMALY_DETECTED,
        "squeeze": EventType.SQUEEZE_ALERT,
        "liquidation": EventType.LIQUIDATION_RISK,
    }
    
    return Event(
        type=event_type_map.get(alert_type, EventType.CUSTOM),
        source=source,
        data={
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            **(data or {})
        },
        priority=8 if severity == "critical" else 6 if severity == "warning" else 4
    )


def create_agent_event(
    agent_id: str,
    event_type: EventType,
    action: str,
    data: Optional[Dict[str, Any]] = None
) -> Event:
    """Helper to create an agent event."""
    return Event(
        type=event_type,
        source=f"agent:{agent_id}",
        data={
            "agent_id": agent_id,
            "action": action,
            **(data or {})
        }
    )
