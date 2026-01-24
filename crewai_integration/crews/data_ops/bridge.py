"""
Phase 2: Streaming Controller Bridge
====================================

This module provides the bridge between ProductionStreamingController 
and the Data Operations Crew, enabling real-time event-driven processing.

The bridge:
1. Subscribes to streaming controller events
2. Forwards data to validation agents
3. Handles disconnect/error events
4. Triggers autonomous responses
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Set
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import weakref

logger = logging.getLogger(__name__)


class StreamingEventType(Enum):
    """Types of streaming events."""
    DATA_RECEIVED = "data_received"
    EXCHANGE_CONNECTED = "exchange_connected"
    EXCHANGE_DISCONNECTED = "exchange_disconnected"
    EXCHANGE_ERROR = "exchange_error"
    STREAM_STARTED = "stream_started"
    STREAM_STOPPED = "stream_stopped"
    HEALTH_UPDATE = "health_update"
    BACKFILL_COMPLETE = "backfill_complete"
    DATA_GAP_DETECTED = "data_gap_detected"


@dataclass
class StreamingEvent:
    """Represents a streaming event."""
    event_type: StreamingEventType
    exchange: str
    symbol: Optional[str] = None
    data_type: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None


class StreamingControllerBridge:
    """
    Bridge between ProductionStreamingController and Data Operations Crew.
    
    This bridge:
    - Subscribes to streaming controller events
    - Forwards incoming data to the Data Validator for validation
    - Notifies Data Collector of connection status changes
    - Triggers autonomous behaviors on errors/disconnects
    
    Usage:
        controller = ProductionStreamingController()
        crew = DataOperationsCrew()
        bridge = StreamingControllerBridge(controller, crew)
        await bridge.start()
    """
    
    def __init__(
        self,
        streaming_controller=None,
        data_ops_crew=None,
        event_bus=None
    ):
        """
        Initialize the bridge.
        
        Args:
            streaming_controller: ProductionStreamingController instance
            data_ops_crew: DataOperationsCrew instance
            event_bus: Optional EventBus for broader event distribution
        """
        self._controller = streaming_controller
        self._crew = data_ops_crew
        self._event_bus = event_bus
        
        # Use weak references to avoid circular dependencies
        self._controller_ref = weakref.ref(streaming_controller) if streaming_controller else None
        self._crew_ref = weakref.ref(data_ops_crew) if data_ops_crew else None
        
        # State tracking
        self._running = False
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._events_processed = 0
        self._events_by_type: Dict[StreamingEventType, int] = {}
        self._last_event_time: Optional[datetime] = None
        
        # Callbacks registry
        self._callbacks: Dict[StreamingEventType, List[Callable]] = {
            event_type: [] for event_type in StreamingEventType
        }
        
        # Exchange status tracking
        self._exchange_status: Dict[str, Dict[str, Any]] = {}
        
        logger.info("StreamingControllerBridge initialized")
    
    @property
    def controller(self):
        """Get the streaming controller (safely via weak reference)."""
        if self._controller_ref:
            return self._controller_ref()
        return self._controller
    
    @property
    def crew(self):
        """Get the data ops crew (safely via weak reference)."""
        if self._crew_ref:
            return self._crew_ref()
        return self._crew
    
    def set_streaming_controller(self, controller):
        """Set or update the streaming controller reference."""
        self._controller = controller
        self._controller_ref = weakref.ref(controller) if controller else None
        
        # Register callbacks with the controller
        if controller:
            self._register_controller_callbacks()
    
    def set_data_ops_crew(self, crew):
        """Set or update the data ops crew reference."""
        self._crew = crew
        self._crew_ref = weakref.ref(crew) if crew else None
        
        # Inject bridge into crew's tool wrappers
        if crew:
            self._inject_into_crew()
    
    def set_event_bus(self, event_bus):
        """Set the event bus for broader event distribution."""
        self._event_bus = event_bus
    
    def _register_controller_callbacks(self):
        """Register callbacks with the streaming controller."""
        controller = self.controller
        if not controller:
            logger.warning("Cannot register callbacks - no controller set")
            return
        
        try:
            # Register data callbacks
            if hasattr(controller, 'register_price_callback'):
                controller.register_price_callback(self._on_price_data)
            if hasattr(controller, 'register_trade_callback'):
                controller.register_trade_callback(self._on_trade_data)
            if hasattr(controller, 'register_orderbook_callback'):
                controller.register_orderbook_callback(self._on_orderbook_data)
            
            # Register health callbacks
            if hasattr(controller, 'on_exchange_connected'):
                controller.on_exchange_connected = self._on_exchange_connected
            if hasattr(controller, 'on_exchange_disconnected'):
                controller.on_exchange_disconnected = self._on_exchange_disconnected
            if hasattr(controller, 'on_error'):
                controller.on_error = self._on_controller_error
            
            logger.info("Registered callbacks with streaming controller")
            
        except Exception as e:
            logger.error(f"Error registering controller callbacks: {e}")
    
    def _inject_into_crew(self):
        """Inject bridge into crew's tool wrappers."""
        crew = self.crew
        if not crew:
            return
        
        try:
            # Set streaming controller in collector tools
            if hasattr(crew, 'collector_tools'):
                crew.collector_tools.set_streaming_controller(self.controller)
            
            logger.info("Injected streaming controller into crew tools")
            
        except Exception as e:
            logger.error(f"Error injecting into crew: {e}")
    
    async def start(self):
        """Start the bridge event processing."""
        if self._running:
            logger.warning("Bridge already running")
            return
        
        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        logger.info("StreamingControllerBridge started")
    
    async def stop(self):
        """Stop the bridge event processing."""
        self._running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"StreamingControllerBridge stopped (processed {self._events_processed} events)")
    
    async def _process_events(self):
        """Background task to process events from the queue."""
        while self._running:
            try:
                # Wait for event with timeout
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=1.0
                    )
                    await self._handle_event(event)
                except asyncio.TimeoutError:
                    continue
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _handle_event(self, event: StreamingEvent):
        """Handle a streaming event."""
        self._events_processed += 1
        self._last_event_time = datetime.now(timezone.utc)
        
        # Track event types
        self._events_by_type[event.event_type] = \
            self._events_by_type.get(event.event_type, 0) + 1
        
        try:
            # Route to appropriate handler
            if event.event_type == StreamingEventType.DATA_RECEIVED:
                await self._handle_data_received(event)
            elif event.event_type == StreamingEventType.EXCHANGE_CONNECTED:
                await self._handle_exchange_connected(event)
            elif event.event_type == StreamingEventType.EXCHANGE_DISCONNECTED:
                await self._handle_exchange_disconnected(event)
            elif event.event_type == StreamingEventType.EXCHANGE_ERROR:
                await self._handle_exchange_error(event)
            elif event.event_type == StreamingEventType.DATA_GAP_DETECTED:
                await self._handle_data_gap(event)
            
            # Call registered callbacks
            for callback in self._callbacks.get(event.event_type, []):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            # Publish to event bus if available
            if self._event_bus:
                await self._publish_to_event_bus(event)
                
        except Exception as e:
            logger.error(f"Error handling event {event.event_type}: {e}")
    
    async def _handle_data_received(self, event: StreamingEvent):
        """Handle incoming data by forwarding to validators."""
        crew = self.crew
        if not crew:
            return
        
        # Forward to validator tools for validation
        if hasattr(crew, 'validator_tools') and event.data_type == 'price':
            price = event.data.get('price', 0)
            if price > 0:
                result = crew.validator_tools.validate_price_data(
                    exchange=event.exchange,
                    symbol=event.symbol,
                    price=price
                )
                
                # Check for validation issues
                if not result.get('valid', True):
                    logger.warning(
                        f"Validation failed for {event.exchange}/{event.symbol}: "
                        f"{result.get('issues', [])}"
                    )
                    
                    # Trigger crew escalation if critical
                    issues = result.get('issues', [])
                    if any(i.get('type') == 'price_deviation' for i in issues):
                        await self._trigger_crew_event('data_anomaly', {
                            'exchange': event.exchange,
                            'symbol': event.symbol,
                            'issues': issues
                        })
    
    async def _handle_exchange_connected(self, event: StreamingEvent):
        """Handle exchange connection event."""
        self._exchange_status[event.exchange] = {
            'connected': True,
            'connected_at': event.timestamp,
            'error': None
        }
        
        crew = self.crew
        if crew and hasattr(crew, 'collector_tools'):
            logger.info(f"Exchange connected: {event.exchange}")
    
    async def _handle_exchange_disconnected(self, event: StreamingEvent):
        """Handle exchange disconnection by notifying collector agent."""
        self._exchange_status[event.exchange] = {
            'connected': False,
            'disconnected_at': event.timestamp,
            'error': event.error
        }
        
        crew = self.crew
        if not crew:
            return
        
        logger.warning(f"Exchange disconnected: {event.exchange}")
        
        # Trigger autonomous reconnection behavior
        await self._trigger_crew_event('exchange_disconnected', {
            'exchange': event.exchange,
            'error': event.error
        })
    
    async def _handle_exchange_error(self, event: StreamingEvent):
        """Handle exchange error by triggering appropriate response."""
        self._exchange_status[event.exchange] = {
            **self._exchange_status.get(event.exchange, {}),
            'last_error': event.error,
            'error_at': event.timestamp
        }
        
        logger.error(f"Exchange error on {event.exchange}: {event.error}")
        
        # Trigger escalation for critical errors
        await self._trigger_crew_event('exchange_error', {
            'exchange': event.exchange,
            'error': event.error,
            'severity': 'high'
        })
    
    async def _handle_data_gap(self, event: StreamingEvent):
        """Handle detected data gap by triggering backfill."""
        crew = self.crew
        if not crew:
            return
        
        logger.warning(
            f"Data gap detected: {event.exchange}/{event.symbol} "
            f"({event.data_type})"
        )
        
        # Trigger autonomous backfill behavior
        await self._trigger_crew_event('data_gap_detected', {
            'exchange': event.exchange,
            'symbol': event.symbol,
            'data_type': event.data_type,
            'gap_info': event.data
        })
    
    async def _trigger_crew_event(self, event_name: str, data: Dict[str, Any]):
        """Trigger an event for the crew's autonomous behavior engine."""
        crew = self.crew
        if not crew:
            return
        
        # If crew has behavior engine, queue the event
        if hasattr(crew, '_event_queue'):
            await crew._event_queue.put({
                'event': event_name,
                'data': data,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
    
    async def _publish_to_event_bus(self, event: StreamingEvent):
        """Publish event to the central event bus."""
        if not self._event_bus:
            return
        
        try:
            from crewai_integration.events.bus import Event, EventType
            
            # Map streaming events to event bus types
            event_type_map = {
                StreamingEventType.DATA_RECEIVED: EventType.DATA_RECEIVED,
                StreamingEventType.EXCHANGE_CONNECTED: EventType.STREAMING_STARTED,
                StreamingEventType.EXCHANGE_DISCONNECTED: EventType.STREAMING_STOPPED,
                StreamingEventType.EXCHANGE_ERROR: EventType.STREAMING_ERROR,
                StreamingEventType.DATA_GAP_DETECTED: EventType.DATA_GAP_DETECTED,
            }
            
            bus_event_type = event_type_map.get(event.event_type, EventType.CUSTOM)
            
            bus_event = Event(
                type=bus_event_type,
                source=f"streaming_bridge:{event.exchange}",
                data={
                    'exchange': event.exchange,
                    'symbol': event.symbol,
                    'data_type': event.data_type,
                    **event.data
                }
            )
            
            await self._event_bus.publish(bus_event)
            
        except Exception as e:
            logger.error(f"Error publishing to event bus: {e}")
    
    # === Callback Methods for Streaming Controller ===
    
    def _on_price_data(self, exchange: str, symbol: str, data: Dict[str, Any]):
        """Callback for price data from streaming controller."""
        event = StreamingEvent(
            event_type=StreamingEventType.DATA_RECEIVED,
            exchange=exchange,
            symbol=symbol,
            data_type='price',
            data=data
        )
        self._queue_event(event)
    
    def _on_trade_data(self, exchange: str, symbol: str, data: Dict[str, Any]):
        """Callback for trade data from streaming controller."""
        event = StreamingEvent(
            event_type=StreamingEventType.DATA_RECEIVED,
            exchange=exchange,
            symbol=symbol,
            data_type='trade',
            data=data
        )
        self._queue_event(event)
    
    def _on_orderbook_data(self, exchange: str, symbol: str, data: Dict[str, Any]):
        """Callback for orderbook data from streaming controller."""
        event = StreamingEvent(
            event_type=StreamingEventType.DATA_RECEIVED,
            exchange=exchange,
            symbol=symbol,
            data_type='orderbook',
            data=data
        )
        self._queue_event(event)
    
    def _on_exchange_connected(self, exchange: str):
        """Callback for exchange connection."""
        event = StreamingEvent(
            event_type=StreamingEventType.EXCHANGE_CONNECTED,
            exchange=exchange
        )
        self._queue_event(event)
    
    def _on_exchange_disconnected(self, exchange: str):
        """Callback for exchange disconnection."""
        event = StreamingEvent(
            event_type=StreamingEventType.EXCHANGE_DISCONNECTED,
            exchange=exchange
        )
        self._queue_event(event)
    
    def _on_controller_error(self, exchange: str, error: str):
        """Callback for controller errors."""
        event = StreamingEvent(
            event_type=StreamingEventType.EXCHANGE_ERROR,
            exchange=exchange,
            error=error
        )
        self._queue_event(event)
    
    def _queue_event(self, event: StreamingEvent):
        """Queue an event for processing (non-blocking)."""
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping event: {event.event_type}")
    
    # === External API ===
    
    def register_callback(
        self,
        event_type: StreamingEventType,
        callback: Callable
    ):
        """Register a callback for a specific event type."""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
    
    def unregister_callback(
        self,
        event_type: StreamingEventType,
        callback: Callable
    ):
        """Unregister a callback."""
        if event_type in self._callbacks:
            if callback in self._callbacks[event_type]:
                self._callbacks[event_type].remove(callback)
    
    def get_exchange_status(self, exchange: str) -> Dict[str, Any]:
        """Get current status of an exchange."""
        return self._exchange_status.get(exchange, {
            'connected': False,
            'error': 'Status unknown'
        })
    
    def get_all_exchange_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tracked exchanges."""
        return dict(self._exchange_status)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get bridge metrics."""
        return {
            'running': self._running,
            'events_processed': self._events_processed,
            'events_by_type': {
                k.value: v for k, v in self._events_by_type.items()
            },
            'last_event_time': self._last_event_time.isoformat() if self._last_event_time else None,
            'exchange_status': self.get_all_exchange_status(),
            'queue_size': self._event_queue.qsize()
        }


def create_bridge(
    streaming_controller=None,
    data_ops_crew=None,
    event_bus=None
) -> StreamingControllerBridge:
    """
    Factory function to create a configured bridge.
    
    Usage:
        from src.streaming.production_controller import ProductionStreamingController
        from crewai_integration.crews.data_ops.crew import DataOperationsCrew
        from crewai_integration.events.bus import EventBus
        
        controller = ProductionStreamingController()
        crew = DataOperationsCrew()
        event_bus = EventBus()
        
        bridge = create_bridge(controller, crew, event_bus)
        await bridge.start()
    """
    bridge = StreamingControllerBridge(
        streaming_controller=streaming_controller,
        data_ops_crew=data_ops_crew,
        event_bus=event_bus
    )
    return bridge
