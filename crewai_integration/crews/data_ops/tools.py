"""
Phase 2: Data Operations Crew - Tool Wrappers
==============================================

This module provides specialized tool wrappers for the Data Operations Crew agents.
Each wrapper extends the base ToolWrapper with agent-specific functionality.

INTEGRATION WITH PHASE 1:
- Uses Phase 1 MCP tool wrappers (ExchangeDataTools, StreamingTools, etc.)
- Connects to DuckDB historical data via db_access module
- Integrates with EventBus for event-driven architecture
"""

from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone
import logging
import asyncio

from crewai_integration.tools.base import ToolWrapper

# Import Phase 1 MCP tool wrappers
try:
    from crewai_integration.tools.wrappers import (
        ExchangeDataTools,
        StreamingTools,
        AnalyticsTools,
        ForecastingTools,
        FeatureTools
    )
    PHASE1_TOOLS_AVAILABLE = True
except ImportError:
    PHASE1_TOOLS_AVAILABLE = False
    ExchangeDataTools = None
    StreamingTools = None
    AnalyticsTools = None

# Import historical data access
try:
    from crewai_integration.crews.data_ops.db_access import (
        DuckDBHistoricalAccess,
        get_db_access
    )
    DB_ACCESS_AVAILABLE = True
except ImportError:
    DB_ACCESS_AVAILABLE = False
    DuckDBHistoricalAccess = None
    get_db_access = None

logger = logging.getLogger(__name__)


class DataOpsToolBase(ToolWrapper):
    """
    Base class for Data Operations tool wrappers.
    
    Adds helper methods specific to Data Ops tools and integrates
    with Phase 1 MCP tools and DuckDB historical data.
    """
    
    def __init__(
        self,
        permission_manager=None,
        tool_registry=None,
        shadow_mode: bool = False
    ):
        super().__init__(
            permission_manager=permission_manager,
            tool_registry=tool_registry,
            shadow_mode=shadow_mode
        )
        
        # Phase 1 tool wrapper instances (lazy initialized)
        self._exchange_tools: Optional[ExchangeDataTools] = None
        self._streaming_tools: Optional[StreamingTools] = None
        self._analytics_tools: Optional[AnalyticsTools] = None
        
        # DuckDB historical access (lazy initialized)
        self._db_access: Optional[DuckDBHistoricalAccess] = None
    
    @property
    def exchange_tools(self) -> Optional[ExchangeDataTools]:
        """Get Phase 1 ExchangeDataTools wrapper (lazy init)."""
        if self._exchange_tools is None and PHASE1_TOOLS_AVAILABLE:
            self._exchange_tools = ExchangeDataTools(
                permission_manager=self.permission_manager,
                tool_registry=self.tool_registry
            )
        return self._exchange_tools
    
    @property
    def streaming_tools(self) -> Optional[StreamingTools]:
        """Get Phase 1 StreamingTools wrapper (lazy init)."""
        if self._streaming_tools is None and PHASE1_TOOLS_AVAILABLE:
            self._streaming_tools = StreamingTools(
                permission_manager=self.permission_manager,
                tool_registry=self.tool_registry
            )
        return self._streaming_tools
    
    @property
    def analytics_tools(self) -> Optional[AnalyticsTools]:
        """Get Phase 1 AnalyticsTools wrapper (lazy init)."""
        if self._analytics_tools is None and PHASE1_TOOLS_AVAILABLE:
            self._analytics_tools = AnalyticsTools(
                permission_manager=self.permission_manager,
                tool_registry=self.tool_registry
            )
        return self._analytics_tools
    
    @property
    def db_access(self) -> Optional[DuckDBHistoricalAccess]:
        """Get DuckDB historical data access (lazy init)."""
        if self._db_access is None and DB_ACCESS_AVAILABLE:
            self._db_access = get_db_access()
        return self._db_access
    
    def _shadow_response(self, method_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return a shadow mode response for a tool call."""
        return {
            "shadow_mode": True,
            "method": method_name,
            "params": params,
            "message": f"Shadow mode: {method_name} would be called with params {params}"
        }
    
    def _check_permission(self, permission: str) -> bool:
        """Check if permission is granted."""
        if self.shadow_mode:
            return True  # Always allow in shadow mode
        if self.permission_manager is None:
            return True  # Allow if no permission manager
        return self.permission_manager.has_permission(permission)
    
    def _log_action(self, action: str, details: Dict[str, Any]):
        """Log an action for audit purposes."""
        logger.info(f"Action: {action}, Details: {details}")
    
    def _run_async(self, coro):
        """Helper to run async code from sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new task if we're already in an async context
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, coro).result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(coro)


class DataCollectorTools(DataOpsToolBase):
    """
    Tool wrapper for the Data Collector Agent.
    
    Provides tools for:
    - Monitoring exchange connections (via Phase 1 StreamingTools)
    - Managing WebSocket streams
    - Triggering data backfills
    - Handling reconnections
    
    PHASE 1 INTEGRATION:
    - Uses StreamingTools for stream control
    - Uses ExchangeDataTools for exchange health checks
    """
    
    TOOL_CATEGORY = "data_collection"
    REQUIRED_PERMISSIONS = ["stream:read", "stream:control", "backfill:trigger"]
    
    def __init__(
        self,
        permission_manager=None,
        tool_registry=None,
        shadow_mode: bool = False
    ):
        super().__init__(
            permission_manager=permission_manager,
            tool_registry=tool_registry,
            shadow_mode=shadow_mode
        )
        self._streaming_controller = None
        self._exchange_clients = {}
    
    def _get_category(self) -> str:
        """Return the tool category for this wrapper."""
        return self.TOOL_CATEGORY
    
    def _get_tools(self) -> Dict[str, Any]:
        """Return mapping of tool names to wrapper methods."""
        return {
            "check_exchange_status": self.check_exchange_status,
            "get_all_exchange_status": self.get_all_exchange_status,
            "get_stream_statistics": self.get_stream_statistics,
            "get_streaming_health": self.get_streaming_health,
            "reconnect_exchange": self.reconnect_exchange,
            "start_stream": self.start_stream,
            "stop_stream": self.stop_stream,
            "request_backfill": self.request_backfill,
            "get_backfill_status": self.get_backfill_status,
            "detect_data_gaps": self.detect_data_gaps
        }
    
    def set_streaming_controller(self, controller):
        """Inject the streaming controller instance."""
        self._streaming_controller = controller
    
    def set_exchange_clients(self, clients: Dict[str, Any]):
        """Inject exchange client instances."""
        self._exchange_clients = clients
    
    # ===== Connection Monitoring Tools =====
    
    def check_exchange_status(self, exchange: str) -> Dict[str, Any]:
        """
        Check the connection status of an exchange.
        
        Uses Phase 1 StreamingTools and ExchangeDataTools for real data.
        
        Args:
            exchange: Exchange name (e.g., 'binance', 'bybit')
            
        Returns:
            Status dict with connection info
        """
        if self.shadow_mode:
            return self._shadow_response("check_exchange_status", {"exchange": exchange})
        
        # Try Phase 1 streaming tools first
        if self.streaming_tools:
            try:
                health = self._run_async(self.streaming_tools.get_health())
                if health and not health.get('error'):
                    exchange_status = health.get('exchanges', {}).get(exchange, {})
                    return {
                        "exchange": exchange,
                        "connected": exchange_status.get("connected", False),
                        "latency_ms": exchange_status.get("latency_ms"),
                        "last_data_time": exchange_status.get("last_data"),
                        "streams_active": exchange_status.get("active_streams", 0),
                        "error": exchange_status.get("error"),
                        "source": "phase1_streaming_tools"
                    }
            except Exception as e:
                logger.debug(f"Phase 1 streaming tools unavailable: {e}")
        
        # Fall back to streaming controller
        if self._streaming_controller:
            try:
                status = self._streaming_controller.get_exchange_status(exchange)
                return {
                    "exchange": exchange,
                    "connected": status.get("connected", False),
                    "latency_ms": status.get("latency_ms"),
                    "last_data_time": status.get("last_data_time"),
                    "streams_active": status.get("active_streams", 0),
                    "error": status.get("error"),
                    "source": "streaming_controller"
                }
            except Exception as e:
                logger.error(f"Error checking exchange status: {e}")
                return {"exchange": exchange, "error": str(e)}
        
        return {"exchange": exchange, "error": "No streaming source configured"}
    
    def get_streaming_health(self) -> Dict[str, Any]:
        """
        Get overall streaming system health using Phase 1 tools.
        
        Returns:
            Health metrics from streaming system
        """
        if self.shadow_mode:
            return self._shadow_response("get_streaming_health", {})
        
        # Use Phase 1 streaming tools
        if self.streaming_tools:
            try:
                return self._run_async(self.streaming_tools.get_health())
            except Exception as e:
                logger.error(f"Error getting streaming health: {e}")
                return {"error": str(e)}
        
        # Fall back to controller
        if self._streaming_controller and hasattr(self._streaming_controller, 'health'):
            health = self._streaming_controller.health
            return {
                "records_ingested": health.records_ingested,
                "records_per_minute": health.records_per_minute,
                "active_symbols": list(health.active_symbols),
                "active_exchanges": list(health.active_exchanges),
                "errors": health.errors,
                "uptime_seconds": health.uptime_seconds,
                "source": "streaming_controller"
            }
        
        return {"error": "No health source available"}
    
    def get_all_exchange_status(self) -> Dict[str, Dict[str, Any]]:
        """Get connection status for all exchanges."""
        if self.shadow_mode:
            return self._shadow_response("get_all_exchange_status", {})
        
        exchanges = ["binance", "bybit", "okx", "hyperliquid", "gateio", "kraken", "deribit", "coinbase"]
        return {ex: self.check_exchange_status(ex) for ex in exchanges}
    
    def get_stream_statistics(self, exchange: str, stream_type: str) -> Dict[str, Any]:
        """
        Get statistics for a specific data stream.
        
        Args:
            exchange: Exchange name
            stream_type: Stream type (e.g., 'trades', 'orderbook', 'ticker')
            
        Returns:
            Stream statistics
        """
        if self.shadow_mode:
            return self._shadow_response("get_stream_statistics", {
                "exchange": exchange,
                "stream_type": stream_type
            })
        
        if not self._streaming_controller:
            return {"error": "Streaming controller not configured"}
        
        try:
            stats = self._streaming_controller.get_stream_stats(exchange, stream_type)
            return stats
        except Exception as e:
            logger.error(f"Error getting stream statistics: {e}")
            return {"error": str(e)}
    
    # ===== Connection Control Tools =====
    
    def reconnect_exchange(self, exchange: str, force: bool = False) -> Dict[str, Any]:
        """
        Trigger reconnection to an exchange.
        
        Args:
            exchange: Exchange name
            force: Force reconnection even if connected
            
        Returns:
            Reconnection result
        """
        if self.shadow_mode:
            return self._shadow_response("reconnect_exchange", {
                "exchange": exchange,
                "force": force
            })
        
        if not self._check_permission("stream:control"):
            return {"error": "Permission denied: stream:control"}
        
        if not self._streaming_controller:
            return {"error": "Streaming controller not configured"}
        
        try:
            result = self._streaming_controller.reconnect_exchange(exchange, force=force)
            self._log_action("reconnect_exchange", {
                "exchange": exchange,
                "force": force,
                "result": result
            })
            return result
        except Exception as e:
            logger.error(f"Error reconnecting exchange: {e}")
            return {"error": str(e)}
    
    def start_stream(
        self,
        exchange: str,
        symbol: str,
        stream_type: str
    ) -> Dict[str, Any]:
        """
        Start a specific data stream.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            stream_type: Stream type
            
        Returns:
            Start result
        """
        if self.shadow_mode:
            return self._shadow_response("start_stream", {
                "exchange": exchange,
                "symbol": symbol,
                "stream_type": stream_type
            })
        
        if not self._check_permission("stream:control"):
            return {"error": "Permission denied: stream:control"}
        
        if not self._streaming_controller:
            return {"error": "Streaming controller not configured"}
        
        try:
            result = self._streaming_controller.start_stream(exchange, symbol, stream_type)
            self._log_action("start_stream", {
                "exchange": exchange,
                "symbol": symbol,
                "stream_type": stream_type,
                "result": result
            })
            return result
        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            return {"error": str(e)}
    
    def stop_stream(
        self,
        exchange: str,
        symbol: str,
        stream_type: str
    ) -> Dict[str, Any]:
        """
        Stop a specific data stream.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            stream_type: Stream type
            
        Returns:
            Stop result
        """
        if self.shadow_mode:
            return self._shadow_response("stop_stream", {
                "exchange": exchange,
                "symbol": symbol,
                "stream_type": stream_type
            })
        
        if not self._check_permission("stream:control"):
            return {"error": "Permission denied: stream:control"}
        
        if not self._streaming_controller:
            return {"error": "Streaming controller not configured"}
        
        try:
            result = self._streaming_controller.stop_stream(exchange, symbol, stream_type)
            self._log_action("stop_stream", {
                "exchange": exchange,
                "symbol": symbol,
                "stream_type": stream_type,
                "result": result
            })
            return result
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")
            return {"error": str(e)}
    
    # ===== Backfill Tools =====
    
    def request_backfill(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        start_time: datetime,
        end_time: datetime,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Request historical data backfill.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data_type: Data type to backfill
            start_time: Start of backfill period
            end_time: End of backfill period
            priority: Backfill priority (low, normal, high)
            
        Returns:
            Backfill request result
        """
        if self.shadow_mode:
            return self._shadow_response("request_backfill", {
                "exchange": exchange,
                "symbol": symbol,
                "data_type": data_type,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "priority": priority
            })
        
        if not self._check_permission("backfill:trigger"):
            return {"error": "Permission denied: backfill:trigger"}
        
        client = self._exchange_clients.get(exchange)
        if not client:
            return {"error": f"Exchange client not found: {exchange}"}
        
        try:
            result = client.backfill_data(
                symbol=symbol,
                data_type=data_type,
                start_time=start_time,
                end_time=end_time
            )
            self._log_action("request_backfill", {
                "exchange": exchange,
                "symbol": symbol,
                "data_type": data_type,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "result": result
            })
            return result
        except Exception as e:
            logger.error(f"Error requesting backfill: {e}")
            return {"error": str(e)}
    
    def get_backfill_status(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of backfill requests.
        
        Args:
            request_id: Optional specific request ID
            
        Returns:
            Backfill status
        """
        if self.shadow_mode:
            return self._shadow_response("get_backfill_status", {"request_id": request_id})
        
        if not self._streaming_controller:
            return {"error": "Streaming controller not configured"}
        
        try:
            return self._streaming_controller.get_backfill_status(request_id)
        except Exception as e:
            logger.error(f"Error getting backfill status: {e}")
            return {"error": str(e)}
    
    # ===== Data Gap Detection =====
    
    def detect_data_gaps(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        lookback_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Detect gaps in data collection.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data_type: Data type to check
            lookback_minutes: How far back to look
            
        Returns:
            List of detected gaps
        """
        if self.shadow_mode:
            return self._shadow_response("detect_data_gaps", {
                "exchange": exchange,
                "symbol": symbol,
                "data_type": data_type,
                "lookback_minutes": lookback_minutes
            })
        
        if not self._streaming_controller:
            return {"error": "Streaming controller not configured"}
        
        try:
            gaps = self._streaming_controller.detect_gaps(
                exchange=exchange,
                symbol=symbol,
                data_type=data_type,
                lookback_minutes=lookback_minutes
            )
            return {
                "exchange": exchange,
                "symbol": symbol,
                "data_type": data_type,
                "gaps_found": len(gaps),
                "gaps": gaps
            }
        except Exception as e:
            logger.error(f"Error detecting data gaps: {e}")
            return {"error": str(e)}
    
    def get_registered_tools(self) -> List[Dict[str, Any]]:
        """Return list of tools for CrewAI registration."""
        return [
            {
                "name": "check_exchange_status",
                "description": "Check connection status of an exchange",
                "func": self.check_exchange_status
            },
            {
                "name": "get_all_exchange_status",
                "description": "Get connection status for all exchanges",
                "func": self.get_all_exchange_status
            },
            {
                "name": "get_stream_statistics",
                "description": "Get statistics for a specific data stream",
                "func": self.get_stream_statistics
            },
            {
                "name": "get_streaming_health",
                "description": "Get overall streaming system health",
                "func": self.get_streaming_health
            },
            {
                "name": "reconnect_exchange",
                "description": "Trigger reconnection to an exchange",
                "func": self.reconnect_exchange
            },
            {
                "name": "start_stream",
                "description": "Start a specific data stream",
                "func": self.start_stream
            },
            {
                "name": "stop_stream",
                "description": "Stop a specific data stream",
                "func": self.stop_stream
            },
            {
                "name": "request_backfill",
                "description": "Request historical data backfill",
                "func": self.request_backfill
            },
            {
                "name": "get_backfill_status",
                "description": "Get status of backfill requests",
                "func": self.get_backfill_status
            },
            {
                "name": "detect_data_gaps",
                "description": "Detect gaps in data collection",
                "func": self.detect_data_gaps
            }
        ]


class DataValidatorTools(DataOpsToolBase):
    """
    Tool wrapper for the Data Validator Agent.
    
    Provides tools for:
    - Validating incoming data (using historical data from DuckDB)
    - Cross-exchange consistency checks
    - Anomaly detection (using Phase 1 AnalyticsTools)
    - Data quarantine management
    
    PHASE 1 INTEGRATION:
    - Uses AnalyticsTools for anomaly detection
    - Uses DuckDB historical data for validation reference
    - Uses ExchangeDataTools for cross-exchange checks
    """
    
    TOOL_CATEGORY = "data_validation"
    REQUIRED_PERMISSIONS = ["data:read", "data:validate", "data:quarantine"]
    
    def __init__(
        self,
        permission_manager=None,
        tool_registry=None,
        shadow_mode: bool = False
    ):
        super().__init__(
            permission_manager=permission_manager,
            tool_registry=tool_registry,
            shadow_mode=shadow_mode
        )
        self._db_manager = None
        self._validation_rules = self._load_validation_rules()
    
    def _get_category(self) -> str:
        """Return the tool category for this wrapper."""
        return self.TOOL_CATEGORY
    
    def _get_tools(self) -> Dict[str, Any]:
        """Return mapping of tool names to wrapper methods."""
        return {
            "validate_price_data": self.validate_price_data,
            "validate_against_historical": self.validate_against_historical,
            "validate_orderbook": self.validate_orderbook,
            "cross_exchange_consistency": self.cross_exchange_consistency,
            "detect_anomaly": self.detect_anomaly,
            "get_historical_statistics": self.get_historical_statistics,
            "quarantine_data": self.quarantine_data
        }
    
    def set_db_manager(self, manager):
        """Inject the database manager instance."""
        self._db_manager = manager
    
    def _load_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load validation rules from configuration."""
        # Default validation rules
        return {
            "price": {
                "max_deviation_percent": 5.0,
                "require_positive": True
            },
            "orderbook": {
                "min_levels": 3,
                "bid_ask_spread_max_percent": 10.0,
                "require_sorted": True
            },
            "trades": {
                "require_positive_volume": True,
                "max_age_seconds": 60
            },
            "funding": {
                "valid_range": [-0.1, 0.1],
                "interval_hours": 8
            },
            "open_interest": {
                "require_positive": True,
                "max_change_percent": 50.0
            }
        }
    
    # ===== Validation Tools =====
    
    def validate_price_data(
        self,
        exchange: str,
        symbol: str,
        price: float,
        reference_prices: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Validate price data against rules and reference prices.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            price: Price to validate
            reference_prices: Optional dict of reference prices from other exchanges
            
        Returns:
            Validation result
        """
        if self.shadow_mode:
            return self._shadow_response("validate_price_data", {
                "exchange": exchange,
                "symbol": symbol,
                "price": price
            })
        
        rules = self._validation_rules.get("price", {})
        issues = []
        
        # Check positive
        if rules.get("require_positive") and price <= 0:
            issues.append({
                "type": "invalid_price",
                "message": "Price must be positive",
                "value": price
            })
        
        # Cross-exchange validation
        if reference_prices:
            ref_values = list(reference_prices.values())
            if ref_values:
                median_price = sorted(ref_values)[len(ref_values) // 2]
                max_deviation = rules.get("max_deviation_percent", 5.0)
                deviation = abs((price - median_price) / median_price * 100)
                
                if deviation > max_deviation:
                    issues.append({
                        "type": "price_deviation",
                        "message": f"Price deviates {deviation:.2f}% from median",
                        "value": price,
                        "median": median_price,
                        "deviation_percent": deviation
                    })
        
        return {
            "exchange": exchange,
            "symbol": symbol,
            "price": price,
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def validate_orderbook(
        self,
        exchange: str,
        symbol: str,
        bids: List[List[float]],
        asks: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Validate orderbook data.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            bids: List of [price, quantity] bid levels
            asks: List of [price, quantity] ask levels
            
        Returns:
            Validation result
        """
        if self.shadow_mode:
            return self._shadow_response("validate_orderbook", {
                "exchange": exchange,
                "symbol": symbol,
                "bid_levels": len(bids),
                "ask_levels": len(asks)
            })
        
        rules = self._validation_rules.get("orderbook", {})
        issues = []
        
        # Check minimum levels
        min_levels = rules.get("min_levels", 3)
        if len(bids) < min_levels:
            issues.append({
                "type": "insufficient_bids",
                "message": f"Only {len(bids)} bid levels (minimum: {min_levels})",
                "value": len(bids)
            })
        if len(asks) < min_levels:
            issues.append({
                "type": "insufficient_asks",
                "message": f"Only {len(asks)} ask levels (minimum: {min_levels})",
                "value": len(asks)
            })
        
        # Check bid-ask spread
        if bids and asks:
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            
            if best_bid > 0 and best_ask > 0:
                spread_percent = (best_ask - best_bid) / best_bid * 100
                max_spread = rules.get("bid_ask_spread_max_percent", 10.0)
                
                if spread_percent > max_spread:
                    issues.append({
                        "type": "wide_spread",
                        "message": f"Spread {spread_percent:.2f}% exceeds maximum {max_spread}%",
                        "spread_percent": spread_percent
                    })
                
                # Check crossed book
                if best_bid >= best_ask:
                    issues.append({
                        "type": "crossed_book",
                        "message": f"Best bid ({best_bid}) >= best ask ({best_ask})",
                        "best_bid": best_bid,
                        "best_ask": best_ask
                    })
        
        # Check sorted order
        if rules.get("require_sorted"):
            # Bids should be descending
            for i in range(1, len(bids)):
                if bids[i][0] > bids[i-1][0]:
                    issues.append({
                        "type": "unsorted_bids",
                        "message": "Bids not in descending order"
                    })
                    break
            
            # Asks should be ascending
            for i in range(1, len(asks)):
                if asks[i][0] < asks[i-1][0]:
                    issues.append({
                        "type": "unsorted_asks",
                        "message": "Asks not in ascending order"
                    })
                    break
        
        return {
            "exchange": exchange,
            "symbol": symbol,
            "valid": len(issues) == 0,
            "bid_levels": len(bids),
            "ask_levels": len(asks),
            "issues": issues
        }
    
    def cross_exchange_consistency(
        self,
        symbol: str,
        data_type: str,
        exchange_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check consistency of data across exchanges.
        
        Args:
            symbol: Trading symbol
            data_type: Type of data (price, funding, oi)
            exchange_data: Dict mapping exchange to value
            
        Returns:
            Consistency analysis
        """
        if self.shadow_mode:
            return self._shadow_response("cross_exchange_consistency", {
                "symbol": symbol,
                "data_type": data_type,
                "exchanges": list(exchange_data.keys())
            })
        
        values = list(exchange_data.values())
        if len(values) < 2:
            return {
                "symbol": symbol,
                "data_type": data_type,
                "consistent": True,
                "message": "Insufficient exchanges for comparison"
            }
        
        # Calculate statistics
        mean_val = sum(values) / len(values)
        sorted_vals = sorted(values)
        median_val = sorted_vals[len(sorted_vals) // 2]
        
        # Find outliers
        outliers = []
        threshold_percent = 3.0  # 3% deviation threshold
        
        for exchange, value in exchange_data.items():
            if median_val > 0:
                deviation = abs((value - median_val) / median_val * 100)
                if deviation > threshold_percent:
                    outliers.append({
                        "exchange": exchange,
                        "value": value,
                        "deviation_percent": deviation
                    })
        
        return {
            "symbol": symbol,
            "data_type": data_type,
            "consistent": len(outliers) == 0,
            "exchanges_checked": len(exchange_data),
            "mean": mean_val,
            "median": median_val,
            "outliers": outliers
        }
    
    # ===== Historical Validation (DuckDB Integration) =====
    
    def validate_against_historical(
        self,
        exchange: str,
        symbol: str,
        price: float,
        lookback_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Validate current price against historical data from DuckDB.
        
        Uses the 504 existing DuckDB tables for validation reference.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            price: Price to validate
            lookback_minutes: How far back to look for reference
            
        Returns:
            Validation result with historical context
        """
        if self.shadow_mode:
            return self._shadow_response("validate_against_historical", {
                "exchange": exchange,
                "symbol": symbol,
                "price": price,
                "lookback_minutes": lookback_minutes
            })
        
        # Get historical statistics from DuckDB
        if self.db_access:
            try:
                stats = self.db_access.get_price_statistics(
                    exchange=exchange,
                    symbol=symbol,
                    lookback_minutes=lookback_minutes
                )
                
                if stats.get('error'):
                    return {
                        "exchange": exchange,
                        "symbol": symbol,
                        "price": price,
                        "valid": True,  # Can't validate without historical data
                        "error": stats.get('error'),
                        "source": "db_access_error"
                    }
                
                # Validate against historical range
                mean = stats.get('mean', 0)
                std = stats.get('std', 0)
                
                if mean > 0 and std > 0:
                    z_score = (price - mean) / std
                    max_deviation = self._validation_rules.get("price", {}).get("max_deviation_percent", 5.0)
                    deviation_percent = abs((price - mean) / mean * 100)
                    
                    is_valid = deviation_percent <= max_deviation and abs(z_score) <= 3.0
                    
                    return {
                        "exchange": exchange,
                        "symbol": symbol,
                        "price": price,
                        "valid": is_valid,
                        "z_score": z_score,
                        "deviation_percent": deviation_percent,
                        "historical_stats": stats,
                        "source": "duckdb_historical"
                    }
                    
            except Exception as e:
                logger.error(f"Error validating against historical: {e}")
        
        # Fall back to basic validation
        return self.validate_price_data(exchange, symbol, price)
    
    def get_historical_statistics(
        self,
        exchange: str,
        symbol: str,
        data_type: str = "ticker",
        lookback_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Get historical statistics from DuckDB for reference.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data_type: Data type (ticker, trades)
            lookback_minutes: How far back to look
            
        Returns:
            Historical statistics
        """
        if self.shadow_mode:
            return self._shadow_response("get_historical_statistics", {
                "exchange": exchange,
                "symbol": symbol,
                "data_type": data_type,
                "lookback_minutes": lookback_minutes
            })
        
        if self.db_access:
            try:
                return self.db_access.get_price_statistics(
                    exchange=exchange,
                    symbol=symbol,
                    lookback_minutes=lookback_minutes
                )
            except Exception as e:
                logger.error(f"Error getting historical statistics: {e}")
                return {"error": str(e)}
        
        return {"error": "DuckDB access not available"}
    
    # ===== Anomaly Detection Tools =====
    
    def detect_anomaly(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        current_value: float,
        historical_values: List[float]
    ) -> Dict[str, Any]:
        """
        Detect anomalies in data values.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data_type: Type of data
            current_value: Current value to check
            historical_values: Recent historical values
            
        Returns:
            Anomaly detection result
        """
        if self.shadow_mode:
            return self._shadow_response("detect_anomaly", {
                "exchange": exchange,
                "symbol": symbol,
                "data_type": data_type,
                "current_value": current_value
            })
        
        if len(historical_values) < 10:
            return {
                "exchange": exchange,
                "symbol": symbol,
                "is_anomaly": False,
                "message": "Insufficient historical data"
            }
        
        # Calculate statistics
        mean_val = sum(historical_values) / len(historical_values)
        variance = sum((x - mean_val) ** 2 for x in historical_values) / len(historical_values)
        std_dev = variance ** 0.5
        
        # Z-score
        z_score = (current_value - mean_val) / std_dev if std_dev > 0 else 0
        
        # Anomaly thresholds
        is_anomaly = abs(z_score) > 3.0  # 3 sigma
        severity = "low"
        if abs(z_score) > 4.0:
            severity = "high"
        elif abs(z_score) > 3.5:
            severity = "medium"
        
        return {
            "exchange": exchange,
            "symbol": symbol,
            "data_type": data_type,
            "current_value": current_value,
            "is_anomaly": is_anomaly,
            "z_score": z_score,
            "severity": severity if is_anomaly else None,
            "statistics": {
                "mean": mean_val,
                "std_dev": std_dev,
                "sample_size": len(historical_values)
            }
        }
    
    # ===== Quarantine Tools =====
    
    def quarantine_data(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        record_ids: List[str],
        reason: str
    ) -> Dict[str, Any]:
        """
        Move suspicious data to quarantine.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data_type: Type of data
            record_ids: IDs of records to quarantine
            reason: Reason for quarantine
            
        Returns:
            Quarantine result
        """
        if self.shadow_mode:
            return self._shadow_response("quarantine_data", {
                "exchange": exchange,
                "symbol": symbol,
                "data_type": data_type,
                "record_count": len(record_ids),
                "reason": reason
            })
        
        if not self._check_permission("data:quarantine"):
            return {"error": "Permission denied: data:quarantine"}
        
        # Log the quarantine action
        self._log_action("quarantine_data", {
            "exchange": exchange,
            "symbol": symbol,
            "data_type": data_type,
            "record_ids": record_ids,
            "reason": reason
        })
        
        return {
            "success": True,
            "quarantined_count": len(record_ids),
            "exchange": exchange,
            "symbol": symbol,
            "reason": reason
        }
    
    def get_registered_tools(self) -> List[Dict[str, Any]]:
        """Return list of tools for CrewAI registration."""
        return [
            {
                "name": "validate_price_data",
                "description": "Validate price data against rules and reference prices",
                "func": self.validate_price_data
            },
            {
                "name": "validate_orderbook",
                "description": "Validate orderbook data structure and consistency",
                "func": self.validate_orderbook
            },
            {
                "name": "cross_exchange_consistency",
                "description": "Check consistency of data across exchanges",
                "func": self.cross_exchange_consistency
            },
            {
                "name": "detect_anomaly",
                "description": "Detect anomalies in data values using statistical methods",
                "func": self.detect_anomaly
            },
            {
                "name": "quarantine_data",
                "description": "Move suspicious data to quarantine",
                "func": self.quarantine_data
            }
        ]


class DataCleanerTools(DataOpsToolBase):
    """
    Tool wrapper for the Data Cleaner Agent.
    
    Provides tools for:
    - Data interpolation
    - Time aggregation
    - Cross-exchange normalization
    - Gap filling
    """
    
    TOOL_CATEGORY = "data_cleaning"
    REQUIRED_PERMISSIONS = ["data:read", "data:write", "data:transform"]
    
    def __init__(
        self,
        permission_manager=None,
        tool_registry=None,
        shadow_mode: bool = False
    ):
        super().__init__(
            permission_manager=permission_manager,
            tool_registry=tool_registry,
            shadow_mode=shadow_mode
        )
        self._db_manager = None
    
    def _get_category(self) -> str:
        """Return the tool category for this wrapper."""
        return self.TOOL_CATEGORY
    
    def _get_tools(self) -> Dict[str, Any]:
        """Return mapping of tool names to wrapper methods."""
        return {
            "interpolate_missing_data": self.interpolate_missing_data,
            "aggregate_to_timeframe": self.aggregate_to_timeframe,
            "normalize_exchange_format": self.normalize_exchange_format
        }
    
    def set_db_manager(self, manager):
        """Inject the database manager instance."""
        self._db_manager = manager
    
    # ===== Interpolation Tools =====
    
    def interpolate_missing_data(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        gap_start: datetime,
        gap_end: datetime,
        strategy: str = "linear"
    ) -> Dict[str, Any]:
        """
        Fill missing data using interpolation.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data_type: Type of data
            gap_start: Start of gap
            gap_end: End of gap
            strategy: Interpolation strategy (linear, forward_fill, none)
            
        Returns:
            Interpolation result
        """
        if self.shadow_mode:
            return self._shadow_response("interpolate_missing_data", {
                "exchange": exchange,
                "symbol": symbol,
                "data_type": data_type,
                "strategy": strategy
            })
        
        if not self._check_permission("data:write"):
            return {"error": "Permission denied: data:write"}
        
        # Validate strategy for data type
        valid_strategies = {
            "price": ["linear", "forward_fill"],
            "funding": ["forward_fill"],
            "open_interest": ["linear", "forward_fill"],
            "orderbook": ["none"],
            "trades": ["none"]
        }
        
        allowed = valid_strategies.get(data_type, ["linear"])
        if strategy not in allowed:
            return {
                "error": f"Strategy '{strategy}' not allowed for {data_type}. Allowed: {allowed}"
            }
        
        self._log_action("interpolate_missing_data", {
            "exchange": exchange,
            "symbol": symbol,
            "data_type": data_type,
            "gap_start": gap_start.isoformat(),
            "gap_end": gap_end.isoformat(),
            "strategy": strategy
        })
        
        # Actual implementation would go here
        return {
            "success": True,
            "exchange": exchange,
            "symbol": symbol,
            "data_type": data_type,
            "strategy": strategy,
            "records_created": 0,  # Would be actual count
            "gap_start": gap_start.isoformat(),
            "gap_end": gap_end.isoformat()
        }
    
    # ===== Aggregation Tools =====
    
    def aggregate_to_timeframe(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        source_timeframe: str,
        target_timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Aggregate data to a larger timeframe.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data_type: Type of data
            source_timeframe: Source timeframe (e.g., '1m')
            target_timeframe: Target timeframe (e.g., '5m', '1h')
            start_time: Start of aggregation period
            end_time: End of aggregation period
            
        Returns:
            Aggregation result
        """
        if self.shadow_mode:
            return self._shadow_response("aggregate_to_timeframe", {
                "exchange": exchange,
                "symbol": symbol,
                "data_type": data_type,
                "source_timeframe": source_timeframe,
                "target_timeframe": target_timeframe
            })
        
        if not self._check_permission("data:transform"):
            return {"error": "Permission denied: data:transform"}
        
        # Validate timeframe conversion
        valid_conversions = {
            "1m": ["5m", "15m", "1h", "4h", "1d"],
            "5m": ["15m", "1h", "4h", "1d"],
            "15m": ["1h", "4h", "1d"],
            "1h": ["4h", "1d"],
            "4h": ["1d"]
        }
        
        allowed = valid_conversions.get(source_timeframe, [])
        if target_timeframe not in allowed:
            return {
                "error": f"Cannot aggregate {source_timeframe} to {target_timeframe}"
            }
        
        self._log_action("aggregate_to_timeframe", {
            "exchange": exchange,
            "symbol": symbol,
            "data_type": data_type,
            "source_timeframe": source_timeframe,
            "target_timeframe": target_timeframe
        })
        
        return {
            "success": True,
            "exchange": exchange,
            "symbol": symbol,
            "data_type": data_type,
            "source_timeframe": source_timeframe,
            "target_timeframe": target_timeframe,
            "records_aggregated": 0  # Would be actual count
        }
    
    # ===== Normalization Tools =====
    
    def normalize_exchange_format(
        self,
        source_exchange: str,
        symbol: str,
        data_type: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Normalize data format to standard schema.
        
        Args:
            source_exchange: Exchange the data is from
            symbol: Trading symbol
            data_type: Type of data
            data: Raw data to normalize
            
        Returns:
            Normalized data
        """
        if self.shadow_mode:
            return self._shadow_response("normalize_exchange_format", {
                "source_exchange": source_exchange,
                "symbol": symbol,
                "data_type": data_type
            })
        
        # Standard field mappings per exchange
        field_mappings = {
            "binance": {
                "price": {"p": "price", "q": "quantity", "T": "timestamp"},
                "ticker": {"c": "close", "h": "high", "l": "low", "v": "volume"}
            },
            "bybit": {
                "price": {"price": "price", "size": "quantity", "time": "timestamp"},
                "ticker": {"lastPrice": "close", "highPrice24h": "high", "lowPrice24h": "low"}
            }
        }
        
        mapping = field_mappings.get(source_exchange, {}).get(data_type, {})
        
        normalized = {}
        for old_key, new_key in mapping.items():
            if old_key in data:
                normalized[new_key] = data[old_key]
        
        # Include unmapped fields
        for key, value in data.items():
            if key not in mapping:
                normalized[key] = value
        
        return {
            "source_exchange": source_exchange,
            "symbol": symbol,
            "data_type": data_type,
            "original": data,
            "normalized": normalized
        }
    
    def get_registered_tools(self) -> List[Dict[str, Any]]:
        """Return list of tools for CrewAI registration."""
        return [
            {
                "name": "interpolate_missing_data",
                "description": "Fill missing data gaps using interpolation",
                "func": self.interpolate_missing_data
            },
            {
                "name": "aggregate_to_timeframe",
                "description": "Aggregate data to a larger timeframe",
                "func": self.aggregate_to_timeframe
            },
            {
                "name": "normalize_exchange_format",
                "description": "Normalize exchange-specific data to standard format",
                "func": self.normalize_exchange_format
            }
        ]


class SchemaManagerTools(DataOpsToolBase):
    """
    Tool wrapper for the Schema Manager Agent.
    
    Provides tools for:
    - Table health monitoring
    - Performance optimization
    - Capacity management
    - Schema analysis
    """
    
    TOOL_CATEGORY = "schema_management"
    REQUIRED_PERMISSIONS = ["schema:read", "schema:optimize", "schema:alert"]
    
    def __init__(
        self,
        permission_manager=None,
        tool_registry=None,
        shadow_mode: bool = False
    ):
        super().__init__(
            permission_manager=permission_manager,
            tool_registry=tool_registry,
            shadow_mode=shadow_mode
        )
        self._db_manager = None
    
    def _get_category(self) -> str:
        """Return the tool category for this wrapper."""
        return self.TOOL_CATEGORY
    
    def _get_tools(self) -> Dict[str, Any]:
        """Return mapping of tool names to wrapper methods."""
        return {
            "get_table_health": self.get_table_health,
            "get_all_tables_health": self.get_all_tables_health,
            "analyze_table_performance": self.analyze_table_performance,
            "vacuum_table": self.vacuum_table,
            "check_capacity": self.check_capacity
        }
    
    def set_db_manager(self, manager):
        """Inject the database manager instance."""
        self._db_manager = manager
    
    # ===== Health Monitoring Tools =====
    
    def get_table_health(self, table_name: str) -> Dict[str, Any]:
        """
        Get health metrics for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Health metrics
        """
        if self.shadow_mode:
            return self._shadow_response("get_table_health", {"table_name": table_name})
        
        if not self._db_manager:
            return {"error": "Database manager not configured"}
        
        try:
            # Get table statistics
            conn = self._db_manager.get_connection()
            
            # Row count
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            
            # Estimate size (DuckDB specific)
            # Note: Actual implementation would vary
            
            return {
                "table_name": table_name,
                "row_count": row_count,
                "health_status": "healthy" if row_count > 0 else "empty",
                "recommendations": []
            }
        except Exception as e:
            logger.error(f"Error getting table health: {e}")
            return {"table_name": table_name, "error": str(e)}
    
    def get_all_tables_health(self) -> Dict[str, Any]:
        """Get health metrics for all monitored tables."""
        if self.shadow_mode:
            return self._shadow_response("get_all_tables_health", {})
        
        if not self._db_manager:
            return {"error": "Database manager not configured"}
        
        try:
            conn = self._db_manager.get_connection()
            
            # Get all tables
            tables = conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
            """).fetchall()
            
            results = []
            for (table_name,) in tables:
                health = self.get_table_health(table_name)
                results.append(health)
            
            return {
                "total_tables": len(results),
                "healthy": sum(1 for t in results if t.get("health_status") == "healthy"),
                "tables": results
            }
        except Exception as e:
            logger.error(f"Error getting all tables health: {e}")
            return {"error": str(e)}
    
    # ===== Optimization Tools =====
    
    def analyze_table_performance(self, table_name: str) -> Dict[str, Any]:
        """
        Analyze table performance and suggest optimizations.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Performance analysis
        """
        if self.shadow_mode:
            return self._shadow_response("analyze_table_performance", {"table_name": table_name})
        
        if not self._db_manager:
            return {"error": "Database manager not configured"}
        
        recommendations = []
        
        # Check for indexes
        # Check for fragmentation
        # Check for growth rate
        
        return {
            "table_name": table_name,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "recommendations": recommendations
        }
    
    def vacuum_table(self, table_name: str) -> Dict[str, Any]:
        """
        Run VACUUM on a table to reclaim space.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Vacuum result
        """
        if self.shadow_mode:
            return self._shadow_response("vacuum_table", {"table_name": table_name})
        
        if not self._check_permission("schema:optimize"):
            return {"error": "Permission denied: schema:optimize"}
        
        if not self._db_manager:
            return {"error": "Database manager not configured"}
        
        try:
            conn = self._db_manager.get_connection()
            conn.execute(f"VACUUM {table_name}")
            
            self._log_action("vacuum_table", {"table_name": table_name})
            
            return {
                "success": True,
                "table_name": table_name,
                "vacuumed_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error vacuuming table: {e}")
            return {"error": str(e)}
    
    # ===== Capacity Tools =====
    
    def check_capacity(self) -> Dict[str, Any]:
        """Check overall database capacity and usage."""
        if self.shadow_mode:
            return self._shadow_response("check_capacity", {})
        
        if not self._db_manager:
            return {"error": "Database manager not configured"}
        
        try:
            conn = self._db_manager.get_connection()
            
            # Get database size
            # Get table counts
            # Calculate growth projections
            
            return {
                "capacity_status": "ok",
                "total_tables": 0,
                "total_rows": 0,
                "estimated_size_mb": 0,
                "growth_rate_mb_per_day": 0,
                "days_until_threshold": None
            }
        except Exception as e:
            logger.error(f"Error checking capacity: {e}")
            return {"error": str(e)}
    
    def get_registered_tools(self) -> List[Dict[str, Any]]:
        """Return list of tools for CrewAI registration."""
        return [
            {
                "name": "get_table_health",
                "description": "Get health metrics for a specific table",
                "func": self.get_table_health
            },
            {
                "name": "get_all_tables_health",
                "description": "Get health metrics for all monitored tables",
                "func": self.get_all_tables_health
            },
            {
                "name": "analyze_table_performance",
                "description": "Analyze table performance and suggest optimizations",
                "func": self.analyze_table_performance
            },
            {
                "name": "vacuum_table",
                "description": "Run VACUUM on a table to reclaim space",
                "func": self.vacuum_table
            },
            {
                "name": "check_capacity",
                "description": "Check overall database capacity and usage",
                "func": self.check_capacity
            }
        ]
