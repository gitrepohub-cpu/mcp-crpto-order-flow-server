"""
MCP Client for Sibyl Integration
================================

HTTP client to communicate with MCP Crypto Order Flow Server.
This is the bridge between Sibyl Streamlit UI and MCP tools.

Usage:
    from sibyl_integration.mcp_client import MCPClient, get_mcp_client
    
    # Get singleton instance
    client = get_mcp_client()
    
    # Call any tool
    result = await client.call_tool("get_price_features", symbol="BTCUSDT")
    
    # Use convenience methods
    features = await client.get_all_features("BTCUSDT", "binance")
    signals = await client.get_all_signals("BTCUSDT", "binance")
"""

import httpx
import asyncio
import json
import logging
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
import os

logger = logging.getLogger(__name__)


@dataclass
class MCPResponse:
    """
    Structured response from MCP tool call.
    
    Attributes:
        success: Whether the call succeeded
        data: Parsed response data (dict)
        xml_raw: Raw XML string if response was XML
        error: Error message if failed
        execution_time_ms: Time taken in milliseconds
        tool_name: Name of tool called
        timestamp: When the call was made
    """
    success: bool
    data: Dict[str, Any]
    xml_raw: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    tool_name: str = ""
    timestamp: str = ""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from data dict"""
        return self.data.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to data"""
        return self.data[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key in data"""
        return key in self.data


class MCPClient:
    """
    HTTP Client to call MCP Crypto Order Flow Server tools.
    
    This client handles:
    - HTTP communication with the MCP HTTP API
    - XML/JSON response parsing
    - Response caching for performance
    - Error handling and retries
    - Convenience methods for common operations
    
    The client is designed to be used with Streamlit's caching
    for optimal performance in the Sibyl UI.
    """
    
    # Default configuration
    DEFAULT_BASE_URL = "http://localhost:8000"
    DEFAULT_TIMEOUT = 60.0
    DEFAULT_CACHE_TTL = 5  # seconds
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        cache_ttl: int = DEFAULT_CACHE_TTL
    ):
        """
        Initialize MCP Client.
        
        Args:
            base_url: MCP HTTP API base URL (default: http://localhost:8000)
            timeout: Request timeout in seconds
            cache_ttl: Cache time-to-live in seconds
        """
        self.base_url = base_url or os.environ.get("MCP_API_URL", self.DEFAULT_BASE_URL)
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        
        # HTTP client with timeout
        self._client: Optional[httpx.AsyncClient] = None
        
        # Simple cache: {cache_key: {"response": MCPResponse, "expires": timestamp}}
        self._cache: Dict[str, Dict] = {}
        
        logger.info(f"MCPClient initialized with base_url={self.base_url}")
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-load async HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
        return self._client
    
    async def close(self):
        """Close HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    def _get_cache_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Generate cache key from tool name and params"""
        params_str = json.dumps(params, sort_keys=True, default=str)
        return f"{tool_name}:{params_str}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached entry is still valid"""
        if cache_key not in self._cache:
            return False
        entry = self._cache[cache_key]
        return datetime.utcnow().timestamp() < entry["expires"]
    
    def _get_from_cache(self, cache_key: str) -> Optional[MCPResponse]:
        """Get response from cache if valid"""
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]["response"]
        return None
    
    def _set_cache(self, cache_key: str, response: MCPResponse):
        """Store response in cache"""
        self._cache[cache_key] = {
            "response": response,
            "expires": datetime.utcnow().timestamp() + self.cache_ttl
        }
    
    def clear_cache(self):
        """Clear all cached responses"""
        self._cache.clear()
    
    # =========================================================================
    # CORE API METHODS
    # =========================================================================
    
    async def call_tool(
        self,
        tool_name: str,
        use_cache: bool = True,
        **params
    ) -> MCPResponse:
        """
        Call any MCP tool by name.
        
        Args:
            tool_name: Name of MCP tool (e.g., "get_price_features")
            use_cache: Whether to use cached results
            **params: Tool parameters (symbol, exchange, etc.)
        
        Returns:
            MCPResponse with parsed data
        
        Example:
            result = await client.call_tool(
                "get_price_features",
                symbol="BTCUSDT",
                exchange="binance"
            )
            print(result.data)
        """
        cache_key = self._get_cache_key(tool_name, params)
        
        # Check cache first
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached:
                logger.debug(f"Cache hit for {tool_name}")
                return cached
        
        try:
            # Make HTTP request
            response = await self.client.post(
                f"/tools/{tool_name}",
                json={"params": params}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Parse the tool result
                parsed_data = self._parse_response(result.get("result", {}))
                
                mcp_response = MCPResponse(
                    success=result.get("success", False),
                    data=parsed_data,
                    xml_raw=result.get("result") if isinstance(result.get("result"), str) else None,
                    execution_time_ms=result.get("execution_time_ms", 0),
                    tool_name=tool_name,
                    timestamp=result.get("timestamp", datetime.utcnow().isoformat())
                )
                
                # Cache successful responses
                if mcp_response.success and use_cache:
                    self._set_cache(cache_key, mcp_response)
                
                return mcp_response
            
            elif response.status_code == 404:
                return MCPResponse(
                    success=False,
                    data={},
                    error=f"Tool not found: {tool_name}",
                    tool_name=tool_name,
                    timestamp=datetime.utcnow().isoformat()
                )
            
            else:
                return MCPResponse(
                    success=False,
                    data={},
                    error=f"HTTP {response.status_code}: {response.text}",
                    tool_name=tool_name,
                    timestamp=datetime.utcnow().isoformat()
                )
        
        except httpx.TimeoutException:
            return MCPResponse(
                success=False,
                data={},
                error=f"Request timeout after {self.timeout}s",
                tool_name=tool_name,
                timestamp=datetime.utcnow().isoformat()
            )
        
        except httpx.ConnectError:
            return MCPResponse(
                success=False,
                data={},
                error=f"Cannot connect to MCP server at {self.base_url}",
                tool_name=tool_name,
                timestamp=datetime.utcnow().isoformat()
            )
        
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name=tool_name,
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def call_tools_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        use_cache: bool = True
    ) -> List[MCPResponse]:
        """
        Call multiple tools in parallel.
        
        Args:
            tool_calls: List of {"tool_name": str, "params": dict}
            use_cache: Whether to use cached results
        
        Returns:
            List of MCPResponse in same order as tool_calls
        
        Example:
            results = await client.call_tools_parallel([
                {"tool_name": "get_price_features", "params": {"symbol": "BTCUSDT"}},
                {"tool_name": "get_trade_features", "params": {"symbol": "BTCUSDT"}},
            ])
        """
        tasks = [
            self.call_tool(
                call["tool_name"],
                use_cache=use_cache,
                **call.get("params", {})
            )
            for call in tool_calls
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    def _parse_response(self, result: Any) -> Dict[str, Any]:
        """
        Parse MCP tool response (handles XML and JSON).
        
        The MCP server returns XML strings for most tools.
        This method parses them into dictionaries.
        """
        if isinstance(result, dict):
            return result
        
        if isinstance(result, str):
            result = result.strip()
            
            # Try XML parsing
            if result.startswith('<?xml') or result.startswith('<'):
                try:
                    return self._parse_xml(result)
                except Exception as e:
                    logger.debug(f"XML parse failed: {e}")
            
            # Try JSON parsing
            try:
                return json.loads(result)
            except:
                pass
            
            # Return as raw text
            return {"raw": result}
        
        return {"raw": str(result)}
    
    def _parse_xml(self, xml_string: str) -> Dict[str, Any]:
        """Parse XML response to dict"""
        root = ET.fromstring(xml_string)
        return self._xml_to_dict(root)
    
    def _xml_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Recursively convert XML element to dict"""
        result = {}
        
        # Add element tag as type indicator
        result["_type"] = element.tag
        
        # Add attributes
        result.update(element.attrib)
        
        # Add children
        for child in element:
            child_data = self._xml_to_dict(child)
            
            if child.tag in result:
                # Convert to list if multiple children with same tag
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        # Add text content
        if element.text and element.text.strip():
            text = element.text.strip()
            if not result or (len(result) == 1 and "_type" in result):
                # Leaf node - try numeric conversion
                try:
                    return float(text) if '.' in text else int(text)
                except ValueError:
                    return text
            else:
                result["_text"] = text
        
        return result
    
    # =========================================================================
    # CONVENIENCE ENDPOINTS (call aggregated HTTP endpoints)
    # =========================================================================
    
    async def get_all_features(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance"
    ) -> MCPResponse:
        """
        Get ALL 139+ institutional features for a symbol.
        Uses the /features/{symbol} aggregated endpoint.
        """
        try:
            response = await self.client.get(
                f"/features/{symbol}",
                params={"exchange": exchange}
            )
            
            if response.status_code == 200:
                result = response.json()
                return MCPResponse(
                    success=True,
                    data=result.get("features", {}),
                    tool_name="get_all_features",
                    timestamp=result.get("timestamp", datetime.utcnow().isoformat())
                )
            else:
                return MCPResponse(
                    success=False,
                    data={},
                    error=f"HTTP {response.status_code}",
                    tool_name="get_all_features",
                    timestamp=datetime.utcnow().isoformat()
                )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name="get_all_features",
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def get_all_signals(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance"
    ) -> MCPResponse:
        """
        Get ALL 15 composite signals for a symbol.
        Uses the /signals/{symbol} aggregated endpoint.
        """
        try:
            response = await self.client.get(
                f"/signals/{symbol}",
                params={"exchange": exchange}
            )
            
            if response.status_code == 200:
                result = response.json()
                return MCPResponse(
                    success=True,
                    data=result.get("signals", {}),
                    tool_name="get_all_signals",
                    timestamp=result.get("timestamp", datetime.utcnow().isoformat())
                )
            else:
                return MCPResponse(
                    success=False,
                    data={},
                    error=f"HTTP {response.status_code}",
                    tool_name="get_all_signals",
                    timestamp=datetime.utcnow().isoformat()
                )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name="get_all_signals",
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def get_forecast(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        horizon: int = 24,
        priority: str = "balanced"
    ) -> MCPResponse:
        """
        Get intelligent forecast using auto-routed model.
        Uses the /forecast/{symbol} endpoint.
        """
        try:
            response = await self.client.get(
                f"/forecast/{symbol}",
                params={
                    "exchange": exchange,
                    "horizon": horizon,
                    "priority": priority
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return MCPResponse(
                    success=True,
                    data=result.get("forecast", {}),
                    tool_name="get_forecast",
                    timestamp=result.get("timestamp", datetime.utcnow().isoformat())
                )
            else:
                return MCPResponse(
                    success=False,
                    data={},
                    error=f"HTTP {response.status_code}",
                    tool_name="get_forecast",
                    timestamp=datetime.utcnow().isoformat()
                )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name="get_forecast",
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def get_streaming_status(self) -> MCPResponse:
        """Get streaming system status and health."""
        try:
            response = await self.client.get("/streaming/status")
            
            if response.status_code == 200:
                result = response.json()
                return MCPResponse(
                    success=True,
                    data=result.get("streaming", {}),
                    tool_name="get_streaming_status",
                    timestamp=result.get("timestamp", datetime.utcnow().isoformat())
                )
            else:
                return MCPResponse(
                    success=False,
                    data={},
                    error=f"HTTP {response.status_code}",
                    tool_name="get_streaming_status",
                    timestamp=datetime.utcnow().isoformat()
                )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name="get_streaming_status",
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def get_dashboard_data(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance"
    ) -> MCPResponse:
        """
        Get complete dashboard data in one call.
        Returns features, signals, forecast, and streaming status.
        """
        try:
            response = await self.client.get(
                f"/dashboard/{symbol}",
                params={"exchange": exchange}
            )
            
            if response.status_code == 200:
                result = response.json()
                return MCPResponse(
                    success=True,
                    data=result,
                    tool_name="get_dashboard_data",
                    timestamp=result.get("timestamp", datetime.utcnow().isoformat())
                )
            else:
                return MCPResponse(
                    success=False,
                    data={},
                    error=f"HTTP {response.status_code}",
                    tool_name="get_dashboard_data",
                    timestamp=datetime.utcnow().isoformat()
                )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name="get_dashboard_data",
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def list_tools(
        self,
        category: Optional[str] = None
    ) -> MCPResponse:
        """List all available tools, optionally by category."""
        try:
            params = {"category": category} if category else {}
            response = await self.client.get("/tools", params=params)
            
            if response.status_code == 200:
                return MCPResponse(
                    success=True,
                    data=response.json(),
                    tool_name="list_tools",
                    timestamp=datetime.utcnow().isoformat()
                )
            else:
                return MCPResponse(
                    success=False,
                    data={},
                    error=f"HTTP {response.status_code}",
                    tool_name="list_tools",
                    timestamp=datetime.utcnow().isoformat()
                )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name="list_tools",
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def health_check(self) -> bool:
        """Check if MCP server is healthy"""
        try:
            response = await self.client.get("/health")
            return response.status_code == 200
        except:
            return False
    
    # =========================================================================
    # INSTITUTIONAL FEATURE CONVENIENCE METHODS
    # =========================================================================
    
    async def get_price_features(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        """Get price features (microprice, spread, pressure, etc.)"""
        return await self.call_tool("get_price_features", symbol=symbol, exchange=exchange)
    
    async def get_orderbook_features(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        """Get orderbook features (depth imbalance, walls, liquidity)"""
        return await self.call_tool("get_orderbook_features", symbol=symbol, exchange=exchange)
    
    async def get_trade_features(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        """Get trade features (CVD, whale detection, flow toxicity)"""
        return await self.call_tool("get_trade_features", symbol=symbol, exchange=exchange)
    
    async def get_funding_features(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        """Get funding features (rate, momentum, stress)"""
        return await self.call_tool("get_funding_features", symbol=symbol, exchange=exchange)
    
    async def get_oi_features(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        """Get open interest features (leverage, cascade risk)"""
        return await self.call_tool("get_oi_features", symbol=symbol, exchange=exchange)
    
    async def get_liquidation_features(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        """Get liquidation features"""
        return await self.call_tool("get_liquidation_features", symbol=symbol, exchange=exchange)
    
    async def get_mark_price_features(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        """Get mark price features (basis, premium/discount)"""
        return await self.call_tool("get_mark_price_features", symbol=symbol, exchange=exchange)
    
    # =========================================================================
    # COMPOSITE SIGNAL CONVENIENCE METHODS
    # =========================================================================
    
    async def get_smart_accumulation(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        """Get smart accumulation signal"""
        return await self.call_tool("get_smart_accumulation_signal", symbol=symbol, exchange=exchange)
    
    async def get_squeeze_probability(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        """Get short squeeze probability"""
        return await self.call_tool("get_short_squeeze_probability", symbol=symbol, exchange=exchange)
    
    async def get_stop_hunt_detector(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        """Get stop hunt detection"""
        return await self.call_tool("get_stop_hunt_detector", symbol=symbol, exchange=exchange)
    
    async def get_momentum_quality(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        """Get momentum quality signal"""
        return await self.call_tool("get_momentum_quality_signal", symbol=symbol, exchange=exchange)
    
    async def get_aggregated_intelligence(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        """Get aggregated intelligence (all signals combined)"""
        return await self.call_tool("get_aggregated_intelligence", symbol=symbol, exchange=exchange)
    
    # =========================================================================
    # FORECASTING CONVENIENCE METHODS
    # =========================================================================
    
    async def forecast_statistical(
        self,
        symbol: str,
        exchange: str = "binance",
        model: str = "auto_arima",
        horizon: int = 24,
        data_hours: int = 168
    ) -> MCPResponse:
        """Statistical forecasting (ARIMA, ETS, Theta)"""
        return await self.call_tool(
            "forecast_with_darts_statistical",
            symbol=symbol,
            exchange=exchange,
            model=model,
            horizon=horizon,
            data_hours=data_hours
        )
    
    async def forecast_ml(
        self,
        symbol: str,
        exchange: str = "binance",
        model: str = "LightGBM",
        horizon: int = 24,
        data_hours: int = 168
    ) -> MCPResponse:
        """Machine learning forecasting (LightGBM, XGBoost, etc.)"""
        return await self.call_tool(
            "forecast_with_darts_ml",
            symbol=symbol,
            exchange=exchange,
            model=model,
            horizon=horizon,
            data_hours=data_hours
        )
    
    async def forecast_deep_learning(
        self,
        symbol: str,
        exchange: str = "binance",
        model: str = "NBEATS",
        horizon: int = 24,
        data_hours: int = 168
    ) -> MCPResponse:
        """Deep learning forecasting (N-BEATS, TFT, etc.)"""
        return await self.call_tool(
            "forecast_with_darts_dl",
            symbol=symbol,
            exchange=exchange,
            model=model,
            horizon=horizon,
            data_hours=data_hours
        )
    
    async def forecast_zero_shot(
        self,
        symbol: str,
        exchange: str = "binance",
        horizon: int = 24,
        model_variant: str = "small"
    ) -> MCPResponse:
        """Zero-shot forecasting with Chronos-2"""
        return await self.call_tool(
            "forecast_zero_shot",
            symbol=symbol,
            exchange=exchange,
            horizon=horizon,
            model_variant=model_variant
        )
    
    async def forecast_ensemble(
        self,
        symbol: str,
        exchange: str = "binance",
        horizon: int = 24,
        method: str = "simple"
    ) -> MCPResponse:
        """Ensemble forecasting"""
        tool = "ensemble_forecast_simple" if method == "simple" else "ensemble_forecast_advanced"
        return await self.call_tool(
            tool,
            symbol=symbol,
            exchange=exchange,
            horizon=horizon
        )
    
    # =========================================================================
    # ANALYTICS CONVENIENCE METHODS
    # =========================================================================
    
    async def detect_regime(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        """Detect market regime"""
        return await self.call_tool("detect_market_regime", symbol=symbol, exchange=exchange)
    
    async def detect_anomalies(
        self,
        symbol: str,
        exchange: str = "binance",
        method: str = "zscore",
        hours: int = 24
    ) -> MCPResponse:
        """Detect anomalies in price data"""
        return await self.call_tool(
            "detect_anomalies",
            symbol=symbol,
            exchange=exchange,
            method=method,
            hours=hours
        )
    
    async def check_model_drift(self, model_id: str) -> MCPResponse:
        """Check for model drift"""
        return await self.call_tool("check_model_drift", model_id=model_id)
    
    async def get_model_health(self, model_id: Optional[str] = None) -> MCPResponse:
        """Get model health report"""
        return await self.call_tool("get_model_health_report", model_id=model_id)
    
    async def cross_validate(
        self,
        symbol: str,
        model: str = "LightGBM",
        method: str = "walk_forward",
        n_folds: int = 5
    ) -> MCPResponse:
        """Cross-validate a model"""
        return await self.call_tool(
            "cross_validate_forecast_model",
            symbol=symbol,
            model=model,
            method=method,
            n_folds=n_folds
        )
    
    # =========================================================================
    # VISUALIZATION CONVENIENCE METHODS
    # =========================================================================
    
    async def get_feature_candles(
        self,
        symbol: str,
        exchange: str = "binance",
        timeframe: str = "5m",
        periods: int = 50,
        overlays: str = "microprice,cvd,depth_imbalance_5"
    ) -> MCPResponse:
        """Get candlestick data with feature overlays"""
        return await self.call_tool(
            "get_feature_candles",
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            periods=periods,
            overlays=overlays
        )
    
    async def get_liquidity_heatmap(
        self,
        symbol: str,
        exchange: str = "binance",
        depth_levels: int = 20
    ) -> MCPResponse:
        """Get liquidity heatmap data"""
        return await self.call_tool(
            "get_liquidity_heatmap",
            symbol=symbol,
            exchange=exchange,
            depth_levels=depth_levels
        )
    
    async def get_signal_dashboard_data(
        self,
        symbol: str,
        exchange: str = "binance"
    ) -> MCPResponse:
        """Get signal dashboard data"""
        return await self.call_tool(
            "get_signal_dashboard",
            symbol=symbol,
            exchange=exchange
        )
    
    async def get_regime_timeline(
        self,
        symbol: str,
        exchange: str = "binance"
    ) -> MCPResponse:
        """Get regime timeline data"""
        return await self.call_tool(
            "get_regime_visualization",
            symbol=symbol,
            exchange=exchange
        )
    
    async def get_correlation_matrix_data(
        self,
        symbol: str,
        exchange: str = "binance"
    ) -> MCPResponse:
        """Get feature correlation matrix data"""
        return await self.call_tool(
            "get_correlation_matrix",
            symbol=symbol,
            exchange=exchange
        )
    
    # =========================================================================
    # ADVANCED VISUALIZATION TOOLS
    # =========================================================================
    
    async def get_orderbook_heatmap(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        depth: int = 50
    ) -> MCPResponse:
        """Generate orderbook heatmap visualization data"""
        return await self.call_tool(
            "generate_orderbook_heatmap",
            symbol=symbol,
            exchange=exchange,
            depth=depth
        )
    
    async def get_depth_chart(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        levels: int = 25
    ) -> MCPResponse:
        """Generate depth chart visualization data"""
        return await self.call_tool(
            "generate_depth_chart",
            symbol=symbol,
            exchange=exchange,
            levels=levels
        )
    
    async def get_cvd_chart(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        hours: int = 24
    ) -> MCPResponse:
        """Generate CVD chart visualization data"""
        return await self.call_tool(
            "generate_cvd_chart",
            symbol=symbol,
            exchange=exchange,
            hours=hours
        )
    
    async def get_liquidation_map(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        hours: int = 24
    ) -> MCPResponse:
        """Generate liquidation heatmap visualization data"""
        return await self.call_tool(
            "generate_liquidation_heatmap",
            symbol=symbol,
            exchange=exchange,
            hours=hours
        )
    
    async def get_funding_chart(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        hours: int = 168
    ) -> MCPResponse:
        """Generate funding rate chart visualization data"""
        return await self.call_tool(
            "generate_funding_chart",
            symbol=symbol,
            exchange=exchange,
            hours=hours
        )
    
    # =========================================================================
    # BATCH/MULTI-SYMBOL OPERATIONS
    # =========================================================================
    
    async def get_multi_symbol_features(
        self,
        symbols: List[str] = None,
        exchange: str = "binance",
        feature_types: List[str] = None
    ) -> MCPResponse:
        """
        Fetch features for multiple symbols at once.
        
        Args:
            symbols: List of symbols (default: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
            exchange: Exchange to use
            feature_types: Types of features to fetch (default: all)
            
        Returns:
            MCPResponse with features for all symbols
        """
        if symbols is None:
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        if feature_types is None:
            feature_types = ["price", "orderbook", "trades", "funding", "oi"]
        
        results = {}
        errors = []
        
        for symbol in symbols:
            symbol_data = {}
            for feature_type in feature_types:
                try:
                    method_map = {
                        "price": "get_price_features",
                        "orderbook": "get_orderbook_features",
                        "trades": "get_trade_features",
                        "funding": "get_funding_features",
                        "oi": "get_oi_features"
                    }
                    method_name = method_map.get(feature_type)
                    if method_name and hasattr(self, method_name):
                        method = getattr(self, method_name)
                        response = await method(symbol=symbol, exchange=exchange)
                        if response.success:
                            symbol_data[feature_type] = response.data
                        else:
                            errors.append(f"{symbol}/{feature_type}: {response.error}")
                except Exception as e:
                    errors.append(f"{symbol}/{feature_type}: {str(e)}")
            
            results[symbol] = symbol_data
        
        return MCPResponse(
            success=len(errors) == 0,
            data={
                "symbols": results,
                "total_symbols": len(symbols),
                "feature_types": feature_types,
                "errors": errors if errors else None,
                "timestamp": datetime.utcnow().isoformat()
            },
            error="; ".join(errors) if errors else None,
            tool_name="get_multi_symbol_features",
            timestamp=datetime.utcnow().isoformat()
        )
    
    async def get_multi_exchange_orderbook(
        self,
        symbol: str = "BTCUSDT",
        exchanges: List[str] = None,
        depth: int = 20
    ) -> MCPResponse:
        """
        Compare orderbooks across multiple exchanges.
        
        Args:
            symbol: Symbol to fetch
            exchanges: List of exchanges (default: all major)
            depth: Orderbook depth
            
        Returns:
            MCPResponse with orderbooks from all exchanges
        """
        if exchanges is None:
            exchanges = ["binance", "bybit", "okx", "gate"]
        
        results = {}
        errors = []
        
        for exchange in exchanges:
            try:
                response = await self.get_exchange_orderbook(
                    symbol=symbol,
                    exchange=exchange,
                    depth=depth
                )
                if response.success:
                    results[exchange] = response.data
                else:
                    errors.append(f"{exchange}: {response.error}")
            except Exception as e:
                errors.append(f"{exchange}: {str(e)}")
        
        # Calculate cross-exchange metrics
        best_bid = None
        best_ask = None
        for ex, data in results.items():
            if data and "bids" in data and data["bids"]:
                bid = float(data["bids"][0][0]) if data["bids"][0] else 0
                if best_bid is None or bid > best_bid["price"]:
                    best_bid = {"exchange": ex, "price": bid}
            if data and "asks" in data and data["asks"]:
                ask = float(data["asks"][0][0]) if data["asks"][0] else float('inf')
                if best_ask is None or ask < best_ask["price"]:
                    best_ask = {"exchange": ex, "price": ask}
        
        return MCPResponse(
            success=len(results) > 0,
            data={
                "symbol": symbol,
                "orderbooks": results,
                "exchanges_queried": exchanges,
                "exchanges_success": list(results.keys()),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread_opportunity": (best_ask["price"] - best_bid["price"]) if best_bid and best_ask else None,
                "errors": errors if errors else None,
                "timestamp": datetime.utcnow().isoformat()
            },
            error="; ".join(errors) if errors else None,
            tool_name="get_multi_exchange_orderbook",
            timestamp=datetime.utcnow().isoformat()
        )
    
    async def batch_forecast(
        self,
        symbols: List[str] = None,
        exchange: str = "binance",
        model: str = "LightGBM",
        horizon: int = 24
    ) -> MCPResponse:
        """
        Run forecasts for multiple symbols.
        
        Args:
            symbols: List of symbols to forecast
            exchange: Exchange to use
            model: Forecasting model
            horizon: Forecast horizon in hours
            
        Returns:
            MCPResponse with forecasts for all symbols
        """
        if symbols is None:
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        results = {}
        errors = []
        
        for symbol in symbols:
            try:
                response = await self.forecast_ml(
                    symbol=symbol,
                    exchange=exchange,
                    model=model,
                    horizon=horizon
                )
                if response.success:
                    results[symbol] = response.data
                else:
                    errors.append(f"{symbol}: {response.error}")
            except Exception as e:
                errors.append(f"{symbol}: {str(e)}")
        
        return MCPResponse(
            success=len(results) > 0,
            data={
                "forecasts": results,
                "model": model,
                "horizon": horizon,
                "exchange": exchange,
                "symbols_requested": symbols,
                "symbols_success": list(results.keys()),
                "errors": errors if errors else None,
                "timestamp": datetime.utcnow().isoformat()
            },
            error="; ".join(errors) if errors else None,
            tool_name="batch_forecast",
            timestamp=datetime.utcnow().isoformat()
        )
    
    # =========================================================================
    # SYSTEM ENDPOINTS - Health, Exchanges, Symbols, Tools
    # =========================================================================
    
    async def get_health(self) -> MCPResponse:
        """Get system health check"""
        try:
            response = await self.client.get("/health")
            if response.status_code == 200:
                return MCPResponse(
                    success=True,
                    data=response.json(),
                    tool_name="get_health",
                    timestamp=datetime.utcnow().isoformat()
                )
            return MCPResponse(
                success=False,
                data={},
                error=f"HTTP {response.status_code}",
                tool_name="get_health",
                timestamp=datetime.utcnow().isoformat()
            )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name="get_health",
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def get_exchanges(self) -> MCPResponse:
        """Get list of supported exchanges"""
        try:
            response = await self.client.get("/exchanges")
            if response.status_code == 200:
                return MCPResponse(
                    success=True,
                    data=response.json(),
                    tool_name="get_exchanges",
                    timestamp=datetime.utcnow().isoformat()
                )
            return MCPResponse(
                success=False,
                data={},
                error=f"HTTP {response.status_code}",
                tool_name="get_exchanges",
                timestamp=datetime.utcnow().isoformat()
            )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name="get_exchanges",
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def get_symbols(
        self,
        exchange: str = "binance",
        market_type: str = "futures"
    ) -> MCPResponse:
        """Get list of symbols for an exchange"""
        try:
            response = await self.client.get(
                f"/symbols/{exchange}",
                params={"market_type": market_type}
            )
            if response.status_code == 200:
                return MCPResponse(
                    success=True,
                    data=response.json(),
                    tool_name="get_symbols",
                    timestamp=datetime.utcnow().isoformat()
                )
            return MCPResponse(
                success=False,
                data={},
                error=f"HTTP {response.status_code}",
                tool_name="get_symbols",
                timestamp=datetime.utcnow().isoformat()
            )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name="get_symbols",
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def get_tools_list(
        self,
        category: Optional[str] = None
    ) -> MCPResponse:
        """Get list of available tools"""
        try:
            params = {}
            if category:
                params["category"] = category
            response = await self.client.get("/tools", params=params)
            if response.status_code == 200:
                return MCPResponse(
                    success=True,
                    data=response.json(),
                    tool_name="get_tools_list",
                    timestamp=datetime.utcnow().isoformat()
                )
            return MCPResponse(
                success=False,
                data={},
                error=f"HTTP {response.status_code}",
                tool_name="get_tools_list",
                timestamp=datetime.utcnow().isoformat()
            )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name="get_tools_list",
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def get_tool_schema(self, tool_name: str) -> MCPResponse:
        """Get schema for a specific tool"""
        try:
            response = await self.client.get(f"/tools/{tool_name}/schema")
            if response.status_code == 200:
                return MCPResponse(
                    success=True,
                    data=response.json(),
                    tool_name="get_tool_schema",
                    timestamp=datetime.utcnow().isoformat()
                )
            return MCPResponse(
                success=False,
                data={},
                error=f"HTTP {response.status_code}",
                tool_name="get_tool_schema",
                timestamp=datetime.utcnow().isoformat()
            )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name="get_tool_schema",
                timestamp=datetime.utcnow().isoformat()
            )
    
    # =========================================================================
    # EXCHANGE-SPECIFIC TOOL WRAPPERS
    # =========================================================================
    
    async def get_exchange_orderbook(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        depth: int = 20
    ) -> MCPResponse:
        """Get orderbook from specific exchange"""
        tool_map = {
            "binance": "binance_get_orderbook",
            "binance_spot": "binance_spot_orderbook",
            "bybit": "bybit_futures_orderbook",
            "okx": "okx_orderbook",
            "kraken": "kraken_futures_orderbook",
            "gateio": "gateio_futures_orderbook",
            "hyperliquid": "hyperliquid_orderbook",
            "deribit": "deribit_orderbook"
        }
        tool_name = tool_map.get(exchange.lower(), "binance_get_orderbook")
        return await self.call_tool(tool_name, symbol=symbol, limit=depth)
    
    async def get_exchange_trades(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        limit: int = 100
    ) -> MCPResponse:
        """Get recent trades from specific exchange"""
        tool_map = {
            "binance": "binance_get_trades",
            "binance_spot": "binance_spot_trades",
            "bybit": "bybit_spot_trades",
            "okx": "okx_trades",
            "kraken": "kraken_spot_trades",
            "gateio": "gateio_futures_trades",
            "hyperliquid": "hyperliquid_recent_trades",
            "deribit": "deribit_trades"
        }
        tool_name = tool_map.get(exchange.lower(), "binance_get_trades")
        return await self.call_tool(tool_name, symbol=symbol, limit=limit)
    
    async def get_exchange_funding(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance"
    ) -> MCPResponse:
        """Get funding rate from specific exchange"""
        tool_map = {
            "binance": "binance_get_funding_rate",
            "bybit": "bybit_funding_rate",
            "okx": "okx_funding_rate",
            "kraken": "kraken_funding_rates",
            "gateio": "gateio_funding_rate",
            "hyperliquid": "hyperliquid_funding_rate",
            "deribit": "deribit_funding_rate"
        }
        tool_name = tool_map.get(exchange.lower(), "binance_get_funding_rate")
        return await self.call_tool(tool_name, symbol=symbol)
    
    async def get_exchange_open_interest(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance"
    ) -> MCPResponse:
        """Get open interest from specific exchange"""
        tool_map = {
            "binance": "binance_get_open_interest",
            "bybit": "bybit_open_interest",
            "okx": "okx_open_interest",
            "kraken": "kraken_open_interest",
            "gateio": "gateio_open_interest",
            "hyperliquid": "hyperliquid_open_interest",
            "deribit": "deribit_open_interest"
        }
        tool_name = tool_map.get(exchange.lower(), "binance_get_open_interest")
        return await self.call_tool(tool_name, symbol=symbol)
    
    async def get_exchange_ticker(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance"
    ) -> MCPResponse:
        """Get ticker from specific exchange"""
        tool_map = {
            "binance": "binance_get_ticker",
            "binance_spot": "binance_spot_ticker",
            "bybit": "bybit_futures_ticker",
            "okx": "okx_ticker",
            "kraken": "kraken_futures_ticker",
            "gateio": "gateio_futures_ticker",
            "hyperliquid": "hyperliquid_ticker",
            "deribit": "deribit_ticker"
        }
        tool_name = tool_map.get(exchange.lower(), "binance_get_ticker")
        return await self.call_tool(tool_name, symbol=symbol)
    
    # =========================================================================
    # NEW CONVENIENCE ENDPOINTS - Regime, Analytics, Backtest, Cross-Exchange
    # =========================================================================
    
    async def get_ticker_features(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance"
    ) -> MCPResponse:
        """
        Get ticker-based 24h market features for a symbol.
        Returns 10 features: volume_24h, high_24h, low_24h, price_change, etc.
        """
        try:
            response = await self.client.get(
                f"/ticker-features/{symbol}",
                params={"exchange": exchange}
            )
            
            if response.status_code == 200:
                result = response.json()
                return MCPResponse(
                    success=True,
                    data=result.get("ticker_features", {}),
                    tool_name="get_ticker_features",
                    timestamp=result.get("timestamp", datetime.utcnow().isoformat())
                )
            else:
                return MCPResponse(
                    success=False,
                    data={},
                    error=f"HTTP {response.status_code}",
                    tool_name="get_ticker_features",
                    timestamp=datetime.utcnow().isoformat()
                )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name="get_ticker_features",
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def get_regime_data(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        include_history: bool = False
    ) -> MCPResponse:
        """
        Get market regime analysis for a symbol.
        Returns current regime, transitions, and change points.
        """
        try:
            response = await self.client.get(
                f"/regimes/{symbol}",
                params={"exchange": exchange, "include_history": include_history}
            )
            
            if response.status_code == 200:
                result = response.json()
                return MCPResponse(
                    success=True,
                    data=result.get("regimes", {}),
                    tool_name="get_regime_data",
                    timestamp=result.get("timestamp", datetime.utcnow().isoformat())
                )
            else:
                return MCPResponse(
                    success=False,
                    data={},
                    error=f"HTTP {response.status_code}",
                    tool_name="get_regime_data",
                    timestamp=datetime.utcnow().isoformat()
                )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name="get_regime_data",
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def get_analytics_data(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        analysis_type: str = "comprehensive"
    ) -> MCPResponse:
        """
        Get comprehensive analytics for a symbol.
        analysis_type: comprehensive, alpha, leverage, timeseries
        """
        try:
            response = await self.client.get(
                f"/analytics/{symbol}",
                params={"exchange": exchange, "analysis_type": analysis_type}
            )
            
            if response.status_code == 200:
                result = response.json()
                return MCPResponse(
                    success=True,
                    data=result.get("analytics", {}),
                    tool_name="get_analytics_data",
                    timestamp=result.get("timestamp", datetime.utcnow().isoformat())
                )
            else:
                return MCPResponse(
                    success=False,
                    data={},
                    error=f"HTTP {response.status_code}",
                    tool_name="get_analytics_data",
                    timestamp=datetime.utcnow().isoformat()
                )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name="get_analytics_data",
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def get_backtest_data(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        model_type: str = "auto",
        periods: int = 30
    ) -> MCPResponse:
        """
        Get backtesting results for forecast models.
        Returns model performance metrics and comparison.
        """
        try:
            response = await self.client.get(
                f"/backtest/{symbol}",
                params={
                    "exchange": exchange,
                    "model_type": model_type,
                    "periods": periods
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return MCPResponse(
                    success=True,
                    data=result.get("backtest_results", {}),
                    tool_name="get_backtest_data",
                    timestamp=result.get("timestamp", datetime.utcnow().isoformat())
                )
            else:
                return MCPResponse(
                    success=False,
                    data={},
                    error=f"HTTP {response.status_code}",
                    tool_name="get_backtest_data",
                    timestamp=datetime.utcnow().isoformat()
                )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name="get_backtest_data",
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def get_cross_exchange_data(
        self,
        symbol: str = "BTCUSDT",
        exchanges: str = "binance,bybit,okx"
    ) -> MCPResponse:
        """
        Get cross-exchange comparison data for a symbol.
        Returns prices, funding rates, and open interest across exchanges.
        """
        try:
            response = await self.client.get(
                f"/cross-exchange/{symbol}",
                params={"exchanges": exchanges}
            )
            
            if response.status_code == 200:
                result = response.json()
                return MCPResponse(
                    success=True,
                    data=result.get("cross_exchange", {}),
                    tool_name="get_cross_exchange_data",
                    timestamp=result.get("timestamp", datetime.utcnow().isoformat())
                )
            else:
                return MCPResponse(
                    success=False,
                    data={},
                    error=f"HTTP {response.status_code}",
                    tool_name="get_cross_exchange_data",
                    timestamp=datetime.utcnow().isoformat()
                )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name="get_cross_exchange_data",
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def get_model_comparison(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "binance",
        horizon: int = 24
    ) -> MCPResponse:
        """
        Compare multiple forecasting models on a symbol.
        Returns performance metrics, rankings, and recommendations.
        """
        try:
            response = await self.client.get(
                "/model-comparison",
                params={
                    "symbol": symbol,
                    "exchange": exchange,
                    "horizon": horizon
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return MCPResponse(
                    success=True,
                    data=result.get("model_comparison", {}),
                    tool_name="get_model_comparison",
                    timestamp=result.get("timestamp", datetime.utcnow().isoformat())
                )
            else:
                return MCPResponse(
                    success=False,
                    data={},
                    error=f"HTTP {response.status_code}",
                    tool_name="get_model_comparison",
                    timestamp=datetime.utcnow().isoformat()
                )
        except Exception as e:
            return MCPResponse(
                success=False,
                data={},
                error=str(e),
                tool_name="get_model_comparison",
                timestamp=datetime.utcnow().isoformat()
            )


# =============================================================================
# SINGLETON FACTORY
# =============================================================================

_client_instance: Optional[MCPClient] = None


def get_mcp_client(
    base_url: Optional[str] = None,
    force_new: bool = False
) -> MCPClient:
    """
    Get singleton MCPClient instance.
    
    Use this in Streamlit with @st.cache_resource for optimal performance.
    
    Args:
        base_url: Optional custom base URL
        force_new: Force create new instance
    
    Returns:
        MCPClient singleton instance
    
    Example:
        # In Streamlit
        @st.cache_resource
        def get_client():
            return get_mcp_client()
        
        client = get_client()
    """
    global _client_instance
    
    if force_new or _client_instance is None:
        _client_instance = MCPClient(base_url=base_url)
    
    return _client_instance


# =============================================================================
# SYNC WRAPPER FOR NON-ASYNC CONTEXTS
# =============================================================================

class SyncMCPClient:
    """
    Synchronous wrapper for MCPClient.
    
    Use when you can't use async/await (e.g., some Streamlit callbacks).
    """
    
    def __init__(self, base_url: Optional[str] = None):
        self._async_client = MCPClient(base_url=base_url)
    
    def _run(self, coro):
        """Run coroutine in new event loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    # System endpoints
    def get_health(self) -> MCPResponse:
        return self._run(self._async_client.get_health())
    
    def get_exchanges(self) -> MCPResponse:
        return self._run(self._async_client.get_exchanges())
    
    def get_symbols(self, exchange: str = "binance", market_type: str = "futures") -> MCPResponse:
        return self._run(self._async_client.get_symbols(exchange, market_type))
    
    def get_tools_list(self, category: Optional[str] = None) -> MCPResponse:
        return self._run(self._async_client.get_tools_list(category))
    
    def get_tool_schema(self, tool_name: str) -> MCPResponse:
        return self._run(self._async_client.get_tool_schema(tool_name))
    
    # Generic tool call
    def call_tool(self, tool_name: str, **params) -> MCPResponse:
        return self._run(self._async_client.call_tool(tool_name, **params))
    
    # Convenience endpoints
    def get_all_features(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_all_features(symbol, exchange))
    
    def get_all_signals(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_all_signals(symbol, exchange))
    
    def get_forecast(self, symbol: str, exchange: str = "binance", horizon: int = 24) -> MCPResponse:
        return self._run(self._async_client.get_forecast(symbol, exchange, horizon))
    
    def get_dashboard_data(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_dashboard_data(symbol, exchange))
    
    def get_streaming_status(self) -> MCPResponse:
        return self._run(self._async_client.get_streaming_status())
    
    def get_ticker_features(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_ticker_features(symbol, exchange))
    
    def get_regime_data(self, symbol: str, exchange: str = "binance", include_history: bool = False) -> MCPResponse:
        return self._run(self._async_client.get_regime_data(symbol, exchange, include_history))
    
    def get_analytics_data(self, symbol: str, exchange: str = "binance", analysis_type: str = "comprehensive") -> MCPResponse:
        return self._run(self._async_client.get_analytics_data(symbol, exchange, analysis_type))
    
    def get_backtest_data(self, symbol: str, exchange: str = "binance", model_type: str = "auto", periods: int = 30) -> MCPResponse:
        return self._run(self._async_client.get_backtest_data(symbol, exchange, model_type, periods))
    
    def get_cross_exchange_data(self, symbol: str, exchanges: str = "binance,bybit,okx") -> MCPResponse:
        return self._run(self._async_client.get_cross_exchange_data(symbol, exchanges))
    
    def get_model_comparison(self, symbol: str, exchange: str = "binance", horizon: int = 24) -> MCPResponse:
        return self._run(self._async_client.get_model_comparison(symbol, exchange, horizon))
    
    # Individual features
    def get_price_features(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_price_features(symbol, exchange))
    
    def get_orderbook_features(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_orderbook_features(symbol, exchange))
    
    def get_trade_features(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_trade_features(symbol, exchange))
    
    def get_funding_features(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_funding_features(symbol, exchange))
    
    def get_oi_features(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_oi_features(symbol, exchange))
    
    def get_liquidation_features(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_liquidation_features(symbol, exchange))
    
    def get_mark_price_features(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_mark_price_features(symbol, exchange))
    
    # Feature query tools
    def query_historical_features(self, symbol: str = "BTCUSDT", exchange: str = "binance", features: str = None, hours: int = 24) -> MCPResponse:
        return self._run(self._async_client.call_tool("query_historical_features", symbol=symbol, exchange=exchange, features=features, hours=hours))
    
    def get_feature_statistics(self, symbol: str = "BTCUSDT", exchange: str = "binance", hours: int = 24) -> MCPResponse:
        return self._run(self._async_client.call_tool("get_feature_statistics", symbol=symbol, exchange=exchange, hours=hours))
    
    def get_feature_correlation_analysis(self, symbol: str = "BTCUSDT", exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.call_tool("get_feature_correlation_analysis", symbol=symbol, exchange=exchange))
    
    def export_features_csv(self, symbol: str = "BTCUSDT", exchange: str = "binance", hours: int = 24) -> MCPResponse:
        return self._run(self._async_client.call_tool("export_features_csv", symbol=symbol, exchange=exchange, hours=hours))
    
    # Exchange-specific tools
    def get_exchange_orderbook(self, symbol: str = "BTCUSDT", exchange: str = "binance", depth: int = 20) -> MCPResponse:
        return self._run(self._async_client.get_exchange_orderbook(symbol, exchange, depth))
    
    def get_exchange_trades(self, symbol: str = "BTCUSDT", exchange: str = "binance", limit: int = 100) -> MCPResponse:
        return self._run(self._async_client.get_exchange_trades(symbol, exchange, limit))
    
    def get_exchange_funding(self, symbol: str = "BTCUSDT", exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_exchange_funding(symbol, exchange))
    
    def get_exchange_open_interest(self, symbol: str = "BTCUSDT", exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_exchange_open_interest(symbol, exchange))
    
    def get_exchange_ticker(self, symbol: str = "BTCUSDT", exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_exchange_ticker(symbol, exchange))
    
    # Forecasting
    def forecast_statistical(self, symbol: str = "BTCUSDT", exchange: str = "binance", model: str = "auto_arima", horizon: int = 24, data_hours: int = 168) -> MCPResponse:
        return self._run(self._async_client.forecast_statistical(symbol, exchange, model, horizon, data_hours))
    
    def forecast_ml(self, symbol: str = "BTCUSDT", exchange: str = "binance", model: str = "LightGBM", horizon: int = 24, data_hours: int = 168) -> MCPResponse:
        return self._run(self._async_client.forecast_ml(symbol, exchange, model, horizon, data_hours))
    
    def forecast_deep_learning(self, symbol: str = "BTCUSDT", exchange: str = "binance", model: str = "NBEATS", horizon: int = 24, data_hours: int = 168) -> MCPResponse:
        return self._run(self._async_client.forecast_deep_learning(symbol, exchange, model, horizon, data_hours))
    
    def forecast_zero_shot(self, symbol: str = "BTCUSDT", exchange: str = "binance", horizon: int = 24, model_variant: str = "small") -> MCPResponse:
        return self._run(self._async_client.forecast_zero_shot(symbol, exchange, horizon, model_variant))
    
    def forecast_ensemble(self, symbol: str = "BTCUSDT", exchange: str = "binance", horizon: int = 24, method: str = "simple") -> MCPResponse:
        return self._run(self._async_client.forecast_ensemble(symbol, exchange, horizon, method))
    
    # Analytics
    def detect_regime(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.detect_regime(symbol, exchange))
    
    def detect_anomalies(self, symbol: str, exchange: str = "binance", method: str = "zscore", hours: int = 24) -> MCPResponse:
        return self._run(self._async_client.detect_anomalies(symbol, exchange, method, hours))
    
    def check_model_drift(self, model_id: str) -> MCPResponse:
        return self._run(self._async_client.check_model_drift(model_id))
    
    def get_model_health(self, model_id: Optional[str] = None) -> MCPResponse:
        return self._run(self._async_client.get_model_health(model_id))
    
    def cross_validate(self, symbol: str, model: str = "LightGBM", method: str = "walk_forward", n_folds: int = 5) -> MCPResponse:
        return self._run(self._async_client.cross_validate(symbol, model, method, n_folds))
    
    # Composite signals
    def get_smart_accumulation(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_smart_accumulation(symbol, exchange))
    
    def get_squeeze_probability(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_squeeze_probability(symbol, exchange))
    
    def get_stop_hunt_detector(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_stop_hunt_detector(symbol, exchange))
    
    def get_momentum_quality(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_momentum_quality(symbol, exchange))
    
    def get_aggregated_intelligence(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_aggregated_intelligence(symbol, exchange))
    
    # Visualization
    def get_feature_candles(self, symbol: str, exchange: str = "binance", timeframe: str = "5m", periods: int = 50, overlays: str = "microprice,cvd,depth_imbalance_5") -> MCPResponse:
        return self._run(self._async_client.get_feature_candles(symbol, exchange, timeframe, periods, overlays))
    
    def get_liquidity_heatmap(self, symbol: str, exchange: str = "binance", depth_levels: int = 20) -> MCPResponse:
        return self._run(self._async_client.get_liquidity_heatmap(symbol, exchange, depth_levels))
    
    def get_signal_dashboard_data(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_signal_dashboard_data(symbol, exchange))
    
    def get_regime_timeline(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_regime_timeline(symbol, exchange))
    
    def get_correlation_matrix_data(self, symbol: str, exchange: str = "binance") -> MCPResponse:
        return self._run(self._async_client.get_correlation_matrix_data(symbol, exchange))
    
    # Advanced visualization tools
    def get_orderbook_heatmap(self, symbol: str = "BTCUSDT", exchange: str = "binance", depth: int = 50) -> MCPResponse:
        return self._run(self._async_client.get_orderbook_heatmap(symbol, exchange, depth))
    
    def get_depth_chart(self, symbol: str = "BTCUSDT", exchange: str = "binance", levels: int = 25) -> MCPResponse:
        return self._run(self._async_client.get_depth_chart(symbol, exchange, levels))
    
    def get_cvd_chart(self, symbol: str = "BTCUSDT", exchange: str = "binance", hours: int = 24) -> MCPResponse:
        return self._run(self._async_client.get_cvd_chart(symbol, exchange, hours))
    
    def get_liquidation_map(self, symbol: str = "BTCUSDT", exchange: str = "binance", hours: int = 24) -> MCPResponse:
        return self._run(self._async_client.get_liquidation_map(symbol, exchange, hours))
    
    def get_funding_chart(self, symbol: str = "BTCUSDT", exchange: str = "binance", hours: int = 168) -> MCPResponse:
        return self._run(self._async_client.get_funding_chart(symbol, exchange, hours))
    
    # Batch/multi-symbol operations
    def get_multi_symbol_features(self, symbols: List[str] = None, exchange: str = "binance", feature_types: List[str] = None) -> MCPResponse:
        return self._run(self._async_client.get_multi_symbol_features(symbols, exchange, feature_types))
    
    def get_multi_exchange_orderbook(self, symbol: str = "BTCUSDT", exchanges: List[str] = None, depth: int = 20) -> MCPResponse:
        return self._run(self._async_client.get_multi_exchange_orderbook(symbol, exchanges, depth))
    
    def batch_forecast(self, symbols: List[str] = None, exchange: str = "binance", model: str = "LightGBM", horizon: int = 24) -> MCPResponse:
        return self._run(self._async_client.batch_forecast(symbols, exchange, model, horizon))
    
    def health_check(self) -> bool:
        return self._run(self._async_client.health_check())


def get_sync_client(base_url: Optional[str] = None) -> SyncMCPClient:
    """Get synchronous MCP client for non-async contexts"""
    return SyncMCPClient(base_url=base_url)
