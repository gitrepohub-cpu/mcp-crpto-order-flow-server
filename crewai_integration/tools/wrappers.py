"""
MCP Tool Wrappers for CrewAI Agents
===================================

Concrete wrapper implementations that expose MCP tools to CrewAI agents.
Each wrapper class handles a category of tools with appropriate:
- Validation rules
- Output formatting
- Rate limiting
- Permission requirements
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from .base import ToolWrapper, tool_wrapper
from ..core.permissions import ToolCategory, AccessLevel

logger = logging.getLogger(__name__)


class ExchangeDataTools(ToolWrapper):
    """
    Wrapper for exchange data tools (60+ tools).
    
    Provides read-only access to real-time and historical exchange data:
    - Tickers and prices
    - Orderbooks
    - Trades
    - Klines/OHLCV
    - Funding rates
    - Open interest
    - Long/short ratios
    """
    
    def _get_category(self) -> str:
        return "exchange_data"
    
    def _get_tools(self) -> Dict[str, Callable]:
        return {
            # Binance Futures
            "binance_get_ticker": self.get_binance_ticker,
            "binance_get_orderbook": self.get_binance_orderbook,
            "binance_get_funding_rate": self.get_binance_funding,
            "binance_get_open_interest": self.get_binance_oi,
            "binance_market_snapshot": self.get_binance_snapshot,
            
            # Bybit
            "bybit_futures_ticker": self.get_bybit_ticker,
            "bybit_futures_orderbook": self.get_bybit_orderbook,
            "bybit_funding_rate": self.get_bybit_funding,
            "bybit_open_interest": self.get_bybit_oi,
            "bybit_market_snapshot": self.get_bybit_snapshot,
            
            # OKX
            "okx_ticker": self.get_okx_ticker,
            "okx_orderbook": self.get_okx_orderbook,
            "okx_funding_rate": self.get_okx_funding,
            "okx_open_interest": self.get_okx_oi,
            "okx_market_snapshot": self.get_okx_snapshot,
            
            # Hyperliquid
            "hyperliquid_ticker_tool": self.get_hyperliquid_ticker,
            "hyperliquid_orderbook_tool": self.get_hyperliquid_orderbook,
            "hyperliquid_funding_rate_tool": self.get_hyperliquid_funding,
            "hyperliquid_open_interest_tool": self.get_hyperliquid_oi,
            "hyperliquid_market_snapshot_tool": self.get_hyperliquid_snapshot,
            
            # Gate.io
            "gateio_futures_ticker_tool": self.get_gateio_ticker,
            "gateio_futures_orderbook_tool": self.get_gateio_orderbook,
            "gateio_funding_rate_tool": self.get_gateio_funding,
            "gateio_open_interest_tool": self.get_gateio_oi,
            "gateio_market_snapshot_tool": self.get_gateio_snapshot,
            
            # Cross-exchange
            "get_exchange_prices": self.get_all_prices,
            "analyze_crypto_arbitrage": self.analyze_arbitrage,
        }
    
    async def _validate_parameters(
        self,
        tool_name: str,
        params: Dict[str, Any]
    ):
        """Validate exchange data tool parameters."""
        from .base import ValidationResult
        
        errors = []
        warnings = []
        sanitized = params.copy()
        
        # Symbol validation
        if "symbol" in params:
            symbol = params["symbol"].upper()
            if not symbol.endswith("USDT") and not symbol.endswith("USD"):
                warnings.append(f"Symbol '{symbol}' may not be supported on all exchanges")
            sanitized["symbol"] = symbol
        
        # Limit validation
        if "limit" in params:
            limit = int(params["limit"])
            if limit > 1000:
                sanitized["limit"] = 1000
                warnings.append("Limit capped at 1000")
            elif limit < 1:
                sanitized["limit"] = 10
                warnings.append("Limit must be >= 1, defaulting to 10")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_params=sanitized
        )
    
    async def _format_output(self, tool_name: str, result: Any) -> Any:
        """Format exchange data for agent consumption."""
        if isinstance(result, dict):
            # Add metadata
            result["_meta"] = {
                "tool": tool_name,
                "timestamp": datetime.utcnow().isoformat(),
                "wrapper": "ExchangeDataTools"
            }
        return result
    
    # === Binance Tools ===
    
    async def get_binance_ticker(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """Get Binance futures ticker data."""
        from src.tools import binance_get_ticker
        return await self._safe_invoke(binance_get_ticker, symbol=symbol)
    
    async def get_binance_orderbook(self, symbol: str = "BTCUSDT", limit: int = 20) -> Dict[str, Any]:
        """Get Binance futures orderbook."""
        from src.tools import binance_get_orderbook
        return await self._safe_invoke(binance_get_orderbook, symbol=symbol, limit=limit)
    
    async def get_binance_funding(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """Get Binance funding rate."""
        from src.tools import binance_get_funding_rate
        return await self._safe_invoke(binance_get_funding_rate, symbol=symbol)
    
    async def get_binance_oi(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """Get Binance open interest."""
        from src.tools import binance_get_open_interest
        return await self._safe_invoke(binance_get_open_interest, symbol=symbol)
    
    async def get_binance_snapshot(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """Get Binance market snapshot."""
        from src.tools import binance_market_snapshot
        return await self._safe_invoke(binance_market_snapshot, symbol=symbol)
    
    # === Bybit Tools ===
    
    async def get_bybit_ticker(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """Get Bybit futures ticker data."""
        from src.tools import bybit_futures_ticker
        return await self._safe_invoke(bybit_futures_ticker, symbol=symbol)
    
    async def get_bybit_orderbook(self, symbol: str = "BTCUSDT", limit: int = 25) -> Dict[str, Any]:
        """Get Bybit futures orderbook."""
        from src.tools import bybit_futures_orderbook
        return await self._safe_invoke(bybit_futures_orderbook, symbol=symbol, limit=limit)
    
    async def get_bybit_funding(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """Get Bybit funding rate."""
        from src.tools import bybit_funding_rate
        return await self._safe_invoke(bybit_funding_rate, symbol=symbol)
    
    async def get_bybit_oi(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """Get Bybit open interest."""
        from src.tools import bybit_open_interest
        return await self._safe_invoke(bybit_open_interest, symbol=symbol)
    
    async def get_bybit_snapshot(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """Get Bybit market snapshot."""
        from src.tools import bybit_market_snapshot
        return await self._safe_invoke(bybit_market_snapshot, symbol=symbol)
    
    # === OKX Tools ===
    
    async def get_okx_ticker(self, symbol: str = "BTC-USDT-SWAP") -> Dict[str, Any]:
        """Get OKX ticker data."""
        from src.tools import okx_ticker
        return await self._safe_invoke(okx_ticker, inst_id=symbol)
    
    async def get_okx_orderbook(self, symbol: str = "BTC-USDT-SWAP", depth: int = 20) -> Dict[str, Any]:
        """Get OKX orderbook."""
        from src.tools import okx_orderbook
        return await self._safe_invoke(okx_orderbook, inst_id=symbol, depth=depth)
    
    async def get_okx_funding(self, symbol: str = "BTC-USDT-SWAP") -> Dict[str, Any]:
        """Get OKX funding rate."""
        from src.tools import okx_funding_rate
        return await self._safe_invoke(okx_funding_rate, inst_id=symbol)
    
    async def get_okx_oi(self, symbol: str = "BTC-USDT-SWAP") -> Dict[str, Any]:
        """Get OKX open interest."""
        from src.tools import okx_open_interest
        return await self._safe_invoke(okx_open_interest, inst_id=symbol)
    
    async def get_okx_snapshot(self, symbol: str = "BTC-USDT-SWAP") -> Dict[str, Any]:
        """Get OKX market snapshot."""
        from src.tools import okx_market_snapshot
        return await self._safe_invoke(okx_market_snapshot, symbol=symbol)
    
    # === Hyperliquid Tools ===
    
    async def get_hyperliquid_ticker(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Get Hyperliquid ticker data."""
        from src.tools import hyperliquid_ticker_tool
        return await self._safe_invoke(hyperliquid_ticker_tool, symbol=symbol)
    
    async def get_hyperliquid_orderbook(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Get Hyperliquid orderbook."""
        from src.tools import hyperliquid_orderbook_tool
        return await self._safe_invoke(hyperliquid_orderbook_tool, symbol=symbol)
    
    async def get_hyperliquid_funding(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Get Hyperliquid funding rate."""
        from src.tools import hyperliquid_funding_rate_tool
        return await self._safe_invoke(hyperliquid_funding_rate_tool, symbol=symbol)
    
    async def get_hyperliquid_oi(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Get Hyperliquid open interest."""
        from src.tools import hyperliquid_open_interest_tool
        return await self._safe_invoke(hyperliquid_open_interest_tool, symbol=symbol)
    
    async def get_hyperliquid_snapshot(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Get Hyperliquid market snapshot."""
        from src.tools import hyperliquid_market_snapshot_tool
        return await self._safe_invoke(hyperliquid_market_snapshot_tool, symbol=symbol)
    
    # === Gate.io Tools ===
    
    async def get_gateio_ticker(self, contract: str = "BTC_USDT") -> Dict[str, Any]:
        """Get Gate.io ticker data."""
        from src.tools import gateio_futures_ticker_tool
        return await self._safe_invoke(gateio_futures_ticker_tool, contract=contract)
    
    async def get_gateio_orderbook(self, contract: str = "BTC_USDT", limit: int = 20) -> Dict[str, Any]:
        """Get Gate.io orderbook."""
        from src.tools import gateio_futures_orderbook_tool
        return await self._safe_invoke(gateio_futures_orderbook_tool, contract=contract, limit=limit)
    
    async def get_gateio_funding(self, contract: str = "BTC_USDT") -> Dict[str, Any]:
        """Get Gate.io funding rate."""
        from src.tools import gateio_funding_rate_tool
        return await self._safe_invoke(gateio_funding_rate_tool, contract=contract)
    
    async def get_gateio_oi(self, contract: str = "BTC_USDT") -> Dict[str, Any]:
        """Get Gate.io open interest."""
        from src.tools import gateio_open_interest_tool
        return await self._safe_invoke(gateio_open_interest_tool, contract=contract)
    
    async def get_gateio_snapshot(self, contract: str = "BTC_USDT") -> Dict[str, Any]:
        """Get Gate.io market snapshot."""
        from src.tools import gateio_market_snapshot_tool
        return await self._safe_invoke(gateio_market_snapshot_tool, symbol=contract)
    
    # === Cross-Exchange Tools ===
    
    async def get_all_prices(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Get prices across all exchanges."""
        from src.tools import get_exchange_prices
        return await self._safe_invoke(get_exchange_prices, symbols=symbols)
    
    async def analyze_arbitrage(
        self,
        min_spread_bps: float = 10,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """Analyze arbitrage opportunities."""
        from src.tools import analyze_crypto_arbitrage
        return await self._safe_invoke(
            analyze_crypto_arbitrage,
            min_spread_bps=min_spread_bps,
            max_results=max_results
        )
    
    async def _safe_invoke(self, func: Callable, **kwargs) -> Dict[str, Any]:
        """Safely invoke an MCP tool with error handling."""
        try:
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return await func(**kwargs)
            else:
                return func(**kwargs)
        except Exception as e:
            logger.error(f"Tool invocation error: {e}")
            return {"error": str(e), "success": False}


class ForecastingTools(ToolWrapper):
    """
    Wrapper for forecasting tools (38+ tools).
    
    Provides access to time series forecasting capabilities:
    - Statistical models (ARIMA, ETS, Theta)
    - Machine learning models (XGBoost, LightGBM)
    - Deep learning models (NBEATS, TFT)
    - Ensemble methods
    - Model registry and comparison
    """
    
    def _get_category(self) -> str:
        return "forecasting"
    
    def _get_tools(self) -> Dict[str, Callable]:
        return {
            "forecast_with_darts_statistical": self.forecast_statistical,
            "forecast_with_darts_ml": self.forecast_ml,
            "forecast_quick": self.forecast_quick,
            "forecast_zero_shot": self.forecast_foundation,
            "route_forecast_request": self.route_forecast,
            "ensemble_forecast_simple": self.ensemble_simple,
            "ensemble_auto_select": self.ensemble_auto,
            "compare_all_models": self.compare_models,
            "backtest_model": self.backtest,
            "registry_recommend_model": self.recommend_model,
        }
    
    async def forecast_statistical(
        self,
        symbol: str,
        model: str = "auto_arima",
        horizon: int = 24,
        **kwargs
    ) -> Dict[str, Any]:
        """Run statistical forecast."""
        from src.tools import forecast_with_darts_statistical
        return await self._safe_invoke(
            forecast_with_darts_statistical,
            symbol=symbol,
            model=model,
            horizon=horizon,
            **kwargs
        )
    
    async def forecast_ml(
        self,
        symbol: str,
        model: str = "xgboost",
        horizon: int = 24,
        **kwargs
    ) -> Dict[str, Any]:
        """Run ML forecast."""
        from src.tools import forecast_with_darts_ml
        return await self._safe_invoke(
            forecast_with_darts_ml,
            symbol=symbol,
            model=model,
            horizon=horizon,
            **kwargs
        )
    
    async def forecast_quick(
        self,
        symbol: str,
        horizon: int = 12,
        **kwargs
    ) -> Dict[str, Any]:
        """Run quick forecast."""
        from src.tools import forecast_quick
        return await self._safe_invoke(
            forecast_quick,
            symbol=symbol,
            horizon=horizon,
            **kwargs
        )
    
    async def forecast_foundation(
        self,
        symbol: str,
        horizon: int = 24,
        **kwargs
    ) -> Dict[str, Any]:
        """Run Chronos-2 foundation model forecast."""
        from src.tools import forecast_zero_shot
        return await self._safe_invoke(
            forecast_zero_shot,
            symbol=symbol,
            horizon=horizon,
            **kwargs
        )
    
    async def route_forecast(
        self,
        symbol: str,
        task_type: str = "short_term",
        **kwargs
    ) -> Dict[str, Any]:
        """Intelligently route forecast request."""
        from src.tools import route_forecast_request
        return await self._safe_invoke(
            route_forecast_request,
            symbol=symbol,
            task_type=task_type,
            **kwargs
        )
    
    async def ensemble_simple(
        self,
        symbol: str,
        models: List[str] = None,
        horizon: int = 24,
        **kwargs
    ) -> Dict[str, Any]:
        """Run simple ensemble forecast."""
        from src.tools import ensemble_forecast_simple
        return await self._safe_invoke(
            ensemble_forecast_simple,
            symbol=symbol,
            models=models,
            horizon=horizon,
            **kwargs
        )
    
    async def ensemble_auto(
        self,
        symbol: str,
        horizon: int = 24,
        **kwargs
    ) -> Dict[str, Any]:
        """Auto-select best ensemble."""
        from src.tools import ensemble_auto_select
        return await self._safe_invoke(
            ensemble_auto_select,
            symbol=symbol,
            horizon=horizon,
            **kwargs
        )
    
    async def compare_models(
        self,
        symbol: str,
        horizon: int = 24,
        **kwargs
    ) -> Dict[str, Any]:
        """Compare all available models."""
        from src.tools import compare_all_models
        return await self._safe_invoke(
            compare_all_models,
            symbol=symbol,
            horizon=horizon,
            **kwargs
        )
    
    async def backtest(
        self,
        symbol: str,
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Backtest a model."""
        from src.tools import backtest_model
        return await self._safe_invoke(
            backtest_model,
            symbol=symbol,
            model=model,
            **kwargs
        )
    
    async def recommend_model(
        self,
        symbol: str,
        task_type: str = "general",
        **kwargs
    ) -> Dict[str, Any]:
        """Get model recommendation."""
        from src.tools import registry_recommend_model
        return await self._safe_invoke(
            registry_recommend_model,
            symbol=symbol,
            task_type=task_type,
            **kwargs
        )
    
    async def _safe_invoke(self, func: Callable, **kwargs) -> Dict[str, Any]:
        """Safely invoke an MCP tool with error handling."""
        try:
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return await func(**kwargs)
            else:
                return func(**kwargs)
        except Exception as e:
            logger.error(f"Forecasting tool error: {e}")
            return {"error": str(e), "success": False}


class AnalyticsTools(ToolWrapper):
    """
    Wrapper for analytics tools.
    
    Provides access to:
    - Alpha signals
    - Leverage analytics
    - Regime detection
    - Anomaly detection
    """
    
    def _get_category(self) -> str:
        return "analytics"
    
    def _get_tools(self) -> Dict[str, Callable]:
        return {
            "compute_alpha_signals": self.compute_alpha,
            "get_institutional_pressure": self.get_inst_pressure,
            "compute_squeeze_probability": self.compute_squeeze,
            "analyze_leverage_positioning": self.analyze_leverage,
            "detect_market_regime": self.detect_regime,
            "detect_anomalies": self.detect_anomalies,
        }
    
    async def compute_alpha(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Compute alpha signals."""
        from src.tools import compute_alpha_signals
        return await self._safe_invoke(compute_alpha_signals, symbol=symbol, **kwargs)
    
    async def get_inst_pressure(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get institutional pressure."""
        from src.tools import get_institutional_pressure
        return await self._safe_invoke(get_institutional_pressure, symbol=symbol, **kwargs)
    
    async def compute_squeeze(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Compute squeeze probability."""
        from src.tools import compute_squeeze_probability
        return await self._safe_invoke(compute_squeeze_probability, symbol=symbol, **kwargs)
    
    async def analyze_leverage(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Analyze leverage positioning."""
        from src.tools import analyze_leverage_positioning
        return await self._safe_invoke(analyze_leverage_positioning, symbol=symbol, **kwargs)
    
    async def detect_regime(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Detect market regime."""
        from src.tools import detect_market_regime
        return await self._safe_invoke(detect_market_regime, symbol=symbol, **kwargs)
    
    async def detect_anomalies(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Detect anomalies."""
        from src.tools import detect_anomalies
        return await self._safe_invoke(detect_anomalies, symbol=symbol, **kwargs)
    
    async def _safe_invoke(self, func: Callable, **kwargs) -> Dict[str, Any]:
        """Safely invoke an MCP tool with error handling."""
        try:
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return await func(**kwargs)
            else:
                return func(**kwargs)
        except Exception as e:
            logger.error(f"Analytics tool error: {e}")
            return {"error": str(e), "success": False}


class StreamingTools(ToolWrapper):
    """
    Wrapper for streaming control tools.
    
    NOTE: These are WRITE operations and require special permissions.
    Only Operations crew can use these tools.
    """
    
    def _get_category(self) -> str:
        return "streaming_control"
    
    def _get_tools(self) -> Dict[str, Callable]:
        return {
            "start_streaming": self.start_streaming,
            "stop_streaming": self.stop_streaming,
            "get_streaming_status": self.get_status,
            "get_streaming_health": self.get_health,
            "configure_streaming": self.configure,
        }
    
    async def start_streaming(
        self,
        symbols: List[str] = None,
        exchanges: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Start streaming system."""
        from src.tools import start_streaming as _start
        return await self._safe_invoke(_start, symbols=symbols, exchanges=exchanges, **kwargs)
    
    async def stop_streaming(self) -> Dict[str, Any]:
        """Stop streaming system."""
        from src.tools import stop_streaming as _stop
        return await self._safe_invoke(_stop)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get streaming status."""
        from src.tools import get_streaming_status
        return await self._safe_invoke(get_streaming_status)
    
    async def get_health(self) -> Dict[str, Any]:
        """Get streaming health."""
        from src.tools import get_streaming_health
        return await self._safe_invoke(get_streaming_health)
    
    async def configure(self, **config) -> Dict[str, Any]:
        """Configure streaming."""
        from src.tools import configure_streaming
        return await self._safe_invoke(configure_streaming, **config)
    
    async def _safe_invoke(self, func: Callable, **kwargs) -> Dict[str, Any]:
        """Safely invoke an MCP tool with error handling."""
        try:
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return await func(**kwargs)
            else:
                return func(**kwargs)
        except Exception as e:
            logger.error(f"Streaming tool error: {e}")
            return {"error": str(e), "success": False}


class FeatureTools(ToolWrapper):
    """
    Wrapper for institutional feature tools (35 tools).
    
    Provides access to computed features:
    - Price features
    - Orderbook features
    - Trade features
    - Funding features
    - OI features
    - Composite intelligence
    """
    
    def _get_category(self) -> str:
        return "feature_calculator"
    
    def _get_tools(self) -> Dict[str, Callable]:
        return {
            "get_price_features": self.get_price_features,
            "get_orderbook_features": self.get_orderbook_features,
            "get_trade_features": self.get_trade_features,
            "get_funding_features": self.get_funding_features,
            "get_oi_features": self.get_oi_features,
            "get_smart_money_flow": self.get_smart_money,
            "get_short_squeeze_probability": self.get_squeeze,
            "get_aggregated_intelligence": self.get_intelligence,
        }
    
    async def get_price_features(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get price features."""
        from src.tools import get_price_features
        return await self._safe_invoke(get_price_features, symbol=symbol, **kwargs)
    
    async def get_orderbook_features(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get orderbook features."""
        from src.tools import get_orderbook_features
        return await self._safe_invoke(get_orderbook_features, symbol=symbol, **kwargs)
    
    async def get_trade_features(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get trade features."""
        from src.tools import get_trade_features
        return await self._safe_invoke(get_trade_features, symbol=symbol, **kwargs)
    
    async def get_funding_features(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get funding features."""
        from src.tools import get_funding_features
        return await self._safe_invoke(get_funding_features, symbol=symbol, **kwargs)
    
    async def get_oi_features(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get OI features."""
        from src.tools import get_oi_features
        return await self._safe_invoke(get_oi_features, symbol=symbol, **kwargs)
    
    async def get_smart_money(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get smart money flow."""
        from src.tools import get_smart_money_flow
        return await self._safe_invoke(get_smart_money_flow, symbol=symbol, **kwargs)
    
    async def get_squeeze(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get squeeze probability."""
        from src.tools import get_short_squeeze_probability
        return await self._safe_invoke(get_short_squeeze_probability, symbol=symbol, **kwargs)
    
    async def get_intelligence(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get aggregated intelligence."""
        from src.tools import get_aggregated_intelligence
        return await self._safe_invoke(get_aggregated_intelligence, symbol=symbol, **kwargs)
    
    async def _safe_invoke(self, func: Callable, **kwargs) -> Dict[str, Any]:
        """Safely invoke an MCP tool with error handling."""
        try:
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return await func(**kwargs)
            else:
                return func(**kwargs)
        except Exception as e:
            logger.error(f"Feature tool error: {e}")
            return {"error": str(e), "success": False}


class VisualizationTools(ToolWrapper):
    """
    Wrapper for visualization tools (5 tools).
    
    Provides access to:
    - Feature candles
    - Liquidity heatmaps
    - Signal dashboards
    - Regime visualization
    - Correlation matrices
    """
    
    def _get_category(self) -> str:
        return "visualization"
    
    def _get_tools(self) -> Dict[str, Callable]:
        return {
            "get_feature_candles": self.get_candles,
            "get_liquidity_heatmap": self.get_heatmap,
            "get_signal_dashboard": self.get_dashboard,
            "get_regime_visualization": self.get_regime_viz,
            "get_correlation_matrix": self.get_correlation,
        }
    
    async def get_candles(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get feature candles."""
        from src.tools import get_feature_candles
        return await self._safe_invoke(get_feature_candles, symbol=symbol, **kwargs)
    
    async def get_heatmap(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get liquidity heatmap."""
        from src.tools import get_liquidity_heatmap
        return await self._safe_invoke(get_liquidity_heatmap, symbol=symbol, **kwargs)
    
    async def get_dashboard(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get signal dashboard."""
        from src.tools import get_signal_dashboard
        return await self._safe_invoke(get_signal_dashboard, symbol=symbol, **kwargs)
    
    async def get_regime_viz(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Get regime visualization."""
        from src.tools import get_regime_visualization
        return await self._safe_invoke(get_regime_visualization, symbol=symbol, **kwargs)
    
    async def get_correlation(self, symbols: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Get correlation matrix."""
        from src.tools import get_correlation_matrix
        return await self._safe_invoke(get_correlation_matrix, symbols=symbols, **kwargs)
    
    async def _safe_invoke(self, func: Callable, **kwargs) -> Dict[str, Any]:
        """Safely invoke an MCP tool with error handling."""
        try:
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return await func(**kwargs)
            else:
                return func(**kwargs)
        except Exception as e:
            logger.error(f"Visualization tool error: {e}")
            return {"error": str(e), "success": False}
