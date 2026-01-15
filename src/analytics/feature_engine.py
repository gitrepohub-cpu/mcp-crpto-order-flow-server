"""
Feature Engine - Main Orchestrator

Layer 6: The central engine that orchestrates all analytics layers
and produces unified market intelligence.

This is the entry point for all analytics computations.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque
import asyncio

from .order_flow_analytics import OrderFlowAnalytics
from .leverage_analytics import LeverageAnalytics
from .cross_exchange_analytics import CrossExchangeAnalytics
from .regime_analytics import RegimeAnalytics
from .alpha_signals import AlphaSignalEngine

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Feature Engine - Central Analytics Orchestrator.
    
    Coordinates all analytics layers and produces unified output.
    Designed for:
    - Real-time dashboard updates
    - Periodic computation cycles
    - Historical feature storage
    """
    
    def __init__(self):
        # Initialize all analytics layers
        self.order_flow = OrderFlowAnalytics()
        self.leverage = LeverageAnalytics()
        self.cross_exchange = CrossExchangeAnalytics()
        self.regime = RegimeAnalytics()
        self.alpha_signals = AlphaSignalEngine()
        
        # Feature history for time-series analysis
        self.feature_history: Dict[str, deque] = {}
        self.history_window = 1000  # Keep 1000 computation cycles
        
        # Computation stats
        self.computation_count = 0
        self.last_computation: Optional[datetime] = None
        self.avg_computation_time_ms = 0
        
    async def compute_all_features(
        self,
        symbol: str,
        data: Dict
    ) -> Dict:
        """
        Compute all features for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            data: Dictionary containing all raw data:
                - orderbook: Dict with bids/asks
                - trades: List of recent trades
                - funding_rates: Dict by exchange
                - open_interest: Dict by exchange
                - liquidations: List of liquidations
                - mark_prices: Dict by exchange
                - ticker_24h: Dict with 24h stats
                
        Returns:
            Dict with all computed features
        """
        start_time = datetime.utcnow()
        
        result = {
            "symbol": symbol,
            "timestamp": start_time.isoformat(),
            "computation_id": self.computation_count + 1,
            "layers": {},
            "summary": {},
            "errors": []
        }
        
        try:
            # =====================================================
            # Layer 1: Order Flow & Microstructure
            # =====================================================
            orderbook = data.get("orderbook", {})
            trades = data.get("trades", [])
            
            layer1 = {}
            try:
                layer1["liquidity_imbalance"] = self.order_flow.compute_liquidity_imbalance(
                    symbol, orderbook
                )
            except Exception as e:
                layer1["liquidity_imbalance"] = {"error": str(e)}
                result["errors"].append(f"liquidity_imbalance: {e}")
            
            try:
                prices = data.get("prices", {})
                layer1["liquidity_vacuum"] = self.order_flow.compute_liquidity_vacuum(
                    symbol, orderbook, prices
                )
            except Exception as e:
                layer1["liquidity_vacuum"] = {"error": str(e)}
                result["errors"].append(f"liquidity_vacuum: {e}")
            
            try:
                layer1["orderbook_persistence"] = self.order_flow.compute_orderbook_persistence(
                    symbol, orderbook
                )
            except Exception as e:
                layer1["orderbook_persistence"] = {"error": str(e)}
                result["errors"].append(f"orderbook_persistence: {e}")
            
            try:
                layer1["trade_aggression"] = self.order_flow.compute_trade_aggression(
                    symbol, trades
                )
            except Exception as e:
                layer1["trade_aggression"] = {"error": str(e)}
                result["errors"].append(f"trade_aggression: {e}")
            
            try:
                prices = data.get("prices", {})
                # Format trades as Dict[exchange, List[trades]]
                trades_by_exchange = data.get("all_exchange_trades", {})
                if not trades_by_exchange and trades:
                    trades_by_exchange = {"combined": trades}
                layer1["microstructure_efficiency"] = self.order_flow.compute_microstructure_efficiency(
                    symbol, trades_by_exchange, prices
                )
            except Exception as e:
                layer1["microstructure_efficiency"] = {"error": str(e)}
                result["errors"].append(f"microstructure_efficiency: {e}")
            
            result["layers"]["order_flow"] = layer1
            
            # =====================================================
            # Layer 2: Leverage, Positioning & Risk
            # =====================================================
            open_interest = data.get("open_interest", {})
            funding_rates = data.get("funding_rates", {})
            liquidations = data.get("liquidations", [])
            mark_prices = data.get("mark_prices", {})
            
            layer2 = {}
            try:
                layer2["oi_flow"] = self.leverage.compute_oi_flow_decomposition(
                    symbol, open_interest, trades
                )
            except Exception as e:
                layer2["oi_flow"] = {"error": str(e)}
                result["errors"].append(f"oi_flow: {e}")
            
            try:
                layer2["leverage_index"] = self.leverage.compute_leverage_index(
                    symbol, open_interest, trades
                )
            except Exception as e:
                layer2["leverage_index"] = {"error": str(e)}
                result["errors"].append(f"leverage_index: {e}")
            
            try:
                current_price = self._get_current_price(data.get("prices", {}), symbol)
                layer2["liquidation_pressure"] = self.leverage.compute_liquidation_pressure(
                    symbol, liquidations, current_price
                )
            except Exception as e:
                layer2["liquidation_pressure"] = {"error": str(e)}
                result["errors"].append(f"liquidation_pressure: {e}")
            
            try:
                layer2["funding_stress"] = self.leverage.compute_funding_stress(
                    symbol, funding_rates
                )
            except Exception as e:
                layer2["funding_stress"] = {"error": str(e)}
                result["errors"].append(f"funding_stress: {e}")
            
            try:
                layer2["basis_regime"] = self.leverage.compute_basis_regime(
                    symbol, mark_prices
                )
            except Exception as e:
                layer2["basis_regime"] = {"error": str(e)}
                result["errors"].append(f"basis_regime: {e}")
            
            result["layers"]["leverage"] = layer2
            
            # =====================================================
            # Layer 3: Cross-Exchange Intelligence
            # =====================================================
            prices = data.get("prices", {})
            all_trades = data.get("all_exchange_trades", {})  # {exchange: [trades]}
            
            layer3 = {}
            try:
                layer3["price_leadership"] = self.cross_exchange.compute_price_leadership(
                    symbol, prices
                )
            except Exception as e:
                layer3["price_leadership"] = {"error": str(e)}
                result["errors"].append(f"price_leadership: {e}")
            
            try:
                layer3["spread_arbitrage"] = self.cross_exchange.compute_spread_arbitrage(
                    symbol, prices, orderbook
                )
            except Exception as e:
                layer3["spread_arbitrage"] = {"error": str(e)}
                result["errors"].append(f"spread_arbitrage: {e}")
            
            try:
                layer3["flow_synchronization"] = self.cross_exchange.compute_flow_synchronization(
                    symbol, all_trades
                )
            except Exception as e:
                layer3["flow_synchronization"] = {"error": str(e)}
                result["errors"].append(f"flow_synchronization: {e}")
            
            result["layers"]["cross_exchange"] = layer3
            
            # =====================================================
            # Layer 4: Regime & Volatility
            # =====================================================
            ticker_24h = data.get("ticker_24h", {})
            
            layer4 = {}
            try:
                layer4["regime"] = self.regime.detect_regime(
                    symbol,
                    layer1,  # order_flow features
                    layer2   # leverage features
                )
            except Exception as e:
                layer4["regime"] = {"error": str(e)}
                result["errors"].append(f"regime: {e}")
            
            try:
                layer4["event_risk"] = self.regime.detect_event_risk(
                    symbol,
                    layer1,
                    layer2
                )
            except Exception as e:
                layer4["event_risk"] = {"error": str(e)}
                result["errors"].append(f"event_risk: {e}")
            
            try:
                layer4["volatility_state"] = self.regime.compute_volatility_state(
                    symbol, layer1  # Pass order_flow features instead of prices/tickers
                )
            except Exception as e:
                layer4["volatility_state"] = {"error": str(e)}
                result["errors"].append(f"volatility_state: {e}")
            
            result["layers"]["regime"] = layer4
            
            # =====================================================
            # Layer 5: Alpha Signals
            # =====================================================
            try:
                layer5 = self.alpha_signals.compute_all_signals(
                    symbol,
                    layer1,
                    layer2,
                    layer3,
                    layer4
                )
                result["layers"]["alpha_signals"] = layer5
            except Exception as e:
                result["layers"]["alpha_signals"] = {"error": str(e)}
                result["errors"].append(f"alpha_signals: {e}")
            
            # =====================================================
            # Generate Summary
            # =====================================================
            result["summary"] = self._generate_summary(symbol, result["layers"])
            
        except Exception as e:
            logger.error(f"Feature computation failed for {symbol}: {e}")
            result["errors"].append(f"Fatal error: {e}")
        
        # Computation stats
        end_time = datetime.utcnow()
        computation_time = (end_time - start_time).total_seconds() * 1000
        
        self.computation_count += 1
        self.last_computation = end_time
        self.avg_computation_time_ms = (
            (self.avg_computation_time_ms * (self.computation_count - 1) + computation_time)
            / self.computation_count
        )
        
        result["computation_time_ms"] = round(computation_time, 2)
        result["errors_count"] = len(result["errors"])
        
        # Store in history
        if symbol not in self.feature_history:
            self.feature_history[symbol] = deque(maxlen=self.history_window)
        
        self.feature_history[symbol].append({
            "timestamp": start_time,
            "summary": result["summary"],
            "computation_id": result["computation_id"]
        })
        
        return result
    
    def _get_current_price(self, prices: Dict, symbol: str) -> float:
        """Extract current price from prices dict."""
        if not prices:
            return 0.0
        
        # Try different exchange sources
        for exchange in ["binance_futures", "binance", "bybit_futures", "bybit", "okx"]:
            if exchange in prices:
                exchange_data = prices[exchange]
                if symbol in exchange_data:
                    return float(exchange_data[symbol].get("price", 0))
                # Try without symbol filter
                if "price" in exchange_data:
                    return float(exchange_data["price"])
        
        # Fallback: get first available price
        for exchange, data in prices.items():
            if isinstance(data, dict):
                if "price" in data:
                    return float(data["price"])
                if symbol in data:
                    return float(data[symbol].get("price", 0))
        
        return 0.0
    
    def _generate_summary(self, symbol: str, layers: Dict) -> Dict:
        """Generate high-level summary from all layers."""
        summary = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "market_bias": "NEUTRAL",
            "market_condition": "NORMAL",
            "risk_level": "LOW",
            "key_signals": [],
            "top_opportunities": [],
            "warnings": [],
            "scores": {}
        }
        
        # Extract key metrics
        order_flow = layers.get("order_flow", {})
        leverage = layers.get("leverage", {})
        cross_exchange = layers.get("cross_exchange", {})
        regime = layers.get("regime", {})
        alpha = layers.get("alpha_signals", {})
        
        # Market bias from alpha signals
        composite = alpha.get("composite_signal", {})
        signal = composite.get("signal", "NEUTRAL")
        
        if "BUY" in signal:
            summary["market_bias"] = "BULLISH" if "STRONG" in signal else "SLIGHTLY_BULLISH"
        elif "SELL" in signal:
            summary["market_bias"] = "BEARISH" if "STRONG" in signal else "SLIGHTLY_BEARISH"
        
        # Market condition from regime
        current_regime = regime.get("regime", {}).get("current_regime", "UNKNOWN")
        summary["market_condition"] = current_regime
        
        # Risk level
        event_risk = regime.get("event_risk", {})
        summary["risk_level"] = event_risk.get("risk_level", "LOW")
        
        # Key signals
        pressure = alpha.get("institutional_pressure", {})
        if pressure.get("pressure_strength") in ("STRONG", "MODERATE"):
            summary["key_signals"].append({
                "type": "INSTITUTIONAL_PRESSURE",
                "direction": pressure.get("pressure_direction"),
                "confidence": pressure.get("confidence")
            })
        
        squeeze = alpha.get("squeeze_probability", {})
        if squeeze.get("squeeze_probability", 0) > 50:
            summary["key_signals"].append({
                "type": "SQUEEZE_ALERT",
                "squeeze_type": squeeze.get("squeeze_type"),
                "probability": squeeze.get("squeeze_probability")
            })
        
        absorption = alpha.get("smart_money_absorption", {})
        if absorption.get("absorption_detected"):
            summary["key_signals"].append({
                "type": "SMART_MONEY_ABSORPTION",
                "absorption_type": absorption.get("absorption_type"),
                "strength": absorption.get("absorption_strength")
            })
        
        # Opportunities from cross-exchange
        arb = cross_exchange.get("spread_arbitrage", {})
        opportunities = arb.get("opportunities", [])
        for opp in opportunities[:3]:
            if opp.get("spread_pct", 0) > 0.05:
                summary["top_opportunities"].append({
                    "type": "ARBITRAGE",
                    "exchanges": [opp.get("buy_exchange"), opp.get("sell_exchange")],
                    "spread_pct": opp.get("spread_pct"),
                    "fill_probability": opp.get("fill_probability")
                })
        
        # Warnings
        warnings = event_risk.get("active_warnings", [])
        summary["warnings"] = warnings[:5]
        
        # Scores summary
        summary["scores"] = {
            "institutional_pressure": pressure.get("pressure_score", 0),
            "squeeze_probability": squeeze.get("squeeze_probability", 0),
            "absorption_strength": absorption.get("absorption_strength", 0),
            "event_risk_score": event_risk.get("risk_score", 0),
            "regime_confidence": regime.get("regime", {}).get("confidence", 0)
        }
        
        return summary
    
    # =========================================================================
    # Periodic Computation Interface
    # =========================================================================
    
    async def compute_periodic(
        self,
        symbol: str,
        data_source: Any,  # DirectExchangeClient or similar
        interval_seconds: float = 1.0,
        max_iterations: Optional[int] = None,
        callback: Optional[Any] = None
    ):
        """
        Run periodic feature computation.
        
        Args:
            symbol: Symbol to compute features for
            data_source: Object with get_all_data(symbol) method
            interval_seconds: Time between computations
            max_iterations: Optional limit on iterations
            callback: Optional callback(result) after each computation
        """
        iteration = 0
        
        while max_iterations is None or iteration < max_iterations:
            try:
                # Get fresh data
                data = await data_source.get_all_data(symbol)
                
                # Compute features
                result = await self.compute_all_features(symbol, data)
                
                # Call callback if provided
                if callback:
                    await callback(result)
                
                iteration += 1
                
            except Exception as e:
                logger.error(f"Periodic computation error: {e}")
            
            # Wait for next interval
            await asyncio.sleep(interval_seconds)
    
    # =========================================================================
    # Feature Access Methods
    # =========================================================================
    
    def get_feature_history(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Dict]:
        """Get historical feature summaries."""
        if symbol not in self.feature_history:
            return []
        
        history = list(self.feature_history[symbol])
        return history[-limit:]
    
    def get_computation_stats(self) -> Dict:
        """Get computation statistics."""
        return {
            "total_computations": self.computation_count,
            "last_computation": self.last_computation.isoformat() if self.last_computation else None,
            "avg_computation_time_ms": round(self.avg_computation_time_ms, 2),
            "symbols_tracked": list(self.feature_history.keys()),
            "history_depth": {
                symbol: len(history) 
                for symbol, history in self.feature_history.items()
            }
        }
    
    def get_layer_features(
        self,
        layer_name: str
    ) -> Dict:
        """Get description of features for a specific layer."""
        layer_descriptions = {
            "order_flow": {
                "name": "Order Flow & Microstructure",
                "features": [
                    "liquidity_imbalance",
                    "liquidity_vacuum",
                    "orderbook_persistence",
                    "trade_aggression",
                    "microstructure_efficiency"
                ],
                "description": "DOM intelligence and trade flow analysis"
            },
            "leverage": {
                "name": "Leverage & Positioning",
                "features": [
                    "oi_flow",
                    "leverage_index",
                    "liquidation_pressure",
                    "funding_stress",
                    "basis_regime"
                ],
                "description": "Risk flows and position intelligence"
            },
            "cross_exchange": {
                "name": "Cross-Exchange Intelligence",
                "features": [
                    "price_leadership",
                    "spread_arbitrage",
                    "flow_synchronization"
                ],
                "description": "Multi-venue flow analysis"
            },
            "regime": {
                "name": "Regime & Volatility",
                "features": [
                    "regime",
                    "event_risk",
                    "volatility_state"
                ],
                "description": "Market state classification"
            },
            "alpha_signals": {
                "name": "Alpha Signals",
                "features": [
                    "institutional_pressure",
                    "squeeze_probability",
                    "smart_money_absorption",
                    "composite_signal"
                ],
                "description": "Composite trading signals"
            }
        }
        
        return layer_descriptions.get(layer_name, {"error": f"Unknown layer: {layer_name}"})
