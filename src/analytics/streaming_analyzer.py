"""
Streaming Analyzer

Collects market data over a specified time window and computes analytics.
Designed for user-specified analysis durations.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class StreamingAnalyzer:
    """
    Collects streaming data over a time window and computes comprehensive analytics.
    
    Usage:
        analyzer = StreamingAnalyzer()
        result = await analyzer.analyze_stream(
            symbol="BTCUSDT",
            duration_seconds=30,
            client=exchange_client
        )
    """
    
    def __init__(self):
        self.collected_data = defaultdict(list)
        
    async def analyze_stream(
        self,
        symbol: str,
        duration_seconds: int,
        client: Any,
        sample_interval: float = 0.5
    ) -> Dict:
        """
        Stream and analyze market data for a specified duration.
        
        Args:
            symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
            duration_seconds: How long to collect data (5-300 seconds)
            client: Exchange client with data methods
            sample_interval: Time between samples (default 0.5s)
            
        Returns:
            Comprehensive analysis of the collected data
        """
        # Validate duration
        duration_seconds = max(5, min(300, duration_seconds))
        
        result = {
            "symbol": symbol,
            "analysis_start": datetime.utcnow().isoformat(),
            "analysis_end": None,
            "duration_seconds": duration_seconds,
            "samples_collected": 0,
            "data_summary": {},
            "price_analysis": {},
            "volume_analysis": {},
            "orderbook_analysis": {},
            "funding_analysis": {},
            "liquidation_analysis": {},
            "flow_analysis": {},
            "regime_analysis": {},
            "signals": {},
            "errors": []
        }
        
        # Data collectors
        prices = []
        spreads = []
        orderbook_snapshots = []
        trades_collected = []
        funding_snapshots = []
        liquidations_collected = []
        oi_snapshots = []
        
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(seconds=duration_seconds)
        sample_count = 0
        
        logger.info(f"Starting {duration_seconds}s stream analysis for {symbol}")
        
        try:
            while datetime.utcnow() < end_time:
                try:
                    # Collect snapshot
                    snapshot_time = datetime.utcnow()
                    
                    # Get prices
                    price_data = await client.get_prices_snapshot(symbol)
                    if price_data:
                        prices.append({
                            "timestamp": snapshot_time,
                            "data": price_data
                        })
                    
                    # Get orderbooks - returns {symbol: {exchange: orderbook}}
                    ob_data = await client.get_orderbooks(symbol)
                    if ob_data:
                        # Extract orderbooks for this symbol
                        sym_obs = ob_data.get(symbol, {})
                        if sym_obs:
                            orderbook_snapshots.append({
                                "timestamp": snapshot_time,
                                "data": sym_obs
                            })
                            # Calculate spread
                            for ex, ob in sym_obs.items():
                                if ob.get("bids") and ob.get("asks"):
                                    best_bid = ob["bids"][0][0] if ob["bids"] else 0
                                    best_ask = ob["asks"][0][0] if ob["asks"] else 0
                                    if best_bid and best_ask:
                                        spread_pct = ((best_ask - best_bid) / best_bid) * 100
                                        spreads.append({
                                            "exchange": ex,
                                            "spread_pct": spread_pct,
                                            "timestamp": snapshot_time
                                        })
                    
                    # Get trades - returns {symbol: {exchange: [trades]}}
                    trade_data = await client.get_trades(symbol)
                    if trade_data:
                        sym_trades = trade_data.get(symbol, {})
                        for ex, trades_list in sym_trades.items():
                            if trades_list:
                                trades_collected.extend(trades_list)
                    
                    # Get funding (less frequently)
                    if sample_count % 4 == 0:
                        funding_data = await client.get_funding_rates(symbol)
                        if funding_data:
                            funding_snapshots.append({
                                "timestamp": snapshot_time,
                                "data": funding_data
                            })
                    
                    # Get liquidations - returns {symbol: [liquidations]}
                    liq_data = await client.get_liquidations(symbol)
                    if liq_data:
                        sym_liqs = liq_data.get(symbol, [])
                        if sym_liqs:
                            liquidations_collected.extend(sym_liqs)
                    
                    # Get OI (less frequently)
                    if sample_count % 4 == 0:
                        oi_data = await client.get_open_interest(symbol)
                        if oi_data:
                            oi_snapshots.append({
                                "timestamp": snapshot_time,
                                "data": oi_data
                            })
                    
                    sample_count += 1
                    
                except Exception as e:
                    result["errors"].append(f"Sample {sample_count}: {str(e)}")
                
                await asyncio.sleep(sample_interval)
                
        except Exception as e:
            result["errors"].append(f"Stream error: {str(e)}")
        
        result["analysis_end"] = datetime.utcnow().isoformat()
        result["samples_collected"] = sample_count
        
        # =====================================================================
        # Analyze collected data
        # =====================================================================
        
        # Data summary
        result["data_summary"] = {
            "price_snapshots": len(prices),
            "orderbook_snapshots": len(orderbook_snapshots),
            "trades_collected": len(trades_collected),
            "funding_snapshots": len(funding_snapshots),
            "liquidations_collected": len(liquidations_collected),
            "oi_snapshots": len(oi_snapshots),
            "spread_samples": len(spreads)
        }
        
        # Price analysis
        result["price_analysis"] = self._analyze_prices(prices, symbol)
        
        # Volume analysis
        result["volume_analysis"] = self._analyze_volume(trades_collected)
        
        # Orderbook analysis
        result["orderbook_analysis"] = self._analyze_orderbook(orderbook_snapshots, spreads)
        
        # Funding analysis
        result["funding_analysis"] = self._analyze_funding(funding_snapshots)
        
        # Liquidation analysis
        result["liquidation_analysis"] = self._analyze_liquidations(liquidations_collected)
        
        # Flow analysis (trade direction)
        result["flow_analysis"] = self._analyze_flow(trades_collected)
        
        # Regime analysis
        result["regime_analysis"] = self._detect_regime(
            result["price_analysis"],
            result["volume_analysis"],
            result["flow_analysis"]
        )
        
        # Generate signals
        result["signals"] = self._generate_signals(result)
        
        logger.info(f"Completed {duration_seconds}s analysis for {symbol}: {sample_count} samples")
        
        return result
    
    def _analyze_prices(self, prices: List[Dict], symbol: str) -> Dict:
        """Analyze price movements over the collection period."""
        if not prices:
            return {"error": "No price data collected"}
        
        # Extract all prices by exchange
        # price_data format: {symbol: {exchange: price_value}}
        exchange_prices = defaultdict(list)
        timestamps = []
        
        for snapshot in prices:
            timestamps.append(snapshot["timestamp"])
            data = snapshot["data"]
            # Get prices for the symbol
            sym_prices = data.get(symbol, {})
            for exchange, price_value in sym_prices.items():
                # price_value can be a dict with 'price' key or just the price
                if isinstance(price_value, dict):
                    price = price_value.get("price", 0)
                else:
                    price = price_value
                if price and price > 0:
                    exchange_prices[exchange].append(price)
        
        if not exchange_prices:
            return {"error": "No valid prices extracted"}
        
        # Calculate stats per exchange
        exchange_stats = {}
        all_prices = []
        
        for exchange, price_list in exchange_prices.items():
            if len(price_list) >= 2:
                all_prices.extend(price_list)
                exchange_stats[exchange] = {
                    "start_price": price_list[0],
                    "end_price": price_list[-1],
                    "high": max(price_list),
                    "low": min(price_list),
                    "mean": round(statistics.mean(price_list), 2),
                    "change_pct": round(((price_list[-1] - price_list[0]) / price_list[0]) * 100, 4),
                    "volatility": round(statistics.stdev(price_list), 4) if len(price_list) > 1 else 0
                }
        
        # Aggregate stats
        if all_prices:
            return {
                "start_price": all_prices[0],
                "end_price": all_prices[-1],
                "high": max(all_prices),
                "low": min(all_prices),
                "range_pct": round(((max(all_prices) - min(all_prices)) / min(all_prices)) * 100, 4),
                "change_pct": round(((all_prices[-1] - all_prices[0]) / all_prices[0]) * 100, 4),
                "direction": "UP" if all_prices[-1] > all_prices[0] else "DOWN" if all_prices[-1] < all_prices[0] else "FLAT",
                "volatility": round(statistics.stdev(all_prices), 4) if len(all_prices) > 1 else 0,
                "price_samples": len(all_prices),
                "by_exchange": exchange_stats
            }
        
        return {"error": "Insufficient price data"}
    
    def _analyze_volume(self, trades: List[Dict]) -> Dict:
        """Analyze trading volume patterns."""
        if not trades:
            return {"error": "No trade data collected"}
        
        total_volume = 0
        total_value = 0
        buy_volume = 0
        sell_volume = 0
        buy_count = 0
        sell_count = 0
        trade_sizes = []
        
        for trade in trades:
            qty = abs(trade.get("quantity", 0))
            price = trade.get("price", 0)
            value = qty * price if price else trade.get("value", 0)
            side = trade.get("side", "").lower()
            
            total_volume += qty
            total_value += value
            trade_sizes.append(qty)
            
            if side == "buy":
                buy_volume += qty
                buy_count += 1
            elif side == "sell":
                sell_volume += qty
                sell_count += 1
        
        # Large trade detection (top 10%)
        if trade_sizes:
            threshold = sorted(trade_sizes, reverse=True)[max(0, len(trade_sizes) // 10)]
            large_trades = [t for t in trades if abs(t.get("quantity", 0)) >= threshold]
        else:
            large_trades = []
        
        total_trades = buy_count + sell_count
        
        return {
            "total_volume": round(total_volume, 4),
            "total_value_usd": round(total_value, 2),
            "total_trades": len(trades),
            "buy_volume": round(buy_volume, 4),
            "sell_volume": round(sell_volume, 4),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_sell_ratio": round(buy_volume / sell_volume, 4) if sell_volume > 0 else float('inf'),
            "avg_trade_size": round(total_volume / len(trades), 6) if trades else 0,
            "large_trade_count": len(large_trades),
            "large_trade_volume": round(sum(abs(t.get("quantity", 0)) for t in large_trades), 4),
            "volume_imbalance": round((buy_volume - sell_volume) / total_volume * 100, 2) if total_volume > 0 else 0
        }
    
    def _analyze_orderbook(self, snapshots: List[Dict], spreads: List[Dict]) -> Dict:
        """Analyze orderbook dynamics."""
        if not snapshots:
            return {"error": "No orderbook data collected"}
        
        # Spread analysis
        spread_values = [s["spread_pct"] for s in spreads if s.get("spread_pct")]
        
        # Depth analysis from last snapshot
        last_snapshot = snapshots[-1]["data"] if snapshots else {}
        
        total_bid_depth = 0
        total_ask_depth = 0
        
        for exchange, ob in last_snapshot.items():
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            
            for bid in bids[:10]:
                total_bid_depth += bid[0] * bid[1] if len(bid) >= 2 else 0
            for ask in asks[:10]:
                total_ask_depth += ask[0] * ask[1] if len(ask) >= 2 else 0
        
        imbalance = (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth) if (total_bid_depth + total_ask_depth) > 0 else 0
        
        return {
            "snapshots_analyzed": len(snapshots),
            "avg_spread_pct": round(statistics.mean(spread_values), 6) if spread_values else 0,
            "min_spread_pct": round(min(spread_values), 6) if spread_values else 0,
            "max_spread_pct": round(max(spread_values), 6) if spread_values else 0,
            "spread_volatility": round(statistics.stdev(spread_values), 6) if len(spread_values) > 1 else 0,
            "bid_depth_usd": round(total_bid_depth, 2),
            "ask_depth_usd": round(total_ask_depth, 2),
            "depth_imbalance": round(imbalance, 4),
            "imbalance_signal": "BULLISH" if imbalance > 0.1 else "BEARISH" if imbalance < -0.1 else "NEUTRAL"
        }
    
    def _analyze_funding(self, snapshots: List[Dict]) -> Dict:
        """Analyze funding rate patterns."""
        if not snapshots:
            return {"error": "No funding data collected"}
        
        rates = []
        for snapshot in snapshots:
            for exchange, data in snapshot["data"].items():
                rate = data.get("funding_rate", 0)
                if rate:
                    rates.append(rate)
        
        if not rates:
            return {"error": "No valid funding rates"}
        
        avg_rate = statistics.mean(rates)
        
        return {
            "snapshots_analyzed": len(snapshots),
            "avg_funding_rate": round(avg_rate, 8),
            "annualized_rate": round(avg_rate * 3 * 365 * 100, 2),  # 8h funding, annualized %
            "min_rate": round(min(rates), 8),
            "max_rate": round(max(rates), 8),
            "sentiment": "BULLISH_CROWDED" if avg_rate > 0.0001 else "BEARISH_CROWDED" if avg_rate < -0.0001 else "NEUTRAL",
            "rate_stability": round(statistics.stdev(rates), 8) if len(rates) > 1 else 0
        }
    
    def _analyze_liquidations(self, liquidations: List[Dict]) -> Dict:
        """Analyze liquidation patterns."""
        if not liquidations:
            return {
                "total_liquidations": 0,
                "long_liquidations": 0,
                "short_liquidations": 0,
                "message": "No liquidations during analysis period"
            }
        
        # Deduplicate by trade_id if available
        seen = set()
        unique_liqs = []
        for liq in liquidations:
            liq_id = liq.get("trade_id", str(liq))
            if liq_id not in seen:
                seen.add(liq_id)
                unique_liqs.append(liq)
        
        long_value = 0
        short_value = 0
        long_count = 0
        short_count = 0
        
        for liq in unique_liqs:
            side = liq.get("side", "").lower()
            value = abs(liq.get("value", 0))
            
            if side in ("long", "buy"):
                long_value += value
                long_count += 1
            elif side in ("short", "sell"):
                short_value += value
                short_count += 1
        
        total_value = long_value + short_value
        
        return {
            "total_liquidations": len(unique_liqs),
            "long_liquidations": long_count,
            "short_liquidations": short_count,
            "long_value_usd": round(long_value, 2),
            "short_value_usd": round(short_value, 2),
            "total_value_usd": round(total_value, 2),
            "dominant_side": "LONGS" if long_value > short_value * 1.5 else "SHORTS" if short_value > long_value * 1.5 else "BALANCED",
            "cascade_risk": "HIGH" if len(unique_liqs) > 10 else "MODERATE" if len(unique_liqs) > 5 else "LOW"
        }
    
    def _analyze_flow(self, trades: List[Dict]) -> Dict:
        """Analyze order flow and trade direction."""
        if not trades:
            return {"error": "No trade data for flow analysis"}
        
        # Calculate delta (buy - sell volume)
        buy_volume = sum(abs(t.get("quantity", 0)) for t in trades if t.get("side", "").lower() == "buy")
        sell_volume = sum(abs(t.get("quantity", 0)) for t in trades if t.get("side", "").lower() == "sell")
        
        delta = buy_volume - sell_volume
        total = buy_volume + sell_volume
        delta_pct = (delta / total * 100) if total > 0 else 0
        
        # CVD trend (simplified)
        cvd = 0
        cvd_values = []
        for trade in trades:
            side = trade.get("side", "").lower()
            qty = abs(trade.get("quantity", 0))
            if side == "buy":
                cvd += qty
            elif side == "sell":
                cvd -= qty
            cvd_values.append(cvd)
        
        # Aggressor analysis
        if delta_pct > 20:
            aggressor = "STRONG_BUYERS"
        elif delta_pct > 5:
            aggressor = "BUYERS"
        elif delta_pct < -20:
            aggressor = "STRONG_SELLERS"
        elif delta_pct < -5:
            aggressor = "SELLERS"
        else:
            aggressor = "BALANCED"
        
        return {
            "delta": round(delta, 4),
            "delta_pct": round(delta_pct, 2),
            "cvd_final": round(cvd, 4),
            "cvd_trend": "RISING" if len(cvd_values) > 1 and cvd_values[-1] > cvd_values[0] else "FALLING" if len(cvd_values) > 1 and cvd_values[-1] < cvd_values[0] else "FLAT",
            "aggressor": aggressor,
            "flow_signal": "BULLISH" if delta_pct > 10 else "BEARISH" if delta_pct < -10 else "NEUTRAL"
        }
    
    def _detect_regime(self, price_analysis: Dict, volume_analysis: Dict, flow_analysis: Dict) -> Dict:
        """Detect market regime from collected data."""
        
        price_change = price_analysis.get("change_pct", 0)
        volatility = price_analysis.get("volatility", 0)
        volume_imbalance = volume_analysis.get("volume_imbalance", 0)
        delta_pct = flow_analysis.get("delta_pct", 0)
        
        # Regime detection logic
        if abs(price_change) > 1 and abs(delta_pct) > 30:
            regime = "BREAKOUT"
            description = "Strong directional move with heavy flow"
        elif abs(price_change) < 0.1 and abs(delta_pct) < 5:
            regime = "CONSOLIDATION"
            description = "Low volatility, balanced flow"
        elif price_change > 0.3 and delta_pct > 15:
            regime = "ACCUMULATION"
            description = "Quiet buying absorption"
        elif price_change < -0.3 and delta_pct < -15:
            regime = "DISTRIBUTION"
            description = "Quiet selling pressure"
        elif volatility > price_analysis.get("mean", 0) * 0.001:
            regime = "HIGH_VOLATILITY"
            description = "Elevated price swings"
        else:
            regime = "NORMAL"
            description = "Standard market conditions"
        
        return {
            "detected_regime": regime,
            "description": description,
            "price_direction": price_analysis.get("direction", "UNKNOWN"),
            "flow_direction": flow_analysis.get("flow_signal", "NEUTRAL"),
            "volatility_state": "HIGH" if volatility > 10 else "NORMAL" if volatility > 1 else "LOW"
        }
    
    def _generate_signals(self, analysis: Dict) -> Dict:
        """Generate trading signals from analysis."""
        
        price = analysis.get("price_analysis", {})
        volume = analysis.get("volume_analysis", {})
        flow = analysis.get("flow_analysis", {})
        regime = analysis.get("regime_analysis", {})
        liquidations = analysis.get("liquidation_analysis", {})
        orderbook = analysis.get("orderbook_analysis", {})
        
        signals = []
        confidence = 50
        
        # Price direction signal
        if price.get("change_pct", 0) > 0.5:
            signals.append("PRICE_UP")
            confidence += 10
        elif price.get("change_pct", 0) < -0.5:
            signals.append("PRICE_DOWN")
            confidence += 10
        
        # Flow signal
        if flow.get("delta_pct", 0) > 15:
            signals.append("BULLISH_FLOW")
            confidence += 15
        elif flow.get("delta_pct", 0) < -15:
            signals.append("BEARISH_FLOW")
            confidence += 15
        
        # Orderbook signal
        if orderbook.get("imbalance_signal") == "BULLISH":
            signals.append("BULLISH_ORDERBOOK")
            confidence += 10
        elif orderbook.get("imbalance_signal") == "BEARISH":
            signals.append("BEARISH_ORDERBOOK")
            confidence += 10
        
        # Liquidation signal
        if liquidations.get("dominant_side") == "LONGS":
            signals.append("LONG_LIQUIDATIONS")
            confidence -= 5
        elif liquidations.get("dominant_side") == "SHORTS":
            signals.append("SHORT_SQUEEZE_POTENTIAL")
            confidence += 5
        
        # Determine overall bias
        bullish_signals = sum(1 for s in signals if "BULLISH" in s or "UP" in s or "SQUEEZE" in s)
        bearish_signals = sum(1 for s in signals if "BEARISH" in s or "DOWN" in s or "LONG_LIQ" in s)
        
        if bullish_signals > bearish_signals + 1:
            bias = "BULLISH"
        elif bearish_signals > bullish_signals + 1:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"
        
        return {
            "overall_bias": bias,
            "confidence": min(100, max(0, confidence)),
            "active_signals": signals,
            "signal_count": len(signals),
            "regime": regime.get("detected_regime", "UNKNOWN"),
            "recommendation": self._get_recommendation(bias, confidence, regime.get("detected_regime"))
        }
    
    def _get_recommendation(self, bias: str, confidence: int, regime: str) -> str:
        """Generate actionable recommendation."""
        if regime == "CONSOLIDATION":
            return "WAIT - Market consolidating, wait for breakout"
        elif regime == "BREAKOUT":
            if bias == "BULLISH":
                return "FAVOR LONGS - Breakout with bullish flow"
            elif bias == "BEARISH":
                return "FAVOR SHORTS - Breakout with bearish flow"
        elif confidence > 70:
            if bias == "BULLISH":
                return "FAVOR LONGS - Strong bullish signals"
            elif bias == "BEARISH":
                return "FAVOR SHORTS - Strong bearish signals"
        
        return "NEUTRAL - No clear edge, wait for better setup"
