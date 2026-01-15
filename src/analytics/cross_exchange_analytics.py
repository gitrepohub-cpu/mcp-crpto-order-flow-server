"""
Cross-Exchange Flow & Arbitrage Intelligence

Layer 3: Provides institutional flow routing visibility.

Derived from: Multi-exchange Prices, Trades, Orderbooks
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import math

logger = logging.getLogger(__name__)


class CrossExchangeAnalytics:
    """
    Cross-Exchange Flow & Arbitrage Intelligence Engine.
    
    Features:
    3.1 Price Leadership Detection
    3.2 Cross-Exchange Spread Arbitrage
    3.3 Flow Synchronization Index
    """
    
    def __init__(self):
        # Historical tracking for lag analysis
        self.price_series: Dict[str, Dict[str, deque]] = {}  # symbol -> exchange -> prices
        self.trade_series: Dict[str, Dict[str, deque]] = {}  # symbol -> exchange -> trade events
        self.spread_history: Dict[str, deque] = {}  # symbol -> spread values
        
        self.history_window = 300  # samples
        
    def compute_all_features(
        self,
        symbol: str,
        prices: Dict[str, Dict],
        trades: Dict[str, List[Dict]],
        orderbooks: Dict[str, Dict]
    ) -> Dict:
        """
        Compute all cross-exchange features.
        """
        result = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "price_leadership": self.compute_price_leadership(symbol, prices),
            "spread_arbitrage": self.compute_spread_arbitrage(symbol, prices, orderbooks),
            "flow_synchronization": self.compute_flow_synchronization(symbol, trades)
        }
        
        # Update historical series
        self._update_series(symbol, prices, trades)
        
        return result
    
    # =========================================================================
    # 3.1 Price Leadership Detection
    # =========================================================================
    
    def compute_price_leadership(
        self,
        symbol: str,
        prices: Dict[str, Dict]
    ) -> Dict:
        """
        Identify which exchange leads price discovery.
        
        Method:
        - Lag correlation analysis
        - Granger causality (simplified)
        - First-mover detection
        """
        result = {
            "leader_exchange": "UNKNOWN",
            "leader_score": 0,
            "leadership_confidence": 0,
            "exchange_rankings": [],
            "lag_analysis": {}
        }
        
        if symbol not in self.price_series:
            self.price_series[symbol] = {}
            for exchange in prices.keys():
                self.price_series[symbol][exchange] = deque(maxlen=self.history_window)
        
        # Need sufficient history
        min_samples = 20
        exchanges_with_data = []
        
        for exchange, price_data in prices.items():
            if exchange in self.price_series[symbol]:
                if len(self.price_series[symbol][exchange]) >= min_samples:
                    exchanges_with_data.append(exchange)
        
        if len(exchanges_with_data) < 2:
            return result
        
        # Compute price returns for each exchange
        returns = {}
        for exchange in exchanges_with_data:
            price_list = list(self.price_series[symbol][exchange])
            if len(price_list) >= 2:
                returns[exchange] = [
                    (price_list[i] - price_list[i-1]) / price_list[i-1] 
                    if price_list[i-1] > 0 else 0
                    for i in range(1, len(price_list))
                ]
        
        # Leadership scoring based on:
        # 1. Who moves first (variance of returns)
        # 2. Cross-correlation at lag 0 vs lag 1
        
        leadership_scores = {}
        
        for exchange, ret in returns.items():
            if not ret:
                continue
                
            # Variance (more variance = more active)
            mean_ret = sum(ret) / len(ret)
            variance = sum((r - mean_ret) ** 2 for r in ret) / len(ret)
            
            # Count how often this exchange moves first (return != 0 when others = 0)
            first_mover_count = 0
            for i in range(len(ret)):
                if abs(ret[i]) > 0.0001:  # This exchange moved
                    others_moved = sum(
                        1 for other_ex, other_ret in returns.items()
                        if other_ex != exchange and i < len(other_ret) and abs(other_ret[i]) > 0.0001
                    )
                    if others_moved == 0:
                        first_mover_count += 1
            
            first_mover_ratio = first_mover_count / len(ret) if ret else 0
            
            # Leadership score combines variance and first-mover frequency
            leadership_scores[exchange] = {
                "variance": variance,
                "first_mover_ratio": first_mover_ratio,
                "score": variance * 1000 + first_mover_ratio * 100
            }
        
        # Rank exchanges
        rankings = sorted(
            leadership_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        if rankings:
            result["leader_exchange"] = rankings[0][0]
            result["leader_score"] = round(rankings[0][1]["score"], 4)
            
            # Confidence based on score differential
            if len(rankings) > 1:
                score_diff = rankings[0][1]["score"] - rankings[1][1]["score"]
                result["leadership_confidence"] = min(score_diff * 100, 100)
            else:
                result["leadership_confidence"] = 50
            
            result["exchange_rankings"] = [
                {
                    "exchange": ex,
                    "score": round(data["score"], 4),
                    "variance": round(data["variance"], 8),
                    "first_mover_ratio": round(data["first_mover_ratio"], 4)
                }
                for ex, data in rankings
            ]
        
        # Lag analysis between leader and followers
        if result["leader_exchange"] != "UNKNOWN":
            leader = result["leader_exchange"]
            leader_ret = returns.get(leader, [])
            
            for exchange in exchanges_with_data:
                if exchange == leader:
                    continue
                
                follower_ret = returns.get(exchange, [])
                if not follower_ret or not leader_ret:
                    continue
                
                # Cross-correlation at different lags
                correlations = {}
                for lag in range(0, min(5, len(leader_ret))):
                    corr = self._compute_correlation(
                        leader_ret[lag:],
                        follower_ret[:len(follower_ret)-lag] if lag > 0 else follower_ret
                    )
                    correlations[f"lag_{lag}"] = round(corr, 4)
                
                # Best lag
                best_lag = max(correlations.items(), key=lambda x: abs(x[1]))[0]
                
                result["lag_analysis"][exchange] = {
                    "correlations": correlations,
                    "best_lag": best_lag,
                    "lag_ms_estimate": int(best_lag.split("_")[1]) * 100  # Rough estimate
                }
        
        return result
    
    def _compute_correlation(self, x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation between two series."""
        n = min(len(x), len(y))
        if n < 2:
            return 0
        
        x = x[:n]
        y = y[:n]
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)
        
        if std_x == 0 or std_y == 0:
            return 0
        
        return cov / (std_x * std_y)
    
    # =========================================================================
    # 3.2 Cross-Exchange Spread Arbitrage
    # =========================================================================
    
    def compute_spread_arbitrage(
        self,
        symbol: str,
        prices: Dict[str, Dict],
        orderbooks: Dict[str, Dict]
    ) -> Dict:
        """
        Compute cross-exchange arbitrage opportunities.
        
        Features:
        - Inter-exchange basis
        - Arbitrage window duration
        - Fill probability
        - Slippage risk
        """
        result = {
            "max_spread_pct": 0,
            "min_spread_pct": 0,
            "avg_spread_pct": 0,
            "best_opportunity": None,
            "opportunities": [],
            "spread_matrix": {},
            "arbitrage_windows": []
        }
        
        # Extract prices
        exchange_prices = {}
        for exchange, data in prices.items():
            if isinstance(data, dict):
                price = data.get('price', 0)
            else:
                price = data
            if price > 0:
                exchange_prices[exchange] = price
        
        if len(exchange_prices) < 2:
            return result
        
        # Build spread matrix
        spreads = []
        opportunities = []
        
        exchanges = list(exchange_prices.keys())
        for i, buy_ex in enumerate(exchanges):
            result["spread_matrix"][buy_ex] = {}
            for j, sell_ex in enumerate(exchanges):
                if i == j:
                    result["spread_matrix"][buy_ex][sell_ex] = 0
                    continue
                
                buy_price = exchange_prices[buy_ex]
                sell_price = exchange_prices[sell_ex]
                spread_pct = ((sell_price - buy_price) / buy_price) * 100
                
                result["spread_matrix"][buy_ex][sell_ex] = round(spread_pct, 4)
                spreads.append(spread_pct)
                
                # Positive spread = potential profit
                if spread_pct > 0.01:  # > 0.01% profit
                    # Estimate execution quality from orderbooks
                    fill_prob = 0.8  # Base probability
                    slippage_risk = "LOW"
                    
                    buy_ob = orderbooks.get(buy_ex, {})
                    sell_ob = orderbooks.get(sell_ex, {})
                    
                    if buy_ob and sell_ob:
                        buy_spread = buy_ob.get('spread_pct', 0)
                        sell_spread = sell_ob.get('spread_pct', 0)
                        
                        # Wider spreads = lower fill probability
                        avg_book_spread = (buy_spread + sell_spread) / 2
                        fill_prob = max(0.5, 1 - avg_book_spread / 10)
                        
                        # Slippage risk based on depth
                        buy_depth = buy_ob.get('ask_depth', 0)
                        sell_depth = sell_ob.get('bid_depth', 0)
                        
                        if buy_depth < 1 or sell_depth < 1:
                            slippage_risk = "HIGH"
                        elif buy_depth < 5 or sell_depth < 5:
                            slippage_risk = "MODERATE"
                    
                    # Net expected profit
                    net_profit = spread_pct * fill_prob
                    
                    opportunity = {
                        "buy_exchange": buy_ex,
                        "sell_exchange": sell_ex,
                        "buy_price": round(buy_price, 2),
                        "sell_price": round(sell_price, 2),
                        "spread_pct": round(spread_pct, 4),
                        "fill_probability": round(fill_prob, 2),
                        "slippage_risk": slippage_risk,
                        "net_expected_profit_pct": round(net_profit, 4)
                    }
                    opportunities.append(opportunity)
        
        # Sort by net profit
        opportunities = sorted(
            opportunities,
            key=lambda x: x["net_expected_profit_pct"],
            reverse=True
        )
        
        result["opportunities"] = opportunities[:10]
        if opportunities:
            result["best_opportunity"] = opportunities[0]
        
        if spreads:
            result["max_spread_pct"] = round(max(spreads), 4)
            result["min_spread_pct"] = round(min(spreads), 4)
            result["avg_spread_pct"] = round(sum(spreads) / len(spreads), 4)
        
        # Track spread windows over time
        if symbol not in self.spread_history:
            self.spread_history[symbol] = deque(maxlen=self.history_window)
        
        if opportunities:
            self.spread_history[symbol].append({
                "timestamp": datetime.utcnow(),
                "max_spread": result["max_spread_pct"]
            })
            
            # Analyze window duration
            if len(self.spread_history[symbol]) > 10:
                recent_spreads = list(self.spread_history[symbol])[-30:]
                profitable_windows = []
                current_window = None
                
                for entry in recent_spreads:
                    if entry["max_spread"] > 0.05:  # > 0.05% spread
                        if current_window is None:
                            current_window = {"start": entry["timestamp"], "spreads": []}
                        current_window["spreads"].append(entry["max_spread"])
                    else:
                        if current_window is not None:
                            current_window["end"] = entry["timestamp"]
                            current_window["duration_samples"] = len(current_window["spreads"])
                            current_window["avg_spread"] = sum(current_window["spreads"]) / len(current_window["spreads"])
                            profitable_windows.append(current_window)
                            current_window = None
                
                if profitable_windows:
                    result["arbitrage_windows"] = [
                        {
                            "duration_samples": w["duration_samples"],
                            "avg_spread_pct": round(w["avg_spread"], 4)
                        }
                        for w in profitable_windows[-5:]
                    ]
        
        return result
    
    # =========================================================================
    # 3.3 Flow Synchronization Index
    # =========================================================================
    
    def compute_flow_synchronization(
        self,
        symbol: str,
        trades: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Detect coordinated institutional flow.
        
        Features:
        - Trade direction correlation
        - Volume burst synchronization
        - Price jump correlation
        """
        result = {
            "synchronization_score": 0,  # 0-100
            "flow_correlation": 0,
            "volume_sync_score": 0,
            "coordinated_flow_detected": False,
            "dominant_direction": "NEUTRAL",
            "exchange_flows": {}
        }
        
        if not trades or len(trades) < 2:
            return result
        
        # Compute flow metrics per exchange
        exchange_flows = {}
        
        for exchange, trade_list in trades.items():
            if not trade_list:
                continue
            
            buy_vol = sum(t.get('quantity', 0) for t in trade_list if t.get('side') == 'buy')
            sell_vol = sum(t.get('quantity', 0) for t in trade_list if t.get('side') == 'sell')
            total_vol = buy_vol + sell_vol
            
            if total_vol > 0:
                delta = buy_vol - sell_vol
                delta_pct = (delta / total_vol) * 100
                
                exchange_flows[exchange] = {
                    "buy_volume": buy_vol,
                    "sell_volume": sell_vol,
                    "delta": delta,
                    "delta_pct": delta_pct,
                    "total_volume": total_vol,
                    "trade_count": len(trade_list)
                }
        
        result["exchange_flows"] = {
            k: {
                "delta_pct": round(v["delta_pct"], 2),
                "total_volume": round(v["total_volume"], 4),
                "trade_count": v["trade_count"]
            }
            for k, v in exchange_flows.items()
        }
        
        if len(exchange_flows) < 2:
            return result
        
        # Flow correlation: are exchanges moving in the same direction?
        deltas = [v["delta_pct"] for v in exchange_flows.values()]
        delta_signs = [1 if d > 5 else (-1 if d < -5 else 0) for d in deltas]
        
        # Count agreements
        positive = delta_signs.count(1)
        negative = delta_signs.count(-1)
        total = len(delta_signs)
        
        agreement_ratio = max(positive, negative) / total if total > 0 else 0
        result["flow_correlation"] = round(agreement_ratio, 2)
        
        # Volume synchronization: similar volume bursts?
        volumes = [v["total_volume"] for v in exchange_flows.values()]
        mean_vol = sum(volumes) / len(volumes)
        vol_std = math.sqrt(sum((v - mean_vol) ** 2 for v in volumes) / len(volumes))
        cv = vol_std / mean_vol if mean_vol > 0 else 0  # Coefficient of variation
        
        # Lower CV = more synchronized
        result["volume_sync_score"] = round(max(0, (1 - cv) * 100), 1)
        
        # Overall synchronization score
        result["synchronization_score"] = round(
            (result["flow_correlation"] * 60 + result["volume_sync_score"] * 0.4),
            1
        )
        
        # Coordinated flow detection
        if result["flow_correlation"] > 0.8 and result["synchronization_score"] > 70:
            result["coordinated_flow_detected"] = True
        
        # Dominant direction
        avg_delta = sum(deltas) / len(deltas)
        if avg_delta > 15:
            result["dominant_direction"] = "STRONG_BUYING"
        elif avg_delta > 5:
            result["dominant_direction"] = "BUYING"
        elif avg_delta < -15:
            result["dominant_direction"] = "STRONG_SELLING"
        elif avg_delta < -5:
            result["dominant_direction"] = "SELLING"
        else:
            result["dominant_direction"] = "NEUTRAL"
        
        return result
    
    def _update_series(
        self,
        symbol: str,
        prices: Dict[str, Dict],
        trades: Dict[str, List[Dict]]
    ):
        """Update historical price and trade series."""
        # Update price series
        if symbol not in self.price_series:
            self.price_series[symbol] = {}
        
        for exchange, data in prices.items():
            if exchange not in self.price_series[symbol]:
                self.price_series[symbol][exchange] = deque(maxlen=self.history_window)
            
            if isinstance(data, dict):
                price = data.get('price', 0)
            else:
                price = data
            
            if price > 0:
                self.price_series[symbol][exchange].append(price)
        
        # Update trade series
        if symbol not in self.trade_series:
            self.trade_series[symbol] = {}
        
        for exchange, trade_list in trades.items():
            if exchange not in self.trade_series[symbol]:
                self.trade_series[symbol][exchange] = deque(maxlen=self.history_window)
            
            if trade_list:
                buy_vol = sum(t.get('quantity', 0) for t in trade_list if t.get('side') == 'buy')
                sell_vol = sum(t.get('quantity', 0) for t in trade_list if t.get('side') == 'sell')
                self.trade_series[symbol][exchange].append({
                    "buy_volume": buy_vol,
                    "sell_volume": sell_vol,
                    "timestamp": datetime.utcnow()
                })
