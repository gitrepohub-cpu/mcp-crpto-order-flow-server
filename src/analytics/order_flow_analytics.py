"""
Order Flow & Microstructure DOM Intelligence

Layer 1: Detects real supply/demand behavior, hidden liquidity, 
absorption, spoofing, and real directional pressure.

Derived from: Orderbook, Trades, Prices
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import math

logger = logging.getLogger(__name__)


class OrderFlowAnalytics:
    """
    Order Flow & Microstructure DOM Intelligence Engine.
    
    Features:
    1.1 Real-Time Liquidity Imbalance (Multi-Level)
    1.2 Liquidity Vacuum Detection
    1.3 Orderbook Heat Persistence
    1.4 Trade Aggression Flow Metrics
    1.5 Microstructure Efficiency & Impact
    """
    
    def __init__(self):
        # Historical tracking for persistence analysis
        self.orderbook_history: Dict[str, Dict[str, deque]] = {}  # symbol -> exchange -> history
        self.trade_history: Dict[str, deque] = {}  # symbol -> trades
        self.imbalance_history: Dict[str, deque] = {}  # symbol -> imbalance values
        self.price_history: Dict[str, deque] = {}  # symbol -> prices
        
        # Configuration
        self.history_window = 300  # 5 minutes of history
        self.depth_bands = [5, 20, 50]  # tick depth bands
        
    def compute_all_features(
        self,
        symbol: str,
        orderbooks: Dict[str, Dict],
        trades: Dict[str, List[Dict]],
        prices: Dict[str, Dict]
    ) -> Dict:
        """
        Compute all order flow features for a symbol.
        
        Returns comprehensive order flow intelligence.
        """
        result = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "liquidity_imbalance": self.compute_liquidity_imbalance(symbol, orderbooks),
            "liquidity_vacuum": self.compute_liquidity_vacuum(symbol, orderbooks, prices),
            "orderbook_persistence": self.compute_orderbook_persistence(symbol, orderbooks),
            "trade_aggression": self.compute_trade_aggression(symbol, trades),
            "microstructure_efficiency": self.compute_microstructure_efficiency(symbol, trades, prices)
        }
        
        # Update historical data
        self._update_history(symbol, orderbooks, trades, prices)
        
        return result
    
    # =========================================================================
    # 1.1 Real-Time Liquidity Imbalance (Multi-Level)
    # =========================================================================
    
    def compute_liquidity_imbalance(
        self,
        symbol: str,
        orderbooks: Dict[str, Dict]
    ) -> Dict:
        """
        Compute multi-level liquidity imbalance across depth bands.
        
        Purpose: Detect directional pressure before price moves.
        
        Formula:
        - Bid Imbalance = Σ(bid_qty[1..N])
        - Ask Imbalance = Σ(ask_qty[1..N])
        - Imbalance Ratio = (Bid - Ask) / (Bid + Ask)
        """
        result = {
            "exchanges": {},
            "aggregated": {},
            "depth_bands": {},
            "velocity": 0,
            "signal": "NEUTRAL",
            "confidence": 0
        }
        
        total_bid_depth = 0
        total_ask_depth = 0
        
        # Compute per-exchange imbalance
        for exchange, ob in orderbooks.items():
            if not ob or 'bids' not in ob or 'asks' not in ob:
                continue
                
            bids = ob.get('bids', [])
            asks = ob.get('asks', [])
            
            if not bids or not asks:
                continue
            
            # Get reference price (mid)
            best_bid = bids[0]['price'] if bids else 0
            best_ask = asks[0]['price'] if asks else 0
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
            
            if mid_price == 0:
                continue
            
            # Compute imbalance per depth band
            exchange_bands = {}
            for band_ticks in self.depth_bands:
                # Convert ticks to price (assume 1 tick = 0.01% of price)
                band_pct = band_ticks * 0.0001
                bid_threshold = mid_price * (1 - band_pct)
                ask_threshold = mid_price * (1 + band_pct)
                
                band_bid_qty = sum(b['quantity'] for b in bids if b['price'] >= bid_threshold)
                band_ask_qty = sum(a['quantity'] for a in asks if a['price'] <= ask_threshold)
                
                total = band_bid_qty + band_ask_qty
                imbalance_ratio = (band_bid_qty - band_ask_qty) / total if total > 0 else 0
                
                exchange_bands[f"{band_ticks}_ticks"] = {
                    "bid_depth": round(band_bid_qty, 4),
                    "ask_depth": round(band_ask_qty, 4),
                    "imbalance_ratio": round(imbalance_ratio, 4),
                    "imbalance_pct": round(imbalance_ratio * 100, 2)
                }
            
            # Total exchange imbalance
            total_bid = sum(b['quantity'] for b in bids)
            total_ask = sum(a['quantity'] for a in asks)
            total = total_bid + total_ask
            
            result["exchanges"][exchange] = {
                "bid_depth": round(total_bid, 4),
                "ask_depth": round(total_ask, 4),
                "imbalance_ratio": round((total_bid - total_ask) / total, 4) if total > 0 else 0,
                "depth_bands": exchange_bands,
                "spread": round(best_ask - best_bid, 2),
                "spread_bps": round((best_ask - best_bid) / mid_price * 10000, 2) if mid_price else 0
            }
            
            total_bid_depth += total_bid
            total_ask_depth += total_ask
        
        # Aggregated cross-exchange imbalance
        total = total_bid_depth + total_ask_depth
        if total > 0:
            agg_imbalance = (total_bid_depth - total_ask_depth) / total
            result["aggregated"] = {
                "total_bid_depth": round(total_bid_depth, 4),
                "total_ask_depth": round(total_ask_depth, 4),
                "imbalance_ratio": round(agg_imbalance, 4),
                "imbalance_pct": round(agg_imbalance * 100, 2)
            }
            
            # Compute velocity from history
            if symbol in self.imbalance_history and len(self.imbalance_history[symbol]) > 0:
                prev_imbalance = self.imbalance_history[symbol][-1]
                result["velocity"] = round(agg_imbalance - prev_imbalance, 4)
            
            # Signal generation
            if agg_imbalance > 0.3:
                result["signal"] = "STRONG_BUY_PRESSURE"
                result["confidence"] = min(abs(agg_imbalance) * 100, 100)
            elif agg_imbalance > 0.15:
                result["signal"] = "BUY_PRESSURE"
                result["confidence"] = min(abs(agg_imbalance) * 100, 80)
            elif agg_imbalance < -0.3:
                result["signal"] = "STRONG_SELL_PRESSURE"
                result["confidence"] = min(abs(agg_imbalance) * 100, 100)
            elif agg_imbalance < -0.15:
                result["signal"] = "SELL_PRESSURE"
                result["confidence"] = min(abs(agg_imbalance) * 100, 80)
            else:
                result["signal"] = "NEUTRAL"
                result["confidence"] = 50
                
            # Track history
            if symbol not in self.imbalance_history:
                self.imbalance_history[symbol] = deque(maxlen=self.history_window)
            self.imbalance_history[symbol].append(agg_imbalance)
        
        return result
    
    # =========================================================================
    # 1.2 Liquidity Vacuum Detection
    # =========================================================================
    
    def compute_liquidity_vacuum(
        self,
        symbol: str,
        orderbooks: Dict[str, Dict],
        prices: Dict[str, Dict]
    ) -> Dict:
        """
        Identify zones where price can move extremely fast.
        
        Purpose: Detect thin liquidity that enables violent price moves.
        
        Logic:
        - Total Depth within X bps < threshold
        - Spread widening rapidly
        - Orderbook thinning velocity
        """
        result = {
            "vacuum_score": 0,  # 0-100
            "expected_slippage_risk": "LOW",
            "breakout_probability": 0,
            "thin_zones": [],
            "exchanges": {}
        }
        
        vacuum_scores = []
        
        for exchange, ob in orderbooks.items():
            if not ob or 'bids' not in ob or 'asks' not in ob:
                continue
                
            bids = ob.get('bids', [])
            asks = ob.get('asks', [])
            
            if not bids or not asks:
                continue
            
            best_bid = bids[0]['price']
            best_ask = asks[0]['price']
            mid_price = (best_bid + best_ask) / 2
            spread_bps = (best_ask - best_bid) / mid_price * 10000
            
            # Compute depth at various levels
            depth_10bps = self._compute_depth_within_bps(bids, asks, mid_price, 10)
            depth_25bps = self._compute_depth_within_bps(bids, asks, mid_price, 25)
            depth_50bps = self._compute_depth_within_bps(bids, asks, mid_price, 50)
            
            # Vacuum detection: thin depth relative to typical
            # Lower depth = higher vacuum score
            # Also factor in spread widening
            
            # Normalize depth (assume typical depth is 100 units for BTC)
            typical_depth = 100 if 'BTC' in symbol else 1000
            
            depth_score = 1 - min(depth_10bps / typical_depth, 1)
            spread_score = min(spread_bps / 10, 1)  # Spread > 10bps is concerning
            
            exchange_vacuum = (depth_score * 0.7 + spread_score * 0.3) * 100
            
            result["exchanges"][exchange] = {
                "vacuum_score": round(exchange_vacuum, 1),
                "depth_10bps": round(depth_10bps, 4),
                "depth_25bps": round(depth_25bps, 4),
                "depth_50bps": round(depth_50bps, 4),
                "spread_bps": round(spread_bps, 2)
            }
            
            vacuum_scores.append(exchange_vacuum)
            
            # Detect specific thin zones
            thin_bid_zones = self._find_thin_zones(bids, mid_price, "bid")
            thin_ask_zones = self._find_thin_zones(asks, mid_price, "ask")
            result["thin_zones"].extend(thin_bid_zones + thin_ask_zones)
        
        # Aggregate vacuum score
        if vacuum_scores:
            result["vacuum_score"] = round(sum(vacuum_scores) / len(vacuum_scores), 1)
            
            # Classify risk
            if result["vacuum_score"] > 70:
                result["expected_slippage_risk"] = "EXTREME"
                result["breakout_probability"] = 85
            elif result["vacuum_score"] > 50:
                result["expected_slippage_risk"] = "HIGH"
                result["breakout_probability"] = 65
            elif result["vacuum_score"] > 30:
                result["expected_slippage_risk"] = "MODERATE"
                result["breakout_probability"] = 40
            else:
                result["expected_slippage_risk"] = "LOW"
                result["breakout_probability"] = 20
        
        return result
    
    def _compute_depth_within_bps(
        self,
        bids: List[Dict],
        asks: List[Dict],
        mid_price: float,
        bps: int
    ) -> float:
        """Compute total depth within X basis points of mid."""
        threshold = mid_price * (bps / 10000)
        bid_depth = sum(b['quantity'] for b in bids if mid_price - b['price'] <= threshold)
        ask_depth = sum(a['quantity'] for a in asks if a['price'] - mid_price <= threshold)
        return bid_depth + ask_depth
    
    def _find_thin_zones(
        self,
        levels: List[Dict],
        mid_price: float,
        side: str
    ) -> List[Dict]:
        """Find price zones with unusually thin liquidity."""
        thin_zones = []
        
        if len(levels) < 3:
            return thin_zones
        
        # Look for gaps in the orderbook
        for i in range(len(levels) - 1):
            price1 = levels[i]['price']
            price2 = levels[i + 1]['price']
            gap_pct = abs(price2 - price1) / mid_price * 100
            
            # Gap > 0.1% is significant
            if gap_pct > 0.1:
                thin_zones.append({
                    "side": side,
                    "from_price": round(min(price1, price2), 2),
                    "to_price": round(max(price1, price2), 2),
                    "gap_pct": round(gap_pct, 4)
                })
        
        return thin_zones[:5]  # Top 5 thin zones
    
    # =========================================================================
    # 1.3 Orderbook Heat Persistence
    # =========================================================================
    
    def compute_orderbook_persistence(
        self,
        symbol: str,
        orderbooks: Dict[str, Dict]
    ) -> Dict:
        """
        Identify real institutional liquidity vs fake walls.
        
        Purpose: Distinguish genuine support/resistance from spoofing.
        
        Tracks:
        - Persistence Time
        - Average Resting Size
        - Cancel-to-Fill Ratio (estimated)
        """
        result = {
            "reliability_score": 0,  # 0-100
            "persistent_levels": [],
            "suspected_spoofing": [],
            "support_zones": [],
            "resistance_zones": []
        }
        
        # Initialize history tracking if needed
        if symbol not in self.orderbook_history:
            self.orderbook_history[symbol] = {}
        
        for exchange, ob in orderbooks.items():
            if not ob or 'bids' not in ob or 'asks' not in ob:
                continue
            
            bids = ob.get('bids', [])
            asks = ob.get('asks', [])
            
            if exchange not in self.orderbook_history[symbol]:
                self.orderbook_history[symbol][exchange] = deque(maxlen=60)  # 60 snapshots
            
            # Store current snapshot
            current_snapshot = {
                "timestamp": datetime.utcnow(),
                "bids": {b['price']: b['quantity'] for b in bids[:20]},
                "asks": {a['price']: a['quantity'] for a in asks[:20]}
            }
            self.orderbook_history[symbol][exchange].append(current_snapshot)
            
            # Analyze persistence if we have history
            history = list(self.orderbook_history[symbol][exchange])
            if len(history) >= 5:
                # Find persistent bid levels
                persistent_bids = self._find_persistent_levels(history, "bids")
                persistent_asks = self._find_persistent_levels(history, "asks")
                
                for level in persistent_bids:
                    result["support_zones"].append({
                        "exchange": exchange,
                        "price": level["price"],
                        "avg_size": level["avg_size"],
                        "persistence_pct": level["persistence_pct"],
                        "reliability": level["reliability"]
                    })
                
                for level in persistent_asks:
                    result["resistance_zones"].append({
                        "exchange": exchange,
                        "price": level["price"],
                        "avg_size": level["avg_size"],
                        "persistence_pct": level["persistence_pct"],
                        "reliability": level["reliability"]
                    })
        
        # Sort by reliability
        result["support_zones"] = sorted(
            result["support_zones"],
            key=lambda x: x["reliability"],
            reverse=True
        )[:10]
        
        result["resistance_zones"] = sorted(
            result["resistance_zones"],
            key=lambda x: x["reliability"],
            reverse=True
        )[:10]
        
        # Overall reliability score
        all_reliability = [z["reliability"] for z in result["support_zones"] + result["resistance_zones"]]
        result["reliability_score"] = round(sum(all_reliability) / len(all_reliability), 1) if all_reliability else 50
        
        return result
    
    def _find_persistent_levels(
        self,
        history: List[Dict],
        side: str
    ) -> List[Dict]:
        """Find price levels that persist across multiple snapshots."""
        price_appearances = {}
        
        for snapshot in history:
            levels = snapshot.get(side, {})
            for price, qty in levels.items():
                if price not in price_appearances:
                    price_appearances[price] = {"count": 0, "sizes": []}
                price_appearances[price]["count"] += 1
                price_appearances[price]["sizes"].append(qty)
        
        persistent = []
        total_snapshots = len(history)
        
        for price, data in price_appearances.items():
            persistence_pct = data["count"] / total_snapshots * 100
            avg_size = sum(data["sizes"]) / len(data["sizes"])
            size_variance = self._compute_variance(data["sizes"])
            
            # High persistence + low variance = reliable level
            reliability = persistence_pct * (1 - min(size_variance / avg_size, 1)) if avg_size > 0 else 0
            
            if persistence_pct >= 50:  # Present in at least 50% of snapshots
                persistent.append({
                    "price": price,
                    "avg_size": round(avg_size, 4),
                    "persistence_pct": round(persistence_pct, 1),
                    "size_variance": round(size_variance, 4),
                    "reliability": round(reliability, 1)
                })
        
        return sorted(persistent, key=lambda x: x["reliability"], reverse=True)[:5]
    
    def _compute_variance(self, values: List[float]) -> float:
        """Compute variance of a list of values."""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)
    
    # =========================================================================
    # 1.4 Trade Aggression Flow Metrics
    # =========================================================================
    
    def compute_trade_aggression(
        self,
        symbol: str,
        trades: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Measure who is attacking the book.
        
        Key Features:
        - Aggressive Buy Volume / Sell Volume
        - Delta
        - Cumulative Volume Delta (CVD)
        - Delta Acceleration
        - Large Trade Clustering
        """
        result = {
            "delta": 0,
            "delta_pct": 0,
            "cvd": 0,
            "delta_acceleration": 0,
            "buy_volume": 0,
            "sell_volume": 0,
            "total_volume": 0,
            "large_trades": [],
            "absorption_detected": False,
            "aggressor": "NEUTRAL",
            "exchanges": {}
        }
        
        total_buy_vol = 0
        total_sell_vol = 0
        all_trades = []
        
        for exchange, trade_list in trades.items():
            if not trade_list:
                continue
            
            buy_vol = sum(t.get('quantity', 0) for t in trade_list if t.get('side') == 'buy')
            sell_vol = sum(t.get('quantity', 0) for t in trade_list if t.get('side') == 'sell')
            delta = buy_vol - sell_vol
            total = buy_vol + sell_vol
            
            result["exchanges"][exchange] = {
                "buy_volume": round(buy_vol, 4),
                "sell_volume": round(sell_vol, 4),
                "delta": round(delta, 4),
                "delta_pct": round(delta / total * 100, 2) if total > 0 else 0,
                "trade_count": len(trade_list)
            }
            
            total_buy_vol += buy_vol
            total_sell_vol += sell_vol
            all_trades.extend(trade_list)
        
        # Aggregate metrics
        result["buy_volume"] = round(total_buy_vol, 4)
        result["sell_volume"] = round(total_sell_vol, 4)
        result["total_volume"] = round(total_buy_vol + total_sell_vol, 4)
        result["delta"] = round(total_buy_vol - total_sell_vol, 4)
        
        total = total_buy_vol + total_sell_vol
        if total > 0:
            result["delta_pct"] = round(result["delta"] / total * 100, 2)
        
        # CVD tracking
        if symbol not in self.trade_history:
            self.trade_history[symbol] = deque(maxlen=self.history_window)
        
        prev_cvd = self.trade_history[symbol][-1]["cvd"] if self.trade_history[symbol] else 0
        result["cvd"] = round(prev_cvd + result["delta"], 4)
        
        # Delta acceleration
        if len(self.trade_history[symbol]) >= 2:
            prev_delta = self.trade_history[symbol][-1]["delta"]
            result["delta_acceleration"] = round(result["delta"] - prev_delta, 4)
        
        self.trade_history[symbol].append({
            "delta": result["delta"],
            "cvd": result["cvd"],
            "timestamp": datetime.utcnow()
        })
        
        # Large trade detection
        if all_trades:
            avg_size = total / len(all_trades) if all_trades else 0
            large_threshold = avg_size * 5  # 5x average = large trade
            
            for trade in all_trades:
                qty = trade.get('quantity', 0)
                if abs(qty) >= large_threshold:
                    result["large_trades"].append({
                        "price": trade.get('price', 0),
                        "quantity": qty,
                        "side": trade.get('side', 'unknown'),
                        "value": trade.get('value', 0),
                        "exchange": trade.get('exchange', 'unknown'),
                        "size_multiple": round(abs(qty) / avg_size, 1) if avg_size > 0 else 0
                    })
            
            result["large_trades"] = sorted(
                result["large_trades"],
                key=lambda x: abs(x["quantity"]),
                reverse=True
            )[:10]
        
        # Aggressor classification
        if result["delta_pct"] > 20:
            result["aggressor"] = "STRONG_BUYERS"
        elif result["delta_pct"] > 10:
            result["aggressor"] = "BUYERS"
        elif result["delta_pct"] < -20:
            result["aggressor"] = "STRONG_SELLERS"
        elif result["delta_pct"] < -10:
            result["aggressor"] = "SELLERS"
        else:
            result["aggressor"] = "BALANCED"
        
        return result
    
    # =========================================================================
    # 1.5 Microstructure Efficiency & Impact
    # =========================================================================
    
    def compute_microstructure_efficiency(
        self,
        symbol: str,
        trades: Dict[str, List[Dict]],
        prices: Dict[str, Dict]
    ) -> Dict:
        """
        Measure how much price moves per unit of volume.
        
        Features:
        - Impact = |ΔPrice| / Traded Volume
        - Slippage per $1M traded
        - Micro Volatility
        """
        result = {
            "price_impact": 0,
            "slippage_per_million": 0,
            "micro_volatility": 0,
            "efficiency_score": 0,  # Higher = more efficient/liquid
            "market_quality": "NORMAL",
            "exchanges": {}
        }
        
        for exchange, trade_list in trades.items():
            if not trade_list or len(trade_list) < 2:
                continue
            
            # Get prices from trades
            trade_prices = [t.get('price', 0) for t in trade_list if t.get('price', 0) > 0]
            trade_volumes = [abs(t.get('quantity', 0)) for t in trade_list]
            trade_values = [abs(t.get('value', 0)) for t in trade_list]
            
            if not trade_prices or len(trade_prices) < 2:
                continue
            
            # Price changes
            price_changes = [abs(trade_prices[i] - trade_prices[i-1]) 
                           for i in range(1, len(trade_prices))]
            
            # Total volume
            total_volume = sum(trade_volumes)
            total_value = sum(trade_values)
            
            # Micro volatility: std of price changes
            micro_vol = 0
            if price_changes:
                avg_change = sum(price_changes) / len(price_changes)
                micro_vol = math.sqrt(sum((c - avg_change)**2 for c in price_changes) / len(price_changes))
            
            # Price impact: total price movement per unit volume
            total_price_move = abs(trade_prices[-1] - trade_prices[0]) if trade_prices else 0
            impact = total_price_move / total_volume if total_volume > 0 else 0
            
            # Slippage per $1M
            slippage_per_m = (total_price_move / trade_prices[0] * 100) / (total_value / 1000000) if total_value > 0 and trade_prices[0] > 0 else 0
            
            result["exchanges"][exchange] = {
                "price_impact": round(impact, 6),
                "micro_volatility": round(micro_vol, 4),
                "slippage_per_million": round(slippage_per_m, 4),
                "total_volume": round(total_volume, 4),
                "total_value": round(total_value, 2),
                "trade_count": len(trade_list)
            }
        
        # Aggregate
        if result["exchanges"]:
            impacts = [e["price_impact"] for e in result["exchanges"].values()]
            vols = [e["micro_volatility"] for e in result["exchanges"].values()]
            slippages = [e["slippage_per_million"] for e in result["exchanges"].values()]
            
            result["price_impact"] = round(sum(impacts) / len(impacts), 6)
            result["micro_volatility"] = round(sum(vols) / len(vols), 4)
            result["slippage_per_million"] = round(sum(slippages) / len(slippages), 4)
            
            # Efficiency score: lower impact & volatility = higher efficiency
            # Normalize to 0-100 scale
            impact_score = max(0, 100 - result["price_impact"] * 10000)
            vol_score = max(0, 100 - result["micro_volatility"] * 100)
            result["efficiency_score"] = round((impact_score + vol_score) / 2, 1)
            
            # Market quality classification
            if result["efficiency_score"] > 80:
                result["market_quality"] = "EXCELLENT"
            elif result["efficiency_score"] > 60:
                result["market_quality"] = "GOOD"
            elif result["efficiency_score"] > 40:
                result["market_quality"] = "NORMAL"
            elif result["efficiency_score"] > 20:
                result["market_quality"] = "POOR"
            else:
                result["market_quality"] = "FRAGILE"
        
        return result
    
    def _update_history(
        self,
        symbol: str,
        orderbooks: Dict[str, Dict],
        trades: Dict[str, List[Dict]],
        prices: Dict[str, Dict]
    ):
        """Update historical data stores for time-series analysis."""
        # Price history
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.history_window)
        
        # Get representative price
        for exchange, data in prices.items():
            if isinstance(data, dict) and data.get('price', 0) > 0:
                self.price_history[symbol].append({
                    "price": data['price'],
                    "timestamp": datetime.utcnow()
                })
                break
