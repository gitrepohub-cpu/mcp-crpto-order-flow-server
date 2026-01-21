"""
Leverage, Positioning & Risk Flows Analytics

Layer 2: Tells you where leverage is building, who is trapped, 
and where forced flows will appear.

Derived from: Open Interest, Liquidations, Trades, Funding, Mark Price
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque
import math

logger = logging.getLogger(__name__)


class LeverageAnalytics:
    """
    Leverage, Positioning & Risk Flows Engine.
    
    Features:
    2.1 Open Interest Flow Decomposition
    2.2 Leverage Build-Up Index
    2.3 Liquidation Pressure Mapping
    2.4 Funding Stress & Carry Pressure
    2.5 Basis Regime Classification
    """
    
    def __init__(self):
        # Historical tracking
        self.oi_history: Dict[str, deque] = {}
        self.price_history: Dict[str, deque] = {}
        self.funding_history: Dict[str, deque] = {}
        self.basis_history: Dict[str, deque] = {}
        self.liquidation_history: Dict[str, deque] = {}
        
        self.history_window = 300  # 5 minutes
        
    def compute_all_features(
        self,
        symbol: str,
        open_interest: Dict[str, Dict],
        liquidations: List[Dict],
        funding_rates: Dict[str, Dict],
        mark_prices: Dict[str, Dict],
        ticker_24h: Dict[str, Dict],
        current_price: float
    ) -> Dict:
        """
        Compute all leverage and positioning features.
        """
        result = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "oi_flow": self.compute_oi_flow_decomposition(symbol, open_interest, current_price),
            "leverage_index": self.compute_leverage_index(symbol, open_interest, ticker_24h),
            "liquidation_pressure": self.compute_liquidation_pressure(symbol, liquidations, current_price),
            "funding_stress": self.compute_funding_stress(symbol, funding_rates),
            "basis_regime": self.compute_basis_regime(symbol, mark_prices)
        }
        
        return result
    
    # =========================================================================
    # 2.1 Open Interest Flow Decomposition
    # =========================================================================
    
    def compute_oi_flow_decomposition(
        self,
        symbol: str,
        open_interest: Dict[str, Dict],
        current_price: float
    ) -> Dict:
        """
        Identify if positions are opening or closing.
        
        Logic:
        - Price ↑ + OI ↑ = Longs opening (bullish)
        - Price ↓ + OI ↑ = Shorts opening (bearish)
        - Price ↑ + OI ↓ = Shorts closing (bullish)
        - Price ↓ + OI ↓ = Longs closing (bearish)
        """
        result = {
            "total_oi": 0,
            "total_oi_value": 0,
            "oi_change": 0,
            "oi_change_pct": 0,
            "price_change": 0,
            "price_change_pct": 0,
            "position_intent": "UNKNOWN",
            "position_intent_score": 0,  # -100 to +100
            "exchanges": {},
            "interpretation": ""
        }
        
        # Aggregate OI
        total_oi = 0
        total_oi_value = 0
        
        for exchange, data in open_interest.items():
            oi = data.get('open_interest', 0)
            oi_value = data.get('open_interest_value', 0)
            
            result["exchanges"][exchange] = {
                "open_interest": oi,
                "open_interest_value": oi_value
            }
            
            total_oi += oi
            total_oi_value += oi_value
        
        result["total_oi"] = round(total_oi, 0)
        result["total_oi_value"] = round(total_oi_value, 2)
        
        # Initialize history
        if symbol not in self.oi_history:
            self.oi_history[symbol] = deque(maxlen=self.history_window)
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.history_window)
        
        # Compute changes if we have history
        if len(self.oi_history[symbol]) > 0 and len(self.price_history[symbol]) > 0:
            prev_oi = self.oi_history[symbol][-1]
            prev_price = self.price_history[symbol][-1]
            
            oi_change = total_oi - prev_oi
            oi_change_pct = (oi_change / prev_oi * 100) if prev_oi > 0 else 0
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price * 100) if prev_price > 0 else 0
            
            result["oi_change"] = round(oi_change, 0)
            result["oi_change_pct"] = round(oi_change_pct, 4)
            result["price_change"] = round(price_change, 2)
            result["price_change_pct"] = round(price_change_pct, 4)
            
            # Position intent analysis
            oi_increasing = oi_change_pct > 0.1
            oi_decreasing = oi_change_pct < -0.1
            price_up = price_change_pct > 0.05
            price_down = price_change_pct < -0.05
            
            if price_up and oi_increasing:
                result["position_intent"] = "LONGS_OPENING"
                result["position_intent_score"] = min(abs(oi_change_pct + price_change_pct) * 10, 100)
                result["interpretation"] = "New longs entering - bullish momentum building"
            elif price_down and oi_increasing:
                result["position_intent"] = "SHORTS_OPENING"
                result["position_intent_score"] = -min(abs(oi_change_pct + abs(price_change_pct)) * 10, 100)
                result["interpretation"] = "New shorts entering - bearish pressure building"
            elif price_up and oi_decreasing:
                result["position_intent"] = "SHORTS_CLOSING"
                result["position_intent_score"] = min(abs(price_change_pct) * 10, 80)
                result["interpretation"] = "Short squeeze / shorts covering - bullish"
            elif price_down and oi_decreasing:
                result["position_intent"] = "LONGS_CLOSING"
                result["position_intent_score"] = -min(abs(price_change_pct) * 10, 80)
                result["interpretation"] = "Long liquidation / longs exiting - bearish"
            else:
                result["position_intent"] = "CONSOLIDATION"
                result["position_intent_score"] = 0
                result["interpretation"] = "Mixed signals - market consolidating"
        
        # Update history
        self.oi_history[symbol].append(total_oi)
        self.price_history[symbol].append(current_price)
        
        return result
    
    # =========================================================================
    # 2.2 Leverage Build-Up Index
    # =========================================================================
    
    def compute_leverage_index(
        self,
        symbol: str,
        open_interest: Dict[str, Dict],
        ticker_24h: Dict[str, Dict]
    ) -> Dict:
        """
        Detect overcrowded positioning.
        
        Formula: Leverage Index = OI / 24h Volume
        
        High leverage index = liquidation cascade risk
        """
        result = {
            "leverage_index": 0,
            "leverage_zscore": 0,
            "leverage_velocity": 0,
            "risk_level": "NORMAL",
            "cascade_probability": 0,
            "exchanges": {}
        }
        
        # Compute per exchange
        leverage_indices = []
        
        for exchange in open_interest.keys():
            oi_data = open_interest.get(exchange, {})
            ticker_data = ticker_24h.get(exchange, {})
            
            oi = oi_data.get('open_interest', 0)
            volume_24h = ticker_data.get('volume', 0)
            
            if volume_24h > 0:
                leverage_idx = oi / volume_24h
                leverage_indices.append(leverage_idx)
                
                result["exchanges"][exchange] = {
                    "open_interest": oi,
                    "volume_24h": volume_24h,
                    "leverage_index": round(leverage_idx, 4)
                }
        
        # Aggregate
        if leverage_indices:
            avg_leverage = sum(leverage_indices) / len(leverage_indices)
            result["leverage_index"] = round(avg_leverage, 4)
            
            # Z-score (assuming mean of 1.0 and std of 0.5 for crypto)
            mean_leverage = 1.0
            std_leverage = 0.5
            result["leverage_zscore"] = round((avg_leverage - mean_leverage) / std_leverage, 2)
            
            # Risk classification
            if result["leverage_zscore"] > 2:
                result["risk_level"] = "EXTREME"
                result["cascade_probability"] = 90
            elif result["leverage_zscore"] > 1:
                result["risk_level"] = "HIGH"
                result["cascade_probability"] = 70
            elif result["leverage_zscore"] > 0:
                result["risk_level"] = "ELEVATED"
                result["cascade_probability"] = 45
            elif result["leverage_zscore"] > -1:
                result["risk_level"] = "NORMAL"
                result["cascade_probability"] = 20
            else:
                result["risk_level"] = "LOW"
                result["cascade_probability"] = 10
        
        return result
    
    # =========================================================================
    # 2.3 Liquidation Pressure Mapping
    # =========================================================================
    
    def compute_liquidation_pressure(
        self,
        symbol: str,
        liquidations: List[Dict],
        current_price: float
    ) -> Dict:
        """
        Identify forced flow zones.
        
        Features:
        - Liquidation Clusters (price bins)
        - Long vs Short Liquidation Dominance
        - Liquidation Momentum
        - Estimated Remaining Liquidation Risk
        """
        result = {
            "total_liquidations": 0,
            "total_value": 0,
            "long_liquidation_value": 0,
            "short_liquidation_value": 0,
            "long_short_ratio": 0,
            "liquidation_momentum": 0,
            "dominance": "NEUTRAL",
            "pressure_zones": [],
            "cascade_risk": "LOW",
            "estimated_remaining_at_risk": 0
        }
        
        if not liquidations:
            return result
        
        # Aggregate liquidations
        long_value = 0
        short_value = 0
        price_clusters = {}
        
        for liq in liquidations:
            value = liq.get('value', 0)
            side = liq.get('side', '').lower()
            price = liq.get('price', 0)
            
            result["total_value"] += value
            result["total_liquidations"] += 1
            
            if side in ('buy', 'long'):
                long_value += value
            elif side in ('sell', 'short'):
                short_value += value
            
            # Cluster by price bins (1% bins)
            if price > 0 and current_price > 0:
                bin_price = round(price / (current_price * 0.01)) * (current_price * 0.01)
                if bin_price not in price_clusters:
                    price_clusters[bin_price] = {"long": 0, "short": 0, "count": 0}
                
                if side in ('buy', 'long'):
                    price_clusters[bin_price]["long"] += value
                else:
                    price_clusters[bin_price]["short"] += value
                price_clusters[bin_price]["count"] += 1
        
        result["long_liquidation_value"] = round(long_value, 2)
        result["short_liquidation_value"] = round(short_value, 2)
        result["total_value"] = round(long_value + short_value, 2)
        
        # Long/short ratio
        if short_value > 0:
            result["long_short_ratio"] = round(long_value / short_value, 2)
        elif long_value > 0:
            result["long_short_ratio"] = float('inf')
        
        # Dominance
        total = long_value + short_value
        if total > 0:
            if long_value / total > 0.7:
                result["dominance"] = "LONG_LIQUIDATION_DOMINANT"
            elif short_value / total > 0.7:
                result["dominance"] = "SHORT_LIQUIDATION_DOMINANT"
            else:
                result["dominance"] = "BALANCED"
        
        # Pressure zones
        for price, data in sorted(price_clusters.items()):
            zone_total = data["long"] + data["short"]
            if zone_total > 0:
                pct_from_current = ((price - current_price) / current_price) * 100
                result["pressure_zones"].append({
                    "price": round(price, 2),
                    "pct_from_current": round(pct_from_current, 2),
                    "long_value": round(data["long"], 2),
                    "short_value": round(data["short"], 2),
                    "total_value": round(zone_total, 2),
                    "count": data["count"]
                })
        
        # Sort by total value
        result["pressure_zones"] = sorted(
            result["pressure_zones"],
            key=lambda x: x["total_value"],
            reverse=True
        )[:10]
        
        # Cascade risk assessment
        if result["total_value"] > 10000000:  # > $10M
            result["cascade_risk"] = "EXTREME"
        elif result["total_value"] > 1000000:  # > $1M
            result["cascade_risk"] = "HIGH"
        elif result["total_value"] > 100000:  # > $100K
            result["cascade_risk"] = "MODERATE"
        else:
            result["cascade_risk"] = "LOW"
        
        # Estimated remaining at risk (rough estimate based on leverage)
        result["estimated_remaining_at_risk"] = round(result["total_value"] * 2.5, 2)
        
        # Update history for momentum
        if symbol not in self.liquidation_history:
            self.liquidation_history[symbol] = deque(maxlen=60)
        
        if len(self.liquidation_history[symbol]) > 0:
            prev_value = self.liquidation_history[symbol][-1]
            result["liquidation_momentum"] = round(result["total_value"] - prev_value, 2)
        
        self.liquidation_history[symbol].append(result["total_value"])
        
        return result
    
    # =========================================================================
    # 2.4 Funding Stress & Carry Pressure
    # =========================================================================
    
    def compute_funding_stress(
        self,
        symbol: str,
        funding_rates: Dict[str, Dict]
    ) -> Dict:
        """
        Detect sentiment extremes via funding rates.
        
        Features:
        - Funding Z-Score
        - Funding Momentum
        - Funding Divergence vs Price
        - Carry Crowding Index
        """
        result = {
            "avg_funding_rate": 0,
            "avg_annualized": 0,
            "funding_zscore": 0,
            "funding_momentum": 0,
            "stress_level": "NORMAL",
            "sentiment": "NEUTRAL",
            "carry_crowding": 0,
            "exchanges": {}
        }
        
        if not funding_rates:
            return result
        
        rates = []
        annualized_rates = []
        
        for exchange, data in funding_rates.items():
            rate_pct = data.get('rate_pct', 0)
            annualized = data.get('annualized_rate', 0)
            
            rates.append(rate_pct)
            annualized_rates.append(annualized)
            
            # Sentiment per exchange
            if rate_pct > 0.05:
                sentiment = "VERY_BULLISH"
            elif rate_pct > 0.01:
                sentiment = "BULLISH"
            elif rate_pct < -0.05:
                sentiment = "VERY_BEARISH"
            elif rate_pct < -0.01:
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"
            
            result["exchanges"][exchange] = {
                "rate_pct": rate_pct,
                "annualized": annualized,
                "sentiment": sentiment
            }
        
        if rates:
            avg_rate = sum(rates) / len(rates)
            avg_annual = sum(annualized_rates) / len(annualized_rates)
            
            result["avg_funding_rate"] = round(avg_rate, 6)
            result["avg_annualized"] = round(avg_annual, 2)
            
            # Z-score (assuming neutral is 0.01%, std is 0.03%)
            mean_funding = 0.01
            std_funding = 0.03
            result["funding_zscore"] = round((avg_rate - mean_funding) / std_funding, 2)
            
            # Stress level
            zscore = abs(result["funding_zscore"])
            if zscore > 3:
                result["stress_level"] = "EXTREME"
            elif zscore > 2:
                result["stress_level"] = "HIGH"
            elif zscore > 1:
                result["stress_level"] = "ELEVATED"
            else:
                result["stress_level"] = "NORMAL"
            
            # Overall sentiment
            if result["funding_zscore"] > 2:
                result["sentiment"] = "EXTREME_BULLISH_CROWDING"
            elif result["funding_zscore"] > 1:
                result["sentiment"] = "BULLISH_CROWDING"
            elif result["funding_zscore"] < -2:
                result["sentiment"] = "EXTREME_BEARISH_CROWDING"
            elif result["funding_zscore"] < -1:
                result["sentiment"] = "BEARISH_CROWDING"
            else:
                result["sentiment"] = "NEUTRAL"
            
            # Carry crowding (how attractive is the carry trade)
            result["carry_crowding"] = round(abs(avg_annual), 2)
            
            # Momentum
            if symbol not in self.funding_history:
                self.funding_history[symbol] = deque(maxlen=self.history_window)
            
            if len(self.funding_history[symbol]) > 0:
                prev_rate = self.funding_history[symbol][-1]
                result["funding_momentum"] = round(avg_rate - prev_rate, 6)
            
            self.funding_history[symbol].append(avg_rate)
        
        return result
    
    # =========================================================================
    # 2.5 Basis Regime Classification
    # =========================================================================
    
    def compute_basis_regime(
        self,
        symbol: str,
        mark_prices: Dict[str, Dict]
    ) -> Dict:
        """
        Understand futures curve health.
        
        Features:
        - Basis Trend
        - Basis Volatility
        - Mean Reversion Speed
        - Basis-Price Divergence
        
        Regimes:
        - Healthy contango
        - Overheated contango
        - Backwardation stress
        - Funding arbitrage regime
        """
        result = {
            "avg_basis_pct": 0,
            "basis_trend": "STABLE",
            "basis_volatility": 0,
            "regime": "UNKNOWN",
            "regime_strength": 0,
            "arbitrage_opportunity": False,
            "exchanges": {}
        }
        
        if not mark_prices:
            return result
        
        basis_values = []
        
        for exchange, data in mark_prices.items():
            mark = data.get('mark_price', 0)
            index = data.get('index_price', 0)
            basis_pct = data.get('basis_pct', 0)
            
            if basis_pct != 0:
                basis_values.append(basis_pct)
            elif mark > 0 and index > 0:
                basis_pct = ((mark - index) / index) * 100
                basis_values.append(basis_pct)
            
            result["exchanges"][exchange] = {
                "mark_price": mark,
                "index_price": index,
                "basis_pct": round(basis_pct, 4)
            }
        
        if basis_values:
            avg_basis = sum(basis_values) / len(basis_values)
            result["avg_basis_pct"] = round(avg_basis, 4)
            
            # Basis volatility
            if len(basis_values) > 1:
                mean = avg_basis
                variance = sum((b - mean) ** 2 for b in basis_values) / len(basis_values)
                result["basis_volatility"] = round(math.sqrt(variance), 4)
            
            # Regime classification
            if avg_basis > 0.1:
                result["regime"] = "OVERHEATED_CONTANGO"
                result["regime_strength"] = min(avg_basis * 100, 100)
            elif avg_basis > 0.02:
                result["regime"] = "HEALTHY_CONTANGO"
                result["regime_strength"] = avg_basis * 200
            elif avg_basis < -0.1:
                result["regime"] = "BACKWARDATION_STRESS"
                result["regime_strength"] = min(abs(avg_basis) * 100, 100)
            elif avg_basis < -0.02:
                result["regime"] = "MILD_BACKWARDATION"
                result["regime_strength"] = abs(avg_basis) * 200
            else:
                result["regime"] = "NEUTRAL_BASIS"
                result["regime_strength"] = 50
            
            # Arbitrage opportunity detection
            if len(basis_values) > 1:
                basis_spread = max(basis_values) - min(basis_values)
                if basis_spread > 0.1:  # > 0.1% spread between exchanges
                    result["arbitrage_opportunity"] = True
            
            # Basis trend from history
            if symbol not in self.basis_history:
                self.basis_history[symbol] = deque(maxlen=self.history_window)
            
            if len(self.basis_history[symbol]) >= 5:
                recent = list(self.basis_history[symbol])[-5:]
                if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                    result["basis_trend"] = "EXPANDING"
                elif all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                    result["basis_trend"] = "CONTRACTING"
                else:
                    result["basis_trend"] = "STABLE"
            
            self.basis_history[symbol].append(avg_basis)
        
        return result
