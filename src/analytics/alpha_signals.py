"""
Alpha Signal Engine (Composite Intelligence)

Layer 5: Combines all features into unified intelligence signals.

This is the final layer that produces actionable trading signals.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class AlphaSignalEngine:
    """
    Alpha Signal Engine - Composite Intelligence Layer.
    
    Features:
    5.1 Institutional Pressure Score
    5.2 Squeeze Probability Model
    5.3 Smart Money Absorption Detector
    """
    
    def __init__(self):
        # Signal history for trend analysis
        self.pressure_history: Dict[str, deque] = {}
        self.squeeze_history: Dict[str, deque] = {}
        self.absorption_history: Dict[str, deque] = {}
        
        self.history_window = 300
        
    def compute_all_signals(
        self,
        symbol: str,
        order_flow: Dict,
        leverage: Dict,
        cross_exchange: Dict,
        regime: Dict
    ) -> Dict:
        """
        Compute all alpha signals.
        
        Takes output from all other analytics layers.
        """
        result = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "institutional_pressure": self.compute_institutional_pressure(
                symbol, order_flow, leverage, cross_exchange
            ),
            "squeeze_probability": self.compute_squeeze_probability(
                symbol, order_flow, leverage, regime
            ),
            "smart_money_absorption": self.detect_smart_money_absorption(
                symbol, order_flow, leverage
            ),
            "composite_signal": {}
        }
        
        # Generate final composite signal
        result["composite_signal"] = self._generate_composite_signal(
            result["institutional_pressure"],
            result["squeeze_probability"],
            result["smart_money_absorption"],
            regime
        )
        
        return result
    
    # =========================================================================
    # 5.1 Institutional Pressure Score
    # =========================================================================
    
    def compute_institutional_pressure(
        self,
        symbol: str,
        order_flow: Dict,
        leverage: Dict,
        cross_exchange: Dict
    ) -> Dict:
        """
        Compute institutional buying/selling pressure.
        
        Inputs:
        - Orderbook imbalance
        - Trade delta
        - OI flow
        - Funding skew
        - Basis trend
        
        Output:
        - Bullish / Neutral / Bearish Pressure (0-100)
        """
        result = {
            "pressure_score": 0,  # -100 (bearish) to +100 (bullish)
            "pressure_direction": "NEUTRAL",
            "pressure_strength": "WEAK",
            "confidence": 0,
            "components": {},
            "trend": "STABLE",
            "recommendation": ""
        }
        
        # Extract components
        components = {}
        
        # 1. Orderbook imbalance (-100 to +100)
        imbalance = order_flow.get("liquidity_imbalance", {})
        agg = imbalance.get("aggregated", {})
        imbalance_ratio = agg.get("imbalance_ratio", 0)
        components["orderbook_imbalance"] = round(imbalance_ratio * 100, 2)
        
        # 2. Trade delta (-100 to +100)
        trade_agg = order_flow.get("trade_aggression", {})
        delta_pct = trade_agg.get("delta_pct", 0)
        components["trade_delta"] = round(min(max(delta_pct, -100), 100), 2)
        
        # 3. OI flow (-100 to +100)
        oi_flow = leverage.get("oi_flow", {})
        intent_score = oi_flow.get("position_intent_score", 0)
        components["oi_flow"] = round(intent_score, 2)
        
        # 4. Funding skew (-100 to +100)
        funding = leverage.get("funding_stress", {})
        funding_zscore = funding.get("funding_zscore", 0)
        # Positive funding = bullish crowding (contrarian bearish)
        # So we invert: high positive = bearish signal
        components["funding_skew"] = round(-funding_zscore * 20, 2)
        
        # 5. Basis trend (-100 to +100)
        basis = leverage.get("basis_regime", {})
        basis_pct = basis.get("avg_basis_pct", 0)
        components["basis_trend"] = round(basis_pct * 50, 2)
        
        # 6. Flow synchronization
        flow_sync = cross_exchange.get("flow_synchronization", {})
        dominant = flow_sync.get("dominant_direction", "NEUTRAL")
        sync_score = flow_sync.get("synchronization_score", 0)
        
        if dominant == "STRONG_BUYING":
            components["flow_sync"] = sync_score
        elif dominant == "BUYING":
            components["flow_sync"] = sync_score * 0.6
        elif dominant == "STRONG_SELLING":
            components["flow_sync"] = -sync_score
        elif dominant == "SELLING":
            components["flow_sync"] = -sync_score * 0.6
        else:
            components["flow_sync"] = 0
        
        result["components"] = components
        
        # Weighted pressure score
        weights = {
            "orderbook_imbalance": 0.20,
            "trade_delta": 0.25,
            "oi_flow": 0.20,
            "funding_skew": 0.15,
            "basis_trend": 0.10,
            "flow_sync": 0.10
        }
        
        pressure_score = sum(
            components.get(k, 0) * v 
            for k, v in weights.items()
        )
        
        result["pressure_score"] = round(pressure_score, 2)
        
        # Direction and strength
        if pressure_score > 50:
            result["pressure_direction"] = "STRONG_BULLISH"
            result["pressure_strength"] = "STRONG"
        elif pressure_score > 25:
            result["pressure_direction"] = "BULLISH"
            result["pressure_strength"] = "MODERATE"
        elif pressure_score > 10:
            result["pressure_direction"] = "SLIGHTLY_BULLISH"
            result["pressure_strength"] = "WEAK"
        elif pressure_score < -50:
            result["pressure_direction"] = "STRONG_BEARISH"
            result["pressure_strength"] = "STRONG"
        elif pressure_score < -25:
            result["pressure_direction"] = "BEARISH"
            result["pressure_strength"] = "MODERATE"
        elif pressure_score < -10:
            result["pressure_direction"] = "SLIGHTLY_BEARISH"
            result["pressure_strength"] = "WEAK"
        else:
            result["pressure_direction"] = "NEUTRAL"
            result["pressure_strength"] = "NONE"
        
        # Confidence based on component agreement
        positive_components = sum(1 for v in components.values() if v > 10)
        negative_components = sum(1 for v in components.values() if v < -10)
        total_components = len(components)
        
        agreement = max(positive_components, negative_components) / total_components
        result["confidence"] = round(agreement * 100, 1)
        
        # Trend from history
        if symbol not in self.pressure_history:
            self.pressure_history[symbol] = deque(maxlen=self.history_window)
        
        self.pressure_history[symbol].append(pressure_score)
        
        if len(self.pressure_history[symbol]) >= 5:
            recent = list(self.pressure_history[symbol])[-5:]
            if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                result["trend"] = "STRENGTHENING_BULLISH"
            elif all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                result["trend"] = "STRENGTHENING_BEARISH"
            else:
                avg_change = (recent[-1] - recent[0]) / len(recent)
                if avg_change > 2:
                    result["trend"] = "TURNING_BULLISH"
                elif avg_change < -2:
                    result["trend"] = "TURNING_BEARISH"
                else:
                    result["trend"] = "STABLE"
        
        # Recommendation
        if result["pressure_direction"] in ("STRONG_BULLISH", "BULLISH") and result["confidence"] > 60:
            result["recommendation"] = "FAVORABLE_FOR_LONGS"
        elif result["pressure_direction"] in ("STRONG_BEARISH", "BEARISH") and result["confidence"] > 60:
            result["recommendation"] = "FAVORABLE_FOR_SHORTS"
        else:
            result["recommendation"] = "NEUTRAL_WAIT"
        
        return result
    
    # =========================================================================
    # 5.2 Squeeze Probability Model
    # =========================================================================
    
    def compute_squeeze_probability(
        self,
        symbol: str,
        order_flow: Dict,
        leverage: Dict,
        regime: Dict
    ) -> Dict:
        """
        Compute probability of a squeeze (short or long).
        
        Inputs:
        - Leverage index
        - Funding extremes
        - Liquidation clusters
        - Liquidity vacuum
        
        Output:
        - Squeeze probability and expected direction
        """
        result = {
            "squeeze_probability": 0,  # 0-100
            "squeeze_type": "NONE",  # SHORT_SQUEEZE, LONG_SQUEEZE, NONE
            "squeeze_intensity": 0,  # 0-100
            "time_to_squeeze": "UNKNOWN",
            "trigger_levels": [],
            "components": {},
            "warning": ""
        }
        
        components = {}
        
        # 1. Leverage index risk
        lev = leverage.get("leverage_index", {})
        lev_zscore = lev.get("leverage_zscore", 0)
        cascade_prob = lev.get("cascade_probability", 0)
        components["leverage_risk"] = cascade_prob
        
        # 2. Funding extremes
        funding = leverage.get("funding_stress", {})
        funding_zscore = funding.get("funding_zscore", 0)
        funding_stress = funding.get("stress_level", "NORMAL")
        
        # High positive funding = longs crowded = short squeeze less likely
        # High negative funding = shorts crowded = long squeeze less likely
        if funding_zscore > 2:
            components["funding_crowding"] = 80  # Long crowding
            components["crowded_side"] = "LONGS"
        elif funding_zscore > 1:
            components["funding_crowding"] = 50
            components["crowded_side"] = "LONGS"
        elif funding_zscore < -2:
            components["funding_crowding"] = 80  # Short crowding
            components["crowded_side"] = "SHORTS"
        elif funding_zscore < -1:
            components["funding_crowding"] = 50
            components["crowded_side"] = "SHORTS"
        else:
            components["funding_crowding"] = 20
            components["crowded_side"] = "NEUTRAL"
        
        # 3. Liquidation clusters
        liq = leverage.get("liquidation_pressure", {})
        long_liq = liq.get("long_liquidation_value", 0)
        short_liq = liq.get("short_liquidation_value", 0)
        total_liq = long_liq + short_liq
        
        if total_liq > 0:
            if long_liq > short_liq * 2:
                components["liq_dominance"] = "LONG_LIQUIDATIONS"
                components["liq_ratio"] = round(long_liq / total_liq * 100, 1)
            elif short_liq > long_liq * 2:
                components["liq_dominance"] = "SHORT_LIQUIDATIONS"
                components["liq_ratio"] = round(short_liq / total_liq * 100, 1)
            else:
                components["liq_dominance"] = "BALANCED"
                components["liq_ratio"] = 50
        else:
            components["liq_dominance"] = "NONE"
            components["liq_ratio"] = 0
        
        # 4. Liquidity vacuum
        vacuum = order_flow.get("liquidity_vacuum", {})
        vacuum_score = vacuum.get("vacuum_score", 0)
        components["vacuum_score"] = vacuum_score
        
        # 5. Regime indicator
        current_regime = regime.get("regime", {}).get("current_regime", "UNKNOWN")
        if current_regime == "SQUEEZE":
            components["regime_confirms"] = True
        else:
            components["regime_confirms"] = False
        
        result["components"] = components
        
        # Calculate squeeze probability
        # Base probability from leverage and funding
        base_prob = (components["leverage_risk"] * 0.4 + 
                    components["funding_crowding"] * 0.3 +
                    components["vacuum_score"] * 0.3)
        
        # Determine squeeze type
        crowded_side = components.get("crowded_side", "NEUTRAL")
        liq_dom = components.get("liq_dominance", "NONE")
        
        if crowded_side == "LONGS" or liq_dom == "LONG_LIQUIDATIONS":
            # Longs are crowded/liquidating -> potential for more long liquidations
            result["squeeze_type"] = "LONG_SQUEEZE"
        elif crowded_side == "SHORTS" or liq_dom == "SHORT_LIQUIDATIONS":
            # Shorts are crowded/covering -> short squeeze potential
            result["squeeze_type"] = "SHORT_SQUEEZE"
        else:
            result["squeeze_type"] = "NONE"
        
        # Boost probability if regime confirms
        if components.get("regime_confirms"):
            base_prob = min(base_prob * 1.3, 100)
        
        result["squeeze_probability"] = round(base_prob, 1)
        
        # Intensity based on leverage and vacuum
        result["squeeze_intensity"] = round(
            (components["leverage_risk"] + components["vacuum_score"]) / 2, 1
        )
        
        # Time to squeeze estimation
        if result["squeeze_probability"] > 80:
            result["time_to_squeeze"] = "IMMINENT"
        elif result["squeeze_probability"] > 60:
            result["time_to_squeeze"] = "LIKELY_SOON"
        elif result["squeeze_probability"] > 40:
            result["time_to_squeeze"] = "POSSIBLE"
        else:
            result["time_to_squeeze"] = "UNLIKELY"
        
        # Trigger levels from liquidation zones
        pressure_zones = liq.get("pressure_zones", [])
        result["trigger_levels"] = [
            {
                "price": z["price"],
                "pct_from_current": z["pct_from_current"],
                "value_at_risk": z["total_value"]
            }
            for z in pressure_zones[:5]
        ]
        
        # Warning message
        if result["squeeze_probability"] > 70:
            if result["squeeze_type"] == "SHORT_SQUEEZE":
                result["warning"] = "HIGH SHORT SQUEEZE RISK - Shorts may be forced to cover"
            elif result["squeeze_type"] == "LONG_SQUEEZE":
                result["warning"] = "HIGH LONG SQUEEZE RISK - Longs may be liquidated"
        
        return result
    
    # =========================================================================
    # 5.3 Smart Money Absorption Detector
    # =========================================================================
    
    def detect_smart_money_absorption(
        self,
        symbol: str,
        order_flow: Dict,
        leverage: Dict
    ) -> Dict:
        """
        Detect smart money absorption patterns.
        
        Detects:
        - Large volume absorbed without price movement
        - Persistent liquidity defending levels
        """
        result = {
            "absorption_detected": False,
            "absorption_type": "NONE",  # BUY_ABSORPTION, SELL_ABSORPTION
            "absorption_strength": 0,  # 0-100
            "defended_levels": [],
            "absorption_zones": [],
            "interpretation": ""
        }
        
        # Get trade and orderbook data
        trade_agg = order_flow.get("trade_aggression", {})
        total_volume = trade_agg.get("total_volume", 0)
        delta = trade_agg.get("delta", 0)
        delta_pct = trade_agg.get("delta_pct", 0)
        
        efficiency = order_flow.get("microstructure_efficiency", {})
        price_impact = efficiency.get("price_impact", 0)
        
        persistence = order_flow.get("orderbook_persistence", {})
        support_zones = persistence.get("support_zones", [])
        resistance_zones = persistence.get("resistance_zones", [])
        
        # Absorption detection logic:
        # High volume + Low price impact + Significant delta = Absorption
        
        # Normalize volume (assume 100 = typical)
        volume_normalized = min(total_volume / 100, 2)  # Cap at 2x typical
        
        # Low price impact with high volume = absorption
        if price_impact > 0:
            absorption_efficiency = volume_normalized / price_impact
        else:
            absorption_efficiency = volume_normalized * 100
        
        absorption_score = min(absorption_efficiency, 100)
        
        # Determine absorption type based on delta
        if delta_pct > 15 and absorption_score > 50:
            result["absorption_detected"] = True
            result["absorption_type"] = "BUY_ABSORPTION"
            result["interpretation"] = "Heavy selling being absorbed by buyers - bullish"
        elif delta_pct < -15 and absorption_score > 50:
            result["absorption_detected"] = True
            result["absorption_type"] = "SELL_ABSORPTION"
            result["interpretation"] = "Heavy buying being absorbed by sellers - bearish"
        elif abs(delta_pct) < 5 and absorption_score > 60:
            # Balanced absorption - market absorbing in both directions
            result["absorption_detected"] = True
            result["absorption_type"] = "BALANCED_ABSORPTION"
            result["interpretation"] = "Two-way absorption - consolidation / accumulation"
        
        result["absorption_strength"] = round(absorption_score, 1)
        
        # Defended levels (from persistent orderbook zones)
        for zone in support_zones[:3]:
            if zone.get("reliability", 0) > 60:
                result["defended_levels"].append({
                    "type": "SUPPORT",
                    "price": zone.get("price"),
                    "reliability": zone.get("reliability"),
                    "avg_size": zone.get("avg_size")
                })
        
        for zone in resistance_zones[:3]:
            if zone.get("reliability", 0) > 60:
                result["defended_levels"].append({
                    "type": "RESISTANCE",
                    "price": zone.get("price"),
                    "reliability": zone.get("reliability"),
                    "avg_size": zone.get("avg_size")
                })
        
        # Sort by reliability
        result["defended_levels"] = sorted(
            result["defended_levels"],
            key=lambda x: x["reliability"],
            reverse=True
        )[:5]
        
        # Track history
        if symbol not in self.absorption_history:
            self.absorption_history[symbol] = deque(maxlen=self.history_window)
        
        self.absorption_history[symbol].append({
            "detected": result["absorption_detected"],
            "type": result["absorption_type"],
            "strength": result["absorption_strength"],
            "timestamp": datetime.utcnow()
        })
        
        return result
    
    # =========================================================================
    # Composite Signal Generation
    # =========================================================================
    
    def _generate_composite_signal(
        self,
        pressure: Dict,
        squeeze: Dict,
        absorption: Dict,
        regime: Dict
    ) -> Dict:
        """
        Generate final composite trading signal.
        """
        result = {
            "signal": "NEUTRAL",  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
            "confidence": 0,
            "urgency": "LOW",  # LOW, MEDIUM, HIGH, IMMEDIATE
            "risk_reward": "UNKNOWN",
            "key_factors": [],
            "action_plan": "",
            "stop_loss_guidance": "",
            "take_profit_guidance": ""
        }
        
        # Collect signals from all components
        signals = []
        
        # 1. Pressure signal
        pressure_dir = pressure.get("pressure_direction", "NEUTRAL")
        pressure_conf = pressure.get("confidence", 0)
        pressure_score = pressure.get("pressure_score", 0)
        
        if "BULLISH" in pressure_dir:
            signals.append(("BUY", pressure_conf, "Institutional pressure bullish"))
        elif "BEARISH" in pressure_dir:
            signals.append(("SELL", pressure_conf, "Institutional pressure bearish"))
        
        # 2. Squeeze signal (can override others)
        squeeze_prob = squeeze.get("squeeze_probability", 0)
        squeeze_type = squeeze.get("squeeze_type", "NONE")
        
        if squeeze_prob > 70:
            if squeeze_type == "SHORT_SQUEEZE":
                signals.append(("STRONG_BUY", squeeze_prob, "Short squeeze imminent"))
            elif squeeze_type == "LONG_SQUEEZE":
                signals.append(("STRONG_SELL", squeeze_prob, "Long squeeze imminent"))
        elif squeeze_prob > 50:
            if squeeze_type == "SHORT_SQUEEZE":
                signals.append(("BUY", squeeze_prob * 0.8, "Short squeeze possible"))
            elif squeeze_type == "LONG_SQUEEZE":
                signals.append(("SELL", squeeze_prob * 0.8, "Long squeeze possible"))
        
        # 3. Absorption signal
        if absorption.get("absorption_detected"):
            abs_type = absorption.get("absorption_type", "NONE")
            abs_strength = absorption.get("absorption_strength", 0)
            
            if abs_type == "BUY_ABSORPTION":
                signals.append(("BUY", abs_strength * 0.8, "Smart money buying"))
            elif abs_type == "SELL_ABSORPTION":
                signals.append(("SELL", abs_strength * 0.8, "Smart money selling"))
        
        # 4. Regime context
        current_regime = regime.get("regime", {}).get("current_regime", "UNKNOWN")
        event_risk = regime.get("event_risk", {}).get("risk_level", "LOW")
        
        # Weight signals based on regime
        regime_multiplier = 1.0
        if current_regime in ("CHAOS", "SQUEEZE"):
            regime_multiplier = 1.2
            result["urgency"] = "HIGH"
        elif current_regime in ("BREAKOUT",):
            regime_multiplier = 1.1
            result["urgency"] = "MEDIUM"
        elif current_regime == "CONSOLIDATION":
            regime_multiplier = 0.8
            result["urgency"] = "LOW"
        
        # Aggregate signals
        if signals:
            buy_weight = sum(c for s, c, _ in signals if "BUY" in s) * regime_multiplier
            sell_weight = sum(c for s, c, _ in signals if "SELL" in s) * regime_multiplier
            
            if buy_weight > sell_weight:
                net_signal = buy_weight - sell_weight
                if net_signal > 100 or any(s == "STRONG_BUY" for s, _, _ in signals):
                    result["signal"] = "STRONG_BUY"
                elif net_signal > 50:
                    result["signal"] = "BUY"
                else:
                    result["signal"] = "WEAK_BUY"
                result["confidence"] = round(min(buy_weight, 100), 1)
            elif sell_weight > buy_weight:
                net_signal = sell_weight - buy_weight
                if net_signal > 100 or any(s == "STRONG_SELL" for s, _, _ in signals):
                    result["signal"] = "STRONG_SELL"
                elif net_signal > 50:
                    result["signal"] = "SELL"
                else:
                    result["signal"] = "WEAK_SELL"
                result["confidence"] = round(min(sell_weight, 100), 1)
            else:
                result["signal"] = "NEUTRAL"
                result["confidence"] = 50
            
            result["key_factors"] = [f for _, _, f in signals]
        
        # Risk/reward assessment
        if event_risk in ("EXTREME", "HIGH"):
            result["risk_reward"] = "UNFAVORABLE"
            result["stop_loss_guidance"] = "Use wider stops or reduce size"
        elif event_risk == "ELEVATED":
            result["risk_reward"] = "MODERATE"
            result["stop_loss_guidance"] = "Standard stops recommended"
        else:
            result["risk_reward"] = "FAVORABLE"
            result["stop_loss_guidance"] = "Tight stops acceptable"
        
        # Action plan
        if result["signal"] in ("STRONG_BUY", "BUY"):
            result["action_plan"] = "Look for long entries on pullbacks"
            result["take_profit_guidance"] = "Scale out at resistance levels"
        elif result["signal"] in ("STRONG_SELL", "SELL"):
            result["action_plan"] = "Look for short entries on bounces"
            result["take_profit_guidance"] = "Scale out at support levels"
        else:
            result["action_plan"] = "Wait for clearer signal or trade range"
            result["take_profit_guidance"] = "N/A"
        
        return result
