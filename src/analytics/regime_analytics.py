"""
Volatility, Regime & Market State Intelligence

Layer 4: Comprehensive market regime detection and event risk analysis.

Derived from: All streams combined
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque
import math

logger = logging.getLogger(__name__)


class RegimeAnalytics:
    """
    Volatility, Regime & Market State Intelligence Engine.
    
    Features:
    4.1 Intraday Regime Detection
    4.2 Event Risk Detection
    """
    
    # Regime definitions
    REGIMES = {
        "ACCUMULATION": "Quiet buying absorption - bullish setup",
        "DISTRIBUTION": "Quiet selling absorption - bearish setup",
        "BREAKOUT": "High volatility expansion - trending",
        "SQUEEZE": "Forced liquidation cascade",
        "MEAN_REVERSION": "Range-bound, reverting to mean",
        "CHAOS": "Extreme volatility, unpredictable",
        "CONSOLIDATION": "Low volatility, building energy"
    }
    
    def __init__(self):
        # Historical tracking
        self.volatility_history: Dict[str, deque] = {}
        self.regime_history: Dict[str, deque] = {}
        self.event_history: Dict[str, deque] = {}
        
        self.history_window = 300
        
    def compute_all_features(
        self,
        symbol: str,
        order_flow_features: Dict,
        leverage_features: Dict,
        cross_exchange_features: Dict
    ) -> Dict:
        """
        Compute all regime and event risk features.
        
        Takes output from other analytics layers as input.
        """
        result = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "regime": self.detect_regime(symbol, order_flow_features, leverage_features),
            "event_risk": self.detect_event_risk(symbol, order_flow_features, leverage_features),
            "volatility_state": self.compute_volatility_state(symbol, order_flow_features)
        }
        
        return result
    
    # =========================================================================
    # 4.1 Intraday Regime Detection
    # =========================================================================
    
    def detect_regime(
        self,
        symbol: str,
        order_flow: Dict,
        leverage: Dict
    ) -> Dict:
        """
        Detect current market regime.
        
        Regimes:
        - Accumulation: Quiet buying absorption
        - Distribution: Quiet selling absorption
        - Breakout: High volatility expansion
        - Squeeze: Forced liquidation cascade
        - Mean Reversion: Range-bound behavior
        - Chaos: Extreme unpredictable volatility
        - Consolidation: Low volatility, energy building
        """
        result = {
            "current_regime": "UNKNOWN",
            "regime_description": "",
            "regime_confidence": 0,
            "regime_duration": 0,
            "regime_strength": 0,
            "sub_regimes": {},
            "transition_probability": {}
        }
        
        # Extract key metrics
        # Order flow metrics
        imbalance = order_flow.get("liquidity_imbalance", {})
        agg_imbalance = imbalance.get("aggregated", {}).get("imbalance_ratio", 0)
        imbalance_signal = imbalance.get("signal", "NEUTRAL")
        
        vacuum = order_flow.get("liquidity_vacuum", {})
        vacuum_score = vacuum.get("vacuum_score", 0)
        
        trade_agg = order_flow.get("trade_aggression", {})
        delta_pct = trade_agg.get("delta_pct", 0)
        aggressor = trade_agg.get("aggressor", "NEUTRAL")
        
        efficiency = order_flow.get("microstructure_efficiency", {})
        market_quality = efficiency.get("market_quality", "NORMAL")
        
        # Leverage metrics
        oi_flow = leverage.get("oi_flow", {})
        position_intent = oi_flow.get("position_intent", "UNKNOWN")
        
        lev_index = leverage.get("leverage_index", {})
        leverage_risk = lev_index.get("risk_level", "NORMAL")
        
        liq_pressure = leverage.get("liquidation_pressure", {})
        cascade_risk = liq_pressure.get("cascade_risk", "LOW")
        liq_dominance = liq_pressure.get("dominance", "NEUTRAL")
        
        funding = leverage.get("funding_stress", {})
        funding_stress = funding.get("stress_level", "NORMAL")
        funding_sentiment = funding.get("sentiment", "NEUTRAL")
        
        basis = leverage.get("basis_regime", {})
        basis_regime = basis.get("regime", "UNKNOWN")
        
        # Sub-regime scores
        scores = {
            "volatility": 0,
            "liquidity": 0,
            "leverage": 0,
            "funding": 0,
            "accumulation": 0,
            "distribution": 0
        }
        
        # Volatility regime
        if vacuum_score > 70 or market_quality in ("FRAGILE", "POOR"):
            scores["volatility"] = 80
        elif vacuum_score > 40:
            scores["volatility"] = 50
        else:
            scores["volatility"] = 20
        
        # Liquidity regime
        if vacuum_score < 30 and market_quality in ("EXCELLENT", "GOOD"):
            scores["liquidity"] = 80
        elif vacuum_score < 50:
            scores["liquidity"] = 50
        else:
            scores["liquidity"] = 20
        
        # Leverage regime
        if leverage_risk in ("EXTREME", "HIGH") or cascade_risk in ("EXTREME", "HIGH"):
            scores["leverage"] = 90
        elif leverage_risk == "ELEVATED":
            scores["leverage"] = 60
        else:
            scores["leverage"] = 30
        
        # Funding regime
        if funding_stress in ("EXTREME", "HIGH"):
            scores["funding"] = 85
        elif funding_stress == "ELEVATED":
            scores["funding"] = 55
        else:
            scores["funding"] = 25
        
        # Accumulation detection
        if (delta_pct > 10 and agg_imbalance > 0.1 and 
            position_intent == "LONGS_OPENING" and scores["volatility"] < 50):
            scores["accumulation"] = 80
        elif delta_pct > 5 and agg_imbalance > 0.05:
            scores["accumulation"] = 50
        
        # Distribution detection
        if (delta_pct < -10 and agg_imbalance < -0.1 and 
            position_intent == "SHORTS_OPENING" and scores["volatility"] < 50):
            scores["distribution"] = 80
        elif delta_pct < -5 and agg_imbalance < -0.05:
            scores["distribution"] = 50
        
        result["sub_regimes"] = scores
        
        # Determine primary regime
        regime = "CONSOLIDATION"
        confidence = 50
        strength = 50
        
        # Squeeze detection (highest priority)
        if (cascade_risk in ("EXTREME", "HIGH") and 
            liq_dominance != "NEUTRAL" and
            scores["volatility"] > 60):
            regime = "SQUEEZE"
            confidence = min(90, scores["leverage"])
            strength = scores["leverage"]
        
        # Chaos detection
        elif (scores["volatility"] > 70 and 
              scores["leverage"] > 70 and
              market_quality == "FRAGILE"):
            regime = "CHAOS"
            confidence = min(85, (scores["volatility"] + scores["leverage"]) / 2)
            strength = scores["volatility"]
        
        # Breakout detection
        elif (scores["volatility"] > 60 and 
              abs(delta_pct) > 20 and
              vacuum_score > 50):
            regime = "BREAKOUT"
            confidence = min(80, scores["volatility"])
            strength = abs(delta_pct)
        
        # Accumulation detection
        elif scores["accumulation"] > 60:
            regime = "ACCUMULATION"
            confidence = scores["accumulation"]
            strength = scores["accumulation"]
        
        # Distribution detection
        elif scores["distribution"] > 60:
            regime = "DISTRIBUTION"
            confidence = scores["distribution"]
            strength = scores["distribution"]
        
        # Mean reversion
        elif (scores["volatility"] < 40 and 
              scores["liquidity"] > 60 and
              abs(delta_pct) < 10):
            regime = "MEAN_REVERSION"
            confidence = scores["liquidity"]
            strength = 100 - scores["volatility"]
        
        # Default: Consolidation
        else:
            regime = "CONSOLIDATION"
            confidence = 60
            strength = 50
        
        result["current_regime"] = regime
        result["regime_description"] = self.REGIMES.get(regime, "Unknown regime")
        result["regime_confidence"] = round(confidence, 1)
        result["regime_strength"] = round(strength, 1)
        
        # Transition probabilities
        result["transition_probability"] = self._compute_transition_probs(regime, scores)
        
        # Track regime history
        if symbol not in self.regime_history:
            self.regime_history[symbol] = deque(maxlen=self.history_window)
        
        self.regime_history[symbol].append({
            "regime": regime,
            "timestamp": datetime.utcnow()
        })
        
        # Compute duration
        duration = 1
        history = list(self.regime_history[symbol])
        for i in range(len(history) - 2, -1, -1):
            if history[i]["regime"] == regime:
                duration += 1
            else:
                break
        result["regime_duration"] = duration
        
        return result
    
    def _compute_transition_probs(
        self,
        current_regime: str,
        scores: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute probabilities of transitioning to other regimes."""
        probs: Dict[str, float] = {}
        
        # Base transition matrix (simplified)
        if current_regime == "CONSOLIDATION":
            probs = {
                "BREAKOUT": float(min(40, scores["volatility"])),
                "ACCUMULATION": float(min(30, scores["accumulation"])),
                "DISTRIBUTION": float(min(30, scores["distribution"])),
                "SQUEEZE": float(min(20, scores["leverage"]))
            }
        elif current_regime == "ACCUMULATION":
            probs = {
                "BREAKOUT": float(min(50, scores["volatility"] + 20)),
                "DISTRIBUTION": float(min(20, scores["distribution"])),
                "CONSOLIDATION": 30.0
            }
        elif current_regime == "DISTRIBUTION":
            probs = {
                "BREAKOUT": float(min(50, scores["volatility"] + 20)),
                "ACCUMULATION": float(min(20, scores["accumulation"])),
                "CONSOLIDATION": 30.0
            }
        elif current_regime == "BREAKOUT":
            probs = {
                "SQUEEZE": float(min(40, scores["leverage"])),
                "CONSOLIDATION": 40.0,
                "CHAOS": float(min(20, scores["volatility"] - 50))
            }
        elif current_regime == "SQUEEZE":
            probs = {
                "CHAOS": float(min(40, scores["volatility"])),
                "MEAN_REVERSION": 30.0,
                "CONSOLIDATION": 30.0
            }
        elif current_regime == "CHAOS":
            probs = {
                "SQUEEZE": float(min(30, scores["leverage"])),
                "MEAN_REVERSION": 40.0,
                "CONSOLIDATION": 30.0
            }
        elif current_regime == "MEAN_REVERSION":
            probs = {
                "CONSOLIDATION": 40.0,
                "BREAKOUT": float(min(30, scores["volatility"])),
                "ACCUMULATION": float(min(20, scores["accumulation"])),
                "DISTRIBUTION": float(min(20, scores["distribution"]))
            }
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: round(v / total * 100, 1) for k, v in probs.items()}
        
        return probs
    
    # =========================================================================
    # 4.2 Event Risk Detection
    # =========================================================================
    
    def detect_event_risk(
        self,
        symbol: str,
        order_flow: Dict,
        leverage: Dict
    ) -> Dict:
        """
        Detect potential market disruption events.
        
        Features:
        - Spread expansion rate
        - Depth collapse rate
        - Trade burst acceleration
        - Liquidation spike probability
        """
        result = {
            "event_risk_score": 0,  # 0-100
            "risk_level": "LOW",
            "active_warnings": [],
            "indicators": {},
            "recommended_actions": []
        }
        
        warnings = []
        risk_scores = []
        
        # 1. Spread expansion risk
        vacuum = order_flow.get("liquidity_vacuum", {})
        vacuum_score = vacuum.get("vacuum_score", 0)
        slippage_risk = vacuum.get("expected_slippage_risk", "LOW")
        
        spread_risk = vacuum_score
        result["indicators"]["spread_expansion"] = {
            "score": round(spread_risk, 1),
            "slippage_risk": slippage_risk
        }
        risk_scores.append(spread_risk)
        
        if spread_risk > 70:
            warnings.append({
                "type": "SPREAD_EXPANSION",
                "severity": "HIGH",
                "message": "Orderbook spreads widening rapidly"
            })
        
        # 2. Depth collapse risk
        thin_zones = vacuum.get("thin_zones", [])
        depth_collapse_score = min(len(thin_zones) * 15, 100)
        
        result["indicators"]["depth_collapse"] = {
            "score": round(depth_collapse_score, 1),
            "thin_zones_count": len(thin_zones)
        }
        risk_scores.append(depth_collapse_score)
        
        if depth_collapse_score > 60:
            warnings.append({
                "type": "DEPTH_COLLAPSE",
                "severity": "MEDIUM",
                "message": f"Detected {len(thin_zones)} thin liquidity zones"
            })
        
        # 3. Trade burst risk
        trade_agg = order_flow.get("trade_aggression", {})
        delta_accel = abs(trade_agg.get("delta_acceleration", 0))
        large_trades = len(trade_agg.get("large_trades", []))
        
        trade_burst_score = min(delta_accel * 10 + large_trades * 10, 100)
        
        result["indicators"]["trade_burst"] = {
            "score": round(trade_burst_score, 1),
            "delta_acceleration": round(delta_accel, 4),
            "large_trade_count": large_trades
        }
        risk_scores.append(trade_burst_score)
        
        if trade_burst_score > 70:
            warnings.append({
                "type": "TRADE_BURST",
                "severity": "HIGH",
                "message": "Aggressive trade flow acceleration detected"
            })
        
        # 4. Liquidation spike risk
        liq = leverage.get("liquidation_pressure", {})
        cascade_risk = liq.get("cascade_risk", "LOW")
        liq_momentum = liq.get("liquidation_momentum", 0)
        
        liq_score_map = {"LOW": 10, "MODERATE": 40, "HIGH": 70, "EXTREME": 95}
        liq_risk_score = liq_score_map.get(cascade_risk, 10) + min(abs(liq_momentum) / 10000, 20)
        
        result["indicators"]["liquidation_spike"] = {
            "score": round(liq_risk_score, 1),
            "cascade_risk": cascade_risk,
            "momentum": round(liq_momentum, 2)
        }
        risk_scores.append(liq_risk_score)
        
        if liq_risk_score > 60:
            warnings.append({
                "type": "LIQUIDATION_CASCADE",
                "severity": "HIGH" if liq_risk_score > 80 else "MEDIUM",
                "message": f"Liquidation cascade risk: {cascade_risk}"
            })
        
        # 5. Leverage stress risk
        lev = leverage.get("leverage_index", {})
        lev_zscore = abs(lev.get("leverage_zscore", 0))
        
        lev_risk_score = min(lev_zscore * 30, 100)
        
        result["indicators"]["leverage_stress"] = {
            "score": round(lev_risk_score, 1),
            "zscore": round(lev_zscore, 2)
        }
        risk_scores.append(lev_risk_score)
        
        if lev_risk_score > 60:
            warnings.append({
                "type": "LEVERAGE_EXTREME",
                "severity": "MEDIUM",
                "message": "Leverage build-up at extreme levels"
            })
        
        # 6. Funding stress risk
        fund = leverage.get("funding_stress", {})
        fund_zscore = abs(fund.get("funding_zscore", 0))
        
        fund_risk_score = min(fund_zscore * 25, 100)
        
        result["indicators"]["funding_stress"] = {
            "score": round(fund_risk_score, 1),
            "zscore": round(fund_zscore, 2)
        }
        risk_scores.append(fund_risk_score)
        
        if fund_risk_score > 70:
            warnings.append({
                "type": "FUNDING_EXTREME",
                "severity": "MEDIUM",
                "message": "Funding rates at extreme levels - reversal risk"
            })
        
        # Aggregate risk score
        if risk_scores:
            # Weighted average with emphasis on highest risks
            sorted_scores = sorted(risk_scores, reverse=True)
            weights = [2, 1.5, 1.2, 1, 1, 1][:len(sorted_scores)]
            weighted_sum = sum(s * w for s, w in zip(sorted_scores, weights))
            total_weight = sum(weights)
            result["event_risk_score"] = round(weighted_sum / total_weight, 1)
        
        # Risk level classification
        if result["event_risk_score"] > 80:
            result["risk_level"] = "EXTREME"
        elif result["event_risk_score"] > 60:
            result["risk_level"] = "HIGH"
        elif result["event_risk_score"] > 40:
            result["risk_level"] = "ELEVATED"
        elif result["event_risk_score"] > 20:
            result["risk_level"] = "MODERATE"
        else:
            result["risk_level"] = "LOW"
        
        result["active_warnings"] = warnings
        
        # Recommended actions
        if result["risk_level"] in ("EXTREME", "HIGH"):
            result["recommended_actions"] = [
                "Reduce position sizes",
                "Widen stop-losses",
                "Avoid market orders",
                "Wait for volatility to subside"
            ]
        elif result["risk_level"] == "ELEVATED":
            result["recommended_actions"] = [
                "Use limit orders only",
                "Monitor liquidation levels",
                "Consider hedging positions"
            ]
        
        return result
    
    # =========================================================================
    # Volatility State Analysis
    # =========================================================================
    
    def compute_volatility_state(
        self,
        symbol: str,
        order_flow: Dict
    ) -> Dict:
        """
        Compute current volatility state and clustering.
        """
        result = {
            "volatility_regime": "NORMAL",
            "volatility_percentile": 50,
            "clustering_detected": False,
            "expansion_probability": 0,
            "contraction_probability": 0
        }
        
        efficiency = order_flow.get("microstructure_efficiency", {})
        micro_vol = efficiency.get("micro_volatility", 0)
        
        # Track volatility history
        if symbol not in self.volatility_history:
            self.volatility_history[symbol] = deque(maxlen=self.history_window)
        
        self.volatility_history[symbol].append(micro_vol)
        
        # Need history for percentile
        history = list(self.volatility_history[symbol])
        if len(history) >= 10:
            sorted_hist = sorted(history)
            percentile = (sorted_hist.index(min(sorted_hist, key=lambda x: abs(x - micro_vol))) / len(sorted_hist)) * 100
            result["volatility_percentile"] = round(percentile, 1)
            
            # Regime classification
            if percentile > 90:
                result["volatility_regime"] = "EXTREME"
            elif percentile > 75:
                result["volatility_regime"] = "HIGH"
            elif percentile > 50:
                result["volatility_regime"] = "ELEVATED"
            elif percentile > 25:
                result["volatility_regime"] = "NORMAL"
            else:
                result["volatility_regime"] = "LOW"
            
            # Clustering detection (volatility tends to cluster)
            recent = history[-10:]
            if len(recent) >= 3:
                # Check if recent volatilities are similar (clustering)
                mean_recent = sum(recent) / len(recent)
                std_recent = math.sqrt(sum((v - mean_recent) ** 2 for v in recent) / len(recent))
                cv = std_recent / mean_recent if mean_recent > 0 else 0
                
                if cv < 0.3:  # Low coefficient of variation = clustering
                    result["clustering_detected"] = True
            
            # Expansion/contraction probability
            if result["volatility_regime"] == "LOW":
                result["expansion_probability"] = 70
                result["contraction_probability"] = 30
            elif result["volatility_regime"] in ("HIGH", "EXTREME"):
                result["expansion_probability"] = 30
                result["contraction_probability"] = 70
            else:
                result["expansion_probability"] = 50
                result["contraction_probability"] = 50
        
        return result
