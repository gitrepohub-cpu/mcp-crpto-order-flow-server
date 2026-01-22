"""
🧠 Composite Signal Calculator
==============================

Combines features from multiple streams into actionable composite signals.

Phase 1 Signals (5):
- Smart Money Index
- Squeeze Probability
- Stop Hunt Detection
- Momentum Quality
- Risk Score

Phase 3 Signals (10 NEW):
- Market Maker Activity Index
- Liquidation Cascade Risk
- Institutional Phase Detection
- Volatility Breakout Predictor
- Mean Reversion Signal
- Momentum Exhaustion Detector
- Smart Money Flow Direction
- Arbitrage Opportunity Score
- Regime Transition Probability
- Execution Quality Score
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field

from .base import InstitutionalFeatureCalculator, FeatureBuffer

logger = logging.getLogger(__name__)


@dataclass
class CompositeSignal:
    """Container for a composite signal."""
    name: str
    value: float
    confidence: float
    components: Dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'confidence': self.confidence,
            'components': self.components,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class EnhancedSignal:
    """Container for enhanced Phase 3 composite signals with richer output."""
    name: str
    value: float
    confidence: float
    components: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'name': self.name,
            'value': self.value,
            'confidence': self.confidence,
            'components': self.components,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }
        result.update(self.metadata)
        return result


class CompositeSignalCalculator:
    """
    Calculates composite signals from multi-stream features.
    
    Combines:
    - Prices features (microprice, spread, efficiency)
    - Orderbook features (depth, absorption, walls)
    - Trades features (CVD, whale detection, flow)
    - Funding features (rate, momentum, carry)
    - OI features (leverage, intent, cascade risk)
    
    Outputs:
    - Smart Money Index (institutional activity detection)
    - Squeeze Probability (short/long squeeze likelihood)
    - Stop Hunt Detection (manipulation detection)
    - Momentum Quality Score (trend sustainability)
    - Composite Risk Score (overall market risk)
    """
    
    def __init__(self, window_size: int = 50):
        """Initialize composite calculator."""
        self.window_size = window_size
        
        # Buffers for composite signals (Phase 1)
        self._smart_money_buffer: List[float] = []
        self._squeeze_buffer: List[float] = []
        self._stop_hunt_buffer: List[float] = []
        self._momentum_buffer: List[float] = []
        self._risk_buffer: List[float] = []
        
        # Buffers for Phase 3 signals
        self._market_maker_buffer: List[float] = []
        self._cascade_risk_buffer: List[float] = []
        self._institutional_phase_buffer: List[str] = []
        self._volatility_breakout_buffer: List[float] = []
        self._mean_reversion_buffer: List[float] = []
        self._exhaustion_buffer: List[float] = []
        self._smart_flow_buffer: List[float] = []
        self._arbitrage_buffer: List[float] = []
        self._regime_buffer: List[str] = []
        self._execution_buffer: List[float] = []
        
        # Feature cache for incomplete data (all 8 stream types)
        self._feature_cache: Dict[str, Dict[str, Any]] = {
            'prices': {},
            'orderbook': {},
            'trades': {},
            'funding': {},
            'oi': {},
            'liquidations': {},
            'mark_prices': {},
            'ticker': {},
        }
        
        # Weights for signal components
        self._smart_money_weights = {
            'orderbook_absorption': 0.25,
            'whale_activity': 0.30,
            'flow_toxicity': 0.20,
            'cvd_divergence': 0.25,
        }
        
        self._squeeze_weights = {
            'funding_extreme': 0.30,
            'oi_leverage': 0.25,
            'crowding': 0.25,
            'price_compression': 0.20,
        }
        
        # Phase 3 weights
        self._market_maker_weights = {
            'wall_activity': 0.30,
            'spread_manipulation': 0.25,
            'quote_stuffing': 0.25,
            'inventory_signal': 0.20,
        }
        
        self._cascade_weights = {
            'liquidation_cluster': 0.35,
            'leverage_stress': 0.30,
            'thin_liquidity': 0.20,
            'funding_pressure': 0.15,
        }
    
    def update_features(self, stream_type: str, features: Dict[str, Any]):
        """
        Update feature cache for a stream type.
        
        Args:
            stream_type: One of 'prices', 'orderbook', 'trades', 'funding', 'oi'
            features: Feature dict from that stream's calculator
        """
        if stream_type in self._feature_cache:
            self._feature_cache[stream_type] = features
    
    def calculate_all(self, timestamp: Optional[datetime] = None) -> Dict[str, CompositeSignal]:
        """
        Calculate all composite signals from cached features.
        
        Returns:
            Dict of signal name -> CompositeSignal (Phase 1) or EnhancedSignal (Phase 3)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        signals = {}
        
        # ===== PHASE 1 SIGNALS (5) =====
        
        # Smart Money Index
        smart_money = self._calculate_smart_money_index()
        if smart_money:
            signals['smart_money_index'] = CompositeSignal(
                name='Smart Money Index',
                value=smart_money['value'],
                confidence=smart_money['confidence'],
                components=smart_money['components'],
                timestamp=timestamp,
            )
        
        # Squeeze Probability
        squeeze = self._calculate_squeeze_probability()
        if squeeze:
            signals['squeeze_probability'] = CompositeSignal(
                name='Squeeze Probability',
                value=squeeze['value'],
                confidence=squeeze['confidence'],
                components=squeeze['components'],
                timestamp=timestamp,
            )
        
        # Stop Hunt Detection
        stop_hunt = self._calculate_stop_hunt_probability()
        if stop_hunt:
            signals['stop_hunt_probability'] = CompositeSignal(
                name='Stop Hunt Probability',
                value=stop_hunt['value'],
                confidence=stop_hunt['confidence'],
                components=stop_hunt['components'],
                timestamp=timestamp,
            )
        
        # Momentum Quality
        momentum = self._calculate_momentum_quality()
        if momentum:
            signals['momentum_quality'] = CompositeSignal(
                name='Momentum Quality',
                value=momentum['value'],
                confidence=momentum['confidence'],
                components=momentum['components'],
                timestamp=timestamp,
            )
        
        # Composite Risk
        risk = self._calculate_composite_risk()
        if risk:
            signals['composite_risk'] = CompositeSignal(
                name='Composite Risk',
                value=risk['value'],
                confidence=risk['confidence'],
                components=risk['components'],
                timestamp=timestamp,
            )
        
        # ===== PHASE 3 SIGNALS (10) =====
        
        # Market Maker Activity
        mm_activity = self._calculate_market_maker_activity()
        if mm_activity:
            signals['market_maker_activity'] = EnhancedSignal(
                name='Market Maker Activity Index',
                value=mm_activity['value'],
                confidence=mm_activity['confidence'],
                components=mm_activity['components'],
                timestamp=timestamp,
                metadata={
                    'activity_type': mm_activity['activity_type'],
                    'inventory_bias': mm_activity['inventory_bias'],
                },
            )
        
        # Liquidation Cascade Risk
        cascade = self._calculate_liquidation_cascade_risk()
        if cascade:
            signals['liquidation_cascade_risk'] = EnhancedSignal(
                name='Liquidation Cascade Risk',
                value=cascade['value'],
                confidence=cascade['confidence'],
                components=cascade['components'],
                timestamp=timestamp,
                metadata={
                    'severity': cascade['severity'],
                    'direction': cascade['direction'],
                    'estimated_trigger_move_pct': cascade['estimated_trigger_move_pct'],
                },
            )
        
        # Institutional Phase
        inst_phase = self._detect_institutional_phase()
        if inst_phase:
            signals['institutional_phase'] = EnhancedSignal(
                name='Institutional Phase Detection',
                value=inst_phase['value'],
                confidence=inst_phase['confidence'],
                components=inst_phase['components'],
                timestamp=timestamp,
                metadata={
                    'phase': inst_phase['phase'],
                    'intensity': inst_phase['intensity'],
                    'price_target_direction': inst_phase['price_target_direction'],
                },
            )
        
        # Volatility Breakout
        vol_breakout = self._predict_volatility_breakout()
        if vol_breakout:
            signals['volatility_breakout'] = EnhancedSignal(
                name='Volatility Breakout Predictor',
                value=vol_breakout['value'],
                confidence=vol_breakout['confidence'],
                components=vol_breakout['components'],
                timestamp=timestamp,
                metadata={
                    'direction': vol_breakout['direction'],
                    'expected_magnitude_pct': vol_breakout['expected_magnitude_pct'],
                    'timeframe_hours': vol_breakout['timeframe_hours'],
                },
            )
        
        # Mean Reversion Signal
        mean_rev = self._calculate_mean_reversion_signal()
        if mean_rev:
            signals['mean_reversion_signal'] = EnhancedSignal(
                name='Mean Reversion Signal',
                value=mean_rev['value'],
                confidence=mean_rev['confidence'],
                components=mean_rev['components'],
                timestamp=timestamp,
                metadata={
                    'signal': mean_rev['signal'],
                    'strength': mean_rev['strength'],
                    'expected_reversion_pct': mean_rev['expected_reversion_pct'],
                },
            )
        
        # Momentum Exhaustion
        exhaustion = self._detect_momentum_exhaustion()
        if exhaustion:
            signals['momentum_exhaustion'] = EnhancedSignal(
                name='Momentum Exhaustion Detector',
                value=exhaustion['value'],
                confidence=exhaustion['confidence'],
                components=exhaustion['components'],
                timestamp=timestamp,
                metadata={
                    'exhaustion': exhaustion['exhaustion'],
                    'trend_direction': exhaustion['trend_direction'],
                    'reversal_probability': exhaustion['reversal_probability'],
                },
            )
        
        # Smart Money Flow
        smart_flow = self._calculate_smart_money_flow()
        if smart_flow:
            signals['smart_money_flow'] = EnhancedSignal(
                name='Smart Money Flow Direction',
                value=smart_flow['value'],
                confidence=smart_flow['confidence'],
                components=smart_flow['components'],
                timestamp=timestamp,
                metadata={
                    'direction': smart_flow['direction'],
                    'strength': smart_flow['strength'],
                    'volume_estimate_btc': smart_flow['volume_estimate_btc'],
                },
            )
        
        # Arbitrage Opportunity
        arbitrage = self._detect_arbitrage_opportunity()
        if arbitrage:
            signals['arbitrage_opportunity'] = EnhancedSignal(
                name='Arbitrage Opportunity Score',
                value=arbitrage['value'],
                confidence=arbitrage['confidence'],
                components=arbitrage['components'],
                timestamp=timestamp,
                metadata={
                    'opportunity': arbitrage['opportunity'],
                    'type': arbitrage['type'],
                    'expected_return_pct_annual': arbitrage['expected_return_pct_annual'],
                    'risk_level': arbitrage['risk_level'],
                },
            )
        
        # Regime Transition
        regime = self._predict_regime_transition()
        if regime:
            signals['regime_transition'] = EnhancedSignal(
                name='Regime Transition Probability',
                value=regime['value'],
                confidence=regime['confidence'],
                components=regime['components'],
                timestamp=timestamp,
                metadata={
                    'current_regime': regime['current_regime'],
                    'predicted_regime': regime['predicted_regime'],
                    'transition_probability': regime['transition_probability'],
                    'timeframe_hours': regime['timeframe_hours'],
                },
            )
        
        # Execution Quality
        execution = self._calculate_execution_quality()
        if execution:
            signals['execution_quality'] = EnhancedSignal(
                name='Execution Quality Score',
                value=execution['value'],
                confidence=execution['confidence'],
                components=execution['components'],
                timestamp=timestamp,
                metadata={
                    'quality': execution['quality'],
                    'slippage_estimate_bps': execution['slippage_estimate_bps'],
                    'max_size_no_impact_btc': execution['max_size_no_impact_btc'],
                    'optimal_order_splits': execution['optimal_order_splits'],
                },
            )
        
        return signals
    
    def get_all_signals_dict(self, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Get all signals as a flat dictionary for storage."""
        signals = self.calculate_all(timestamp)
        
        result = {}
        for name, signal in signals.items():
            result[name] = signal.value
            result[f'{name}_confidence'] = signal.confidence
            
            # Include metadata for EnhancedSignals
            if isinstance(signal, EnhancedSignal) and signal.metadata:
                for key, val in signal.metadata.items():
                    result[f'{name}_{key}'] = val
        
        return result
    
    def get_phase3_signals(self, timestamp: Optional[datetime] = None) -> Dict[str, EnhancedSignal]:
        """Get only Phase 3 enhanced signals."""
        all_signals = self.calculate_all(timestamp)
        return {k: v for k, v in all_signals.items() if isinstance(v, EnhancedSignal)}
    
    def get_trading_summary(self, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get a trading-focused summary of all signals.
        
        Returns actionable intelligence for trading decisions.
        """
        signals = self.calculate_all(timestamp)
        
        summary = {
            'timestamp': timestamp or datetime.now(timezone.utc),
            'signal_count': len(signals),
            'alerts': [],
            'opportunities': [],
            'risks': [],
            'recommendation': 'NEUTRAL',
        }
        
        # Check for critical alerts
        cascade = signals.get('liquidation_cascade_risk')
        if cascade and hasattr(cascade, 'metadata'):
            if cascade.metadata.get('severity') in ('HIGH', 'CRITICAL'):
                summary['alerts'].append({
                    'type': 'LIQUIDATION_CASCADE',
                    'severity': cascade.metadata['severity'],
                    'message': f"High cascade risk - {cascade.metadata['direction']} direction"
                })
        
        stop_hunt = signals.get('stop_hunt_probability')
        if stop_hunt and stop_hunt.value > 0.6:
            summary['alerts'].append({
                'type': 'STOP_HUNT',
                'severity': 'HIGH' if stop_hunt.value > 0.8 else 'MEDIUM',
                'message': 'Stop hunting activity detected'
            })
        
        # Check for opportunities
        arb = signals.get('arbitrage_opportunity')
        if arb and hasattr(arb, 'metadata') and arb.metadata.get('opportunity'):
            summary['opportunities'].append({
                'type': f"{arb.metadata['type'].upper()}_ARBITRAGE",
                'expected_return': arb.metadata['expected_return_pct_annual'],
                'risk': arb.metadata['risk_level']
            })
        
        inst_phase = signals.get('institutional_phase')
        if inst_phase and hasattr(inst_phase, 'metadata'):
            if inst_phase.metadata['phase'] != 'NEUTRAL' and inst_phase.metadata['intensity'] > 0.5:
                summary['opportunities'].append({
                    'type': inst_phase.metadata['phase'],
                    'direction': inst_phase.metadata['price_target_direction'],
                    'intensity': inst_phase.metadata['intensity']
                })
        
        mean_rev = signals.get('mean_reversion_signal')
        if mean_rev and hasattr(mean_rev, 'metadata'):
            if mean_rev.metadata['signal'] != 'NEUTRAL' and mean_rev.metadata['strength'] > 0.5:
                summary['opportunities'].append({
                    'type': 'MEAN_REVERSION',
                    'signal': mean_rev.metadata['signal'],
                    'expected_reversion': mean_rev.metadata['expected_reversion_pct']
                })
        
        # Assess overall recommendation
        bullish_signals = 0
        bearish_signals = 0
        
        smart_flow = signals.get('smart_money_flow')
        if smart_flow and hasattr(smart_flow, 'metadata'):
            if smart_flow.metadata['direction'] == 'BUYING':
                bullish_signals += smart_flow.metadata['strength']
            elif smart_flow.metadata['direction'] == 'SELLING':
                bearish_signals += smart_flow.metadata['strength']
        
        momentum = signals.get('momentum_quality')
        if momentum:
            if momentum.value > 0.6:
                bullish_signals += momentum.value - 0.5
            elif momentum.value < 0.4:
                bearish_signals += 0.5 - momentum.value
        
        # Risk assessment
        risk_score = 0
        composite_risk = signals.get('composite_risk')
        if composite_risk:
            risk_score = composite_risk.value
            if risk_score > 0.6:
                summary['risks'].append({
                    'type': 'HIGH_RISK_ENVIRONMENT',
                    'score': risk_score
                })
        
        # Final recommendation
        if risk_score > 0.7:
            summary['recommendation'] = 'REDUCE_EXPOSURE'
        elif bullish_signals > bearish_signals + 0.3:
            summary['recommendation'] = 'BULLISH'
        elif bearish_signals > bullish_signals + 0.3:
            summary['recommendation'] = 'BEARISH'
        else:
            summary['recommendation'] = 'NEUTRAL'
        
        return summary
    
    def _calculate_smart_money_index(self) -> Optional[Dict[str, Any]]:
        """
        Calculate Smart Money Index.
        
        Detects institutional/informed trading activity.
        
        Components:
        - Orderbook absorption (institutions absorbing without price impact)
        - Whale trade activity (large directional trades)
        - Flow toxicity (informed order flow)
        - CVD-price divergence (accumulation/distribution)
        """
        orderbook = self._feature_cache.get('orderbook', {})
        trades = self._feature_cache.get('trades', {})
        prices = self._feature_cache.get('prices', {})
        
        components = {}
        available = 0
        
        # Absorption ratio (high = institutions absorbing)
        absorption = orderbook.get('absorption_ratio', None)
        if absorption is not None:
            components['orderbook_absorption'] = min(1.0, absorption)
            available += 1
        
        # Whale activity
        whale_detected = trades.get('whale_trade_detected', False)
        whale_ratio = trades.get('whale_volume_ratio', 0.0)
        if whale_ratio is not None:
            whale_score = whale_ratio * 2 + (0.3 if whale_detected else 0)
            components['whale_activity'] = min(1.0, whale_score)
            available += 1
        
        # Flow toxicity
        toxicity = trades.get('flow_toxicity', None)
        if toxicity is not None:
            components['flow_toxicity'] = toxicity
            available += 1
        
        # CVD divergence
        divergence = trades.get('cvd_price_divergence', None)
        if divergence is not None:
            components['cvd_divergence'] = abs(divergence)
            available += 1
        
        if available < 2:
            return None
        
        # Weighted sum
        value = 0.0
        total_weight = 0.0
        for comp_name, comp_value in components.items():
            weight = self._smart_money_weights.get(comp_name, 0.25)
            value += comp_value * weight
            total_weight += weight
        
        if total_weight > 0:
            value /= total_weight
        
        # Confidence based on data availability
        confidence = available / 4
        
        # Update buffer
        self._smart_money_buffer.append(value)
        if len(self._smart_money_buffer) > self.window_size:
            self._smart_money_buffer.pop(0)
        
        return {
            'value': value,
            'confidence': confidence,
            'components': components,
        }
    
    def _calculate_squeeze_probability(self) -> Optional[Dict[str, Any]]:
        """
        Calculate Squeeze Probability.
        
        Detects likelihood of short or long squeeze.
        
        Components:
        - Extreme funding (positions crowded one direction)
        - High OI leverage (over-leveraged positions)
        - Position crowding (L/S ratio extremes)
        - Price compression (building pressure)
        """
        funding = self._feature_cache.get('funding', {})
        oi = self._feature_cache.get('oi', {})
        prices = self._feature_cache.get('prices', {})
        
        components = {}
        available = 0
        
        # Funding extreme
        funding_zscore = funding.get('funding_zscore', None)
        if funding_zscore is not None:
            # Extreme funding = squeeze setup
            components['funding_extreme'] = min(1.0, abs(funding_zscore) / 3)
            available += 1
        
        # OI leverage
        leverage_zscore = oi.get('leverage_zscore', None)
        if leverage_zscore is not None:
            components['oi_leverage'] = min(1.0, abs(leverage_zscore) / 2)
            available += 1
        
        # Position crowding
        crowding = oi.get('position_crowding_score', None)
        if crowding is not None:
            components['crowding'] = crowding
            available += 1
        
        # Price compression (tight spread = pressure building)
        spread_zscore = prices.get('spread_zscore', None)
        if spread_zscore is not None:
            # Negative zscore = compressed spread
            components['price_compression'] = max(0, -spread_zscore / 2)
            available += 1
        
        if available < 2:
            return None
        
        # Weighted sum
        value = 0.0
        total_weight = 0.0
        for comp_name, comp_value in components.items():
            weight = self._squeeze_weights.get(comp_name, 0.25)
            value += comp_value * weight
            total_weight += weight
        
        if total_weight > 0:
            value /= total_weight
        
        confidence = available / 4
        
        self._squeeze_buffer.append(value)
        if len(self._squeeze_buffer) > self.window_size:
            self._squeeze_buffer.pop(0)
        
        return {
            'value': min(1.0, value),
            'confidence': confidence,
            'components': components,
        }
    
    def _calculate_stop_hunt_probability(self) -> Optional[Dict[str, Any]]:
        """
        Calculate Stop Hunt Probability.
        
        Detects potential stop hunting manipulation.
        
        Components:
        - Wall manipulation (pull/push walls)
        - Spoofing indicators (add/cancel ratio)
        - Price efficiency degradation
        - Sweep detection
        """
        orderbook = self._feature_cache.get('orderbook', {})
        trades = self._feature_cache.get('trades', {})
        prices = self._feature_cache.get('prices', {})
        
        components = {}
        available = 0
        
        # Wall manipulation
        pull_wall = orderbook.get('pull_wall_detected', False)
        push_wall = orderbook.get('push_wall_detected', False)
        wall_score = (0.5 if pull_wall else 0) + (0.3 if push_wall else 0)
        if wall_score > 0:
            components['wall_manipulation'] = wall_score
            available += 1
        
        # Spoofing (low add/cancel ratio)
        add_cancel = orderbook.get('add_cancel_ratio', None)
        if add_cancel is not None:
            # Low ratio = more cancellations = spoofing
            spoof_score = max(0, 1 - add_cancel)
            components['spoofing_indicator'] = spoof_score
            available += 1
        
        # Price efficiency degradation
        efficiency = prices.get('price_efficiency', None)
        if efficiency is not None:
            # Low efficiency = potential manipulation
            components['efficiency_degradation'] = max(0, 1 - efficiency)
            available += 1
        
        # Sweep activity
        sweep = trades.get('sweep_detected', False)
        if sweep:
            components['sweep_detected'] = 0.8
            available += 1
        
        if available < 1:
            return None
        
        # Simple average for stop hunt
        value = sum(components.values()) / len(components) if components else 0
        confidence = available / 4
        
        self._stop_hunt_buffer.append(value)
        if len(self._stop_hunt_buffer) > self.window_size:
            self._stop_hunt_buffer.pop(0)
        
        return {
            'value': min(1.0, value),
            'confidence': confidence,
            'components': components,
        }
    
    def _calculate_momentum_quality(self) -> Optional[Dict[str, Any]]:
        """
        Calculate Momentum Quality Score.
        
        Assesses sustainability of current price trend.
        
        Components:
        - CVD confirmation (volume supporting direction)
        - OI confirmation (positions supporting direction)
        - Funding alignment (market sentiment)
        - Orderbook support (depth supporting direction)
        """
        trades = self._feature_cache.get('trades', {})
        oi = self._feature_cache.get('oi', {})
        funding = self._feature_cache.get('funding', {})
        orderbook = self._feature_cache.get('orderbook', {})
        prices = self._feature_cache.get('prices', {})
        
        components = {}
        available = 0
        
        # CVD slope (positive = bullish momentum)
        cvd_slope = trades.get('cvd_slope', None)
        if cvd_slope is not None:
            # Normalize to 0-1
            components['cvd_confirmation'] = 0.5 + np.clip(cvd_slope * 10, -0.5, 0.5)
            available += 1
        
        # OI intent
        intent_score = oi.get('position_intent_score', None)
        if intent_score is not None:
            components['oi_confirmation'] = 0.5 + intent_score * 0.5
            available += 1
        
        # Funding alignment
        funding_regime_score = funding.get('funding_regime_score', None)
        if funding_regime_score is not None:
            components['funding_alignment'] = 0.5 + funding_regime_score * 0.25
            available += 1
        
        # Orderbook support (imbalance)
        imbalance = orderbook.get('depth_imbalance_5', None)
        if imbalance is not None:
            components['orderbook_support'] = 0.5 + imbalance * 0.5
            available += 1
        
        if available < 2:
            return None
        
        # Average components
        value = sum(components.values()) / len(components)
        confidence = available / 4
        
        self._momentum_buffer.append(value)
        if len(self._momentum_buffer) > self.window_size:
            self._momentum_buffer.pop(0)
        
        return {
            'value': np.clip(value, 0, 1),
            'confidence': confidence,
            'components': components,
        }
    
    def _calculate_composite_risk(self) -> Optional[Dict[str, Any]]:
        """
        Calculate Composite Risk Score.
        
        Overall market risk assessment.
        
        Components:
        - Liquidation cascade risk
        - Stop hunt probability
        - Squeeze probability
        - Volatility regime
        """
        oi = self._feature_cache.get('oi', {})
        prices = self._feature_cache.get('prices', {})
        
        components = {}
        available = 0
        
        # Cascade risk
        cascade_risk = oi.get('liquidation_cascade_risk', None)
        if cascade_risk is not None:
            components['cascade_risk'] = cascade_risk
            available += 1
        
        # Include recent squeeze probability
        if self._squeeze_buffer:
            components['squeeze_risk'] = self._squeeze_buffer[-1]
            available += 1
        
        # Include stop hunt probability
        if self._stop_hunt_buffer:
            components['stop_hunt_risk'] = self._stop_hunt_buffer[-1]
            available += 1
        
        # Price volatility (from Hurst exponent)
        hurst = prices.get('hurst_exponent', None)
        if hurst is not None:
            # H < 0.5 = mean reverting (lower risk)
            # H > 0.5 = trending (can be risky)
            vol_risk = abs(hurst - 0.5) * 2
            components['volatility_regime'] = vol_risk
            available += 1
        
        if available < 1:
            return None
        
        value = sum(components.values()) / len(components)
        confidence = available / 4
        
        self._risk_buffer.append(value)
        if len(self._risk_buffer) > self.window_size:
            self._risk_buffer.pop(0)
        
        return {
            'value': np.clip(value, 0, 1),
            'confidence': confidence,
            'components': components,
        }
    
    # ========== PHASE 3: ENHANCED COMPOSITE SIGNALS ==========
    
    def _calculate_market_maker_activity(self) -> Optional[Dict[str, Any]]:
        """
        Calculate Market Maker Activity Index.
        
        Detects market maker positioning and inventory management.
        
        Components:
        - Wall activity (bid/ask wall manipulation)
        - Spread manipulation (artificial tightening/widening)
        - Quote stuffing indicators
        - Inventory rotation signals
        
        Returns:
            {
                'value': 0.0-1.0,
                'confidence': 0.0-1.0,
                'components': {...},
                'activity_type': 'aggressive'|'passive'|'neutral',
                'inventory_bias': 'long'|'short'|'balanced'
            }
        """
        orderbook = self._feature_cache.get('orderbook', {})
        trades = self._feature_cache.get('trades', {})
        prices = self._feature_cache.get('prices', {})
        
        components = {}
        available = 0
        
        # Wall activity detection
        pull_wall = orderbook.get('pull_wall_detected', False)
        push_wall = orderbook.get('push_wall_detected', False)
        bid_wall = orderbook.get('bid_wall_detected', False)
        ask_wall = orderbook.get('ask_wall_detected', False)
        
        wall_score = 0.0
        if pull_wall or push_wall:
            wall_score += 0.4
        if bid_wall or ask_wall:
            wall_score += 0.3
        
        if wall_score > 0:
            components['wall_activity'] = min(1.0, wall_score)
            available += 1
        
        # Spread manipulation (look at spread changes)
        spread_zscore = prices.get('spread_zscore', None)
        spread_velocity = prices.get('spread_compression_velocity', 0)
        
        if spread_zscore is not None:
            # Rapid spread changes indicate MM activity
            spread_manip = abs(spread_zscore) * 0.3 + abs(spread_velocity) * 0.5
            components['spread_manipulation'] = min(1.0, spread_manip)
            available += 1
        
        # Quote stuffing (high add/cancel activity)
        add_cancel = orderbook.get('add_cancel_ratio', None)
        if add_cancel is not None:
            # Low ratio = lots of cancellations = potential quote stuffing
            quote_stuff = max(0, 1 - add_cancel) if add_cancel < 1 else 0
            components['quote_stuffing'] = quote_stuff
            available += 1
        
        # Inventory rotation signal (depth imbalance changes)
        depth_5 = orderbook.get('depth_imbalance_5', 0)
        depth_10 = orderbook.get('depth_imbalance_10', 0)
        
        if depth_5 != 0 or depth_10 != 0:
            # Imbalance divergence suggests inventory management
            imbalance_change = abs(depth_5 - depth_10)
            components['inventory_signal'] = min(1.0, imbalance_change * 2)
            available += 1
        
        if available < 2:
            return None
        
        # Calculate weighted score
        value = 0.0
        total_weight = 0.0
        for comp_name, comp_value in components.items():
            weight = self._market_maker_weights.get(comp_name, 0.25)
            value += comp_value * weight
            total_weight += weight
        
        if total_weight > 0:
            value /= total_weight
        
        confidence = available / 4
        
        # Determine activity type
        if value > 0.7:
            activity_type = 'aggressive'
        elif value > 0.3:
            activity_type = 'passive'
        else:
            activity_type = 'neutral'
        
        # Determine inventory bias
        if depth_5 > 0.2:
            inventory_bias = 'long'
        elif depth_5 < -0.2:
            inventory_bias = 'short'
        else:
            inventory_bias = 'balanced'
        
        self._market_maker_buffer.append(value)
        if len(self._market_maker_buffer) > self.window_size:
            self._market_maker_buffer.pop(0)
        
        return {
            'value': min(1.0, value),
            'confidence': confidence,
            'components': components,
            'activity_type': activity_type,
            'inventory_bias': inventory_bias,
        }
    
    def _calculate_liquidation_cascade_risk(self) -> Optional[Dict[str, Any]]:
        """
        Calculate Liquidation Cascade Risk.
        
        Predicts probability of cascading liquidations.
        
        Components:
        - Liquidation cluster size/activity
        - Leverage stress index
        - Thin liquidity factor
        - Funding pressure
        
        Returns:
            {
                'value': 0.0-1.0 (probability),
                'confidence': 0.0-1.0,
                'components': {...},
                'severity': 'LOW'|'MEDIUM'|'HIGH'|'CRITICAL',
                'direction': 'long'|'short'|'both',
                'estimated_trigger_move_pct': float
            }
        """
        liquidations = self._feature_cache.get('liquidations', {})
        oi = self._feature_cache.get('oi', {})
        orderbook = self._feature_cache.get('orderbook', {})
        funding = self._feature_cache.get('funding', {})
        
        components = {}
        available = 0
        
        # Liquidation cluster activity
        cluster_detected = liquidations.get('liquidation_cluster_detected', False)
        cascade_prob = liquidations.get('cascade_probability', 0)
        cascade_accel = liquidations.get('cascade_acceleration', 0)
        
        if cascade_prob > 0 or cluster_detected:
            cluster_score = cascade_prob * 0.6 + (0.3 if cluster_detected else 0) + min(0.4, cascade_accel)
            components['liquidation_cluster'] = min(1.0, cluster_score)
            available += 1
        
        # Leverage stress
        leverage_idx = oi.get('leverage_index', 1.0)
        leverage_zscore = oi.get('leverage_zscore', 0)
        cascade_risk_oi = oi.get('liquidation_cascade_risk', 0)
        
        if leverage_idx > 1 or leverage_zscore != 0:
            leverage_stress = (leverage_idx - 1) * 0.5 + abs(leverage_zscore) * 0.3 + cascade_risk_oi * 0.5
            components['leverage_stress'] = min(1.0, leverage_stress)
            available += 1
        
        # Thin liquidity factor
        liquidity_persistence = orderbook.get('liquidity_persistence_score', 0.5)
        depth_imb = abs(orderbook.get('depth_imbalance_10', 0))
        
        if liquidity_persistence < 0.5 or depth_imb > 0.3:
            thin_liq = (1 - liquidity_persistence) * 0.5 + depth_imb
            components['thin_liquidity'] = min(1.0, thin_liq)
            available += 1
        
        # Funding pressure
        funding_zscore = funding.get('funding_zscore', 0)
        funding_extreme = funding.get('funding_regime', 'neutral')
        
        if abs(funding_zscore) > 1 or funding_extreme != 'neutral':
            fund_pressure = abs(funding_zscore) / 3 + (0.3 if funding_extreme != 'neutral' else 0)
            components['funding_pressure'] = min(1.0, fund_pressure)
            available += 1
        
        if available < 2:
            return None
        
        # Calculate weighted score
        value = 0.0
        total_weight = 0.0
        for comp_name, comp_value in components.items():
            weight = self._cascade_weights.get(comp_name, 0.25)
            value += comp_value * weight
            total_weight += weight
        
        if total_weight > 0:
            value /= total_weight
        
        confidence = available / 4
        
        # Determine severity
        if value >= 0.8:
            severity = 'CRITICAL'
        elif value >= 0.6:
            severity = 'HIGH'
        elif value >= 0.4:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'
        
        # Determine direction
        liq_imbalance = liquidations.get('liquidation_imbalance', 0)
        if liq_imbalance > 0.3:
            direction = 'long'  # More long liquidations = cascade on longs
        elif liq_imbalance < -0.3:
            direction = 'short'
        else:
            direction = 'both'
        
        # Estimate trigger move
        estimated_trigger = value * 5  # Up to 5% move to trigger cascade
        
        self._cascade_risk_buffer.append(value)
        if len(self._cascade_risk_buffer) > self.window_size:
            self._cascade_risk_buffer.pop(0)
        
        return {
            'value': min(1.0, value),
            'confidence': confidence,
            'components': components,
            'severity': severity,
            'direction': direction,
            'estimated_trigger_move_pct': estimated_trigger,
        }
    
    def _detect_institutional_phase(self) -> Optional[Dict[str, Any]]:
        """
        Detect Institutional Accumulation/Distribution Phase.
        
        Identifies when large players are building or unwinding positions.
        
        Components:
        - Whale activity patterns
        - Absorption patterns
        - OI changes with price
        - VWAP deviation
        
        Returns:
            {
                'value': 0.0-1.0 (intensity),
                'confidence': 0.0-1.0,
                'components': {...},
                'phase': 'ACCUMULATION'|'DISTRIBUTION'|'NEUTRAL',
                'intensity': 0.0-1.0,
                'price_target_direction': 'higher'|'lower'|'neutral'
            }
        """
        trades = self._feature_cache.get('trades', {})
        orderbook = self._feature_cache.get('orderbook', {})
        oi = self._feature_cache.get('oi', {})
        prices = self._feature_cache.get('prices', {})
        
        components = {}
        available = 0
        accumulation_score = 0.0
        distribution_score = 0.0
        
        # Whale activity patterns
        whale_detected = trades.get('whale_trade_detected', False)
        whale_ratio = trades.get('whale_volume_ratio', 0)
        iceberg = trades.get('iceberg_detected', False)
        
        if whale_detected or whale_ratio > 0.1 or iceberg:
            whale_score = whale_ratio * 2 + (0.3 if whale_detected else 0) + (0.4 if iceberg else 0)
            components['whale_activity'] = min(1.0, whale_score)
            available += 1
            
            # Determine direction from CVD
            cvd_slope = trades.get('cvd_slope', 0)
            if cvd_slope > 0:
                accumulation_score += whale_score * 0.5
            else:
                distribution_score += whale_score * 0.5
        
        # Absorption patterns
        absorption = orderbook.get('absorption_ratio', 0)
        
        if absorption > 0:
            components['absorption_pattern'] = min(1.0, absorption)
            available += 1
            
            # High absorption with stable price = accumulation
            depth_imb = orderbook.get('depth_imbalance_5', 0)
            if depth_imb > 0.1:
                accumulation_score += absorption * 0.4
            elif depth_imb < -0.1:
                distribution_score += absorption * 0.4
        
        # OI changes (position building)
        oi_delta_pct = oi.get('oi_delta_pct', 0)
        position_intent = oi.get('position_intent', 'neutral')
        
        if abs(oi_delta_pct) > 0.5:
            oi_score = min(1.0, abs(oi_delta_pct) / 5)
            components['oi_building'] = oi_score
            available += 1
            
            if position_intent in ('long_accumulation', 'short_covering'):
                accumulation_score += oi_score * 0.4
            elif position_intent in ('short_accumulation', 'long_liquidation'):
                distribution_score += oi_score * 0.4
        
        # VWAP deviation
        vwap_dev = prices.get('vwap_deviation', None)
        if vwap_dev is not None:
            vwap_score = min(1.0, abs(vwap_dev) * 10)
            components['vwap_signal'] = vwap_score
            available += 1
            
            if vwap_dev < 0:  # Price below VWAP = potential accumulation
                accumulation_score += vwap_score * 0.3
            else:
                distribution_score += vwap_score * 0.3
        
        if available < 2:
            return None
        
        # Determine phase
        total_score = accumulation_score + distribution_score
        if total_score == 0:
            phase = 'NEUTRAL'
            intensity = 0.0
        elif accumulation_score > distribution_score * 1.5:
            phase = 'ACCUMULATION'
            intensity = accumulation_score / (total_score + 0.001)
        elif distribution_score > accumulation_score * 1.5:
            phase = 'DISTRIBUTION'
            intensity = distribution_score / (total_score + 0.001)
        else:
            phase = 'NEUTRAL'
            intensity = 0.3
        
        confidence = available / 4
        value = max(accumulation_score, distribution_score)
        
        # Price target direction
        if phase == 'ACCUMULATION':
            price_target = 'higher'
        elif phase == 'DISTRIBUTION':
            price_target = 'lower'
        else:
            price_target = 'neutral'
        
        self._institutional_phase_buffer.append(phase)
        if len(self._institutional_phase_buffer) > self.window_size:
            self._institutional_phase_buffer.pop(0)
        
        return {
            'value': min(1.0, value),
            'confidence': confidence,
            'components': components,
            'phase': phase,
            'intensity': intensity,
            'price_target_direction': price_target,
        }
    
    def _predict_volatility_breakout(self) -> Optional[Dict[str, Any]]:
        """
        Predict Volatility Breakout.
        
        Identifies conditions for imminent volatility expansion.
        
        Components:
        - Volatility compression
        - Liquidity thinning
        - Funding stress
        - Trade concentration
        
        Returns:
            {
                'value': 0.0-1.0 (probability),
                'confidence': 0.0-1.0,
                'components': {...},
                'direction': 'UP'|'DOWN'|'NEUTRAL',
                'expected_magnitude_pct': float,
                'timeframe_hours': int
            }
        """
        ticker = self._feature_cache.get('ticker', {})
        orderbook = self._feature_cache.get('orderbook', {})
        trades = self._feature_cache.get('trades', {})
        funding = self._feature_cache.get('funding', {})
        oi = self._feature_cache.get('oi', {})
        
        components = {}
        available = 0
        direction_score = 0.0  # Positive = up, negative = down
        
        # Volatility compression
        vol_compression = ticker.get('volatility_compression_idx', 0)
        range_vs_atr = ticker.get('range_vs_atr', 1.0)
        
        if vol_compression > 0 or range_vs_atr < 0.8:
            comp_score = vol_compression + max(0, (1 - range_vs_atr))
            components['volatility_compression'] = min(1.0, comp_score)
            available += 1
        
        # Liquidity thinning
        liq_gradient = orderbook.get('liquidity_gradient', 1.0)
        liq_persistence = orderbook.get('liquidity_persistence_score', 0.5)
        
        if liq_gradient > 1.2 or liq_persistence < 0.4:
            thin_score = (liq_gradient - 1) * 0.5 + (1 - liq_persistence) * 0.5
            components['liquidity_thinning'] = min(1.0, thin_score)
            available += 1
        
        # Funding stress
        funding_zscore = funding.get('funding_zscore', 0)
        funding_momentum = funding.get('funding_momentum', 0)
        
        if abs(funding_zscore) > 1:
            fund_stress = abs(funding_zscore) / 3
            components['funding_stress'] = min(1.0, fund_stress)
            available += 1
            
            # Funding direction affects breakout direction
            direction_score += -funding_zscore * 0.3  # High funding = short squeeze potential
        
        # Trade concentration
        trade_clustering = trades.get('trade_clustering_index', 0)
        cvd_slope = trades.get('cvd_slope', 0)
        aggressive_delta = trades.get('aggressive_delta', 0)
        
        if trade_clustering > 0.5:
            components['trade_concentration'] = min(1.0, trade_clustering)
            available += 1
            
            # Aggressive flow suggests direction
            direction_score += cvd_slope * 2 + (aggressive_delta * 0.5)
        
        if available < 2:
            return None
        
        # Calculate breakout probability
        value = sum(components.values()) / len(components)
        confidence = available / 4
        
        # Determine direction
        oi_delta = oi.get('oi_delta_pct', 0)
        direction_score += oi_delta * 0.3
        
        if direction_score > 0.3:
            direction = 'UP'
        elif direction_score < -0.3:
            direction = 'DOWN'
        else:
            direction = 'NEUTRAL'
        
        # Estimate magnitude (based on compression and thin liquidity)
        expected_magnitude = value * 3  # Up to 3% move
        
        # Timeframe estimate (higher compression = sooner breakout)
        if value > 0.7:
            timeframe = 4
        elif value > 0.5:
            timeframe = 12
        else:
            timeframe = 24
        
        self._volatility_breakout_buffer.append(value)
        if len(self._volatility_breakout_buffer) > self.window_size:
            self._volatility_breakout_buffer.pop(0)
        
        return {
            'value': min(1.0, value),
            'confidence': confidence,
            'components': components,
            'direction': direction,
            'expected_magnitude_pct': expected_magnitude,
            'timeframe_hours': timeframe,
        }
    
    def _calculate_mean_reversion_signal(self) -> Optional[Dict[str, Any]]:
        """
        Calculate Mean Reversion Signal.
        
        Identifies oversold/overbought extremes for potential reversion.
        
        Components:
        - Price extremity (z-score)
        - Funding extremes
        - Basis dislocation
        - CVD divergence
        
        Returns:
            {
                'value': 0.0-1.0 (strength),
                'confidence': 0.0-1.0,
                'components': {...},
                'signal': 'OVERSOLD'|'OVERBOUGHT'|'NEUTRAL',
                'strength': 0.0-1.0,
                'expected_reversion_pct': float
            }
        """
        prices = self._feature_cache.get('prices', {})
        funding = self._feature_cache.get('funding', {})
        mark_prices = self._feature_cache.get('mark_prices', {})
        trades = self._feature_cache.get('trades', {})
        
        components = {}
        available = 0
        oversold_score = 0.0
        overbought_score = 0.0
        
        # Price extremity
        mid_zscore = prices.get('mid_zscore', None)
        hurst = prices.get('hurst_exponent', 0.5)
        
        if mid_zscore is not None and abs(mid_zscore) > 1.5:
            extremity = min(1.0, abs(mid_zscore) / 3)
            components['price_extremity'] = extremity
            available += 1
            
            if mid_zscore < -1.5:
                oversold_score += extremity * 0.5
            elif mid_zscore > 1.5:
                overbought_score += extremity * 0.5
            
            # Hurst < 0.5 suggests mean reversion
            if hurst < 0.5:
                components['mean_reverting_tendency'] = 1 - hurst * 2
        
        # Funding extremes
        funding_zscore = funding.get('funding_zscore', 0)
        funding_percentile = funding.get('funding_percentile', 50)
        
        if abs(funding_zscore) > 2 or funding_percentile > 90 or funding_percentile < 10:
            fund_extreme = max(abs(funding_zscore) / 3, abs(funding_percentile - 50) / 50)
            components['funding_extreme'] = min(1.0, fund_extreme)
            available += 1
            
            if funding_zscore > 2 or funding_percentile > 90:
                overbought_score += fund_extreme * 0.3
            elif funding_zscore < -2 or funding_percentile < 10:
                oversold_score += fund_extreme * 0.3
        
        # Basis dislocation
        basis_zscore = mark_prices.get('basis_zscore', 0)
        dislocation = mark_prices.get('dislocation_detected', False)
        
        if abs(basis_zscore) > 2 or dislocation:
            basis_extreme = max(abs(basis_zscore) / 3, 0.5 if dislocation else 0)
            components['basis_dislocation'] = min(1.0, basis_extreme)
            available += 1
            
            if basis_zscore > 2:
                overbought_score += basis_extreme * 0.3
            elif basis_zscore < -2:
                oversold_score += basis_extreme * 0.3
        
        # CVD divergence
        cvd_divergence = trades.get('cvd_price_divergence', 0)
        
        if abs(cvd_divergence) > 0.3:
            components['cvd_divergence'] = min(1.0, abs(cvd_divergence))
            available += 1
            
            # Positive divergence = price up but CVD down = overbought
            if cvd_divergence > 0.3:
                overbought_score += abs(cvd_divergence) * 0.3
            elif cvd_divergence < -0.3:
                oversold_score += abs(cvd_divergence) * 0.3
        
        if available < 2:
            return None
        
        # Determine signal
        if oversold_score > overbought_score * 1.3:
            signal = 'OVERSOLD'
            strength = oversold_score / (oversold_score + overbought_score + 0.001)
        elif overbought_score > oversold_score * 1.3:
            signal = 'OVERBOUGHT'
            strength = overbought_score / (oversold_score + overbought_score + 0.001)
        else:
            signal = 'NEUTRAL'
            strength = 0.0
        
        value = max(oversold_score, overbought_score)
        confidence = available / 4
        
        # Expected reversion
        expected_reversion = value * 2  # Up to 2% reversion
        
        self._mean_reversion_buffer.append(value)
        if len(self._mean_reversion_buffer) > self.window_size:
            self._mean_reversion_buffer.pop(0)
        
        return {
            'value': min(1.0, value),
            'confidence': confidence,
            'components': components,
            'signal': signal,
            'strength': strength,
            'expected_reversion_pct': expected_reversion,
        }
    
    def _detect_momentum_exhaustion(self) -> Optional[Dict[str, Any]]:
        """
        Detect Momentum Exhaustion.
        
        Identifies trend exhaustion before reversal.
        
        Components:
        - CVD momentum loss
        - OI-price divergence
        - Liquidation exhaustion
        - Funding reversal
        
        Returns:
            {
                'value': 0.0-1.0 (exhaustion level),
                'confidence': 0.0-1.0,
                'components': {...},
                'exhaustion': bool,
                'trend_direction': 'bullish'|'bearish',
                'reversal_probability': 0.0-1.0
            }
        """
        trades = self._feature_cache.get('trades', {})
        oi = self._feature_cache.get('oi', {})
        liquidations = self._feature_cache.get('liquidations', {})
        funding = self._feature_cache.get('funding', {})
        
        components = {}
        available = 0
        bullish_exhaustion = 0.0
        bearish_exhaustion = 0.0
        
        # CVD momentum loss
        cvd_slope = trades.get('cvd_slope', 0)
        aggressive_delta = trades.get('aggressive_delta', 0)
        
        # Check for weakening CVD
        cvd_weakening = 0
        if len(self._momentum_buffer) > 5:
            recent_avg = np.mean(self._momentum_buffer[-5:])
            older_avg = np.mean(self._momentum_buffer[-10:-5]) if len(self._momentum_buffer) > 10 else recent_avg
            cvd_weakening = older_avg - recent_avg
        
        if abs(cvd_slope) < 0.01 or cvd_weakening > 0.1:
            momentum_loss = max(0.1 - abs(cvd_slope), 0) * 10 + cvd_weakening
            components['cvd_momentum_loss'] = min(1.0, momentum_loss)
            available += 1
            
            # Determine which trend is exhausting
            if aggressive_delta > 0:
                bullish_exhaustion += momentum_loss * 0.4
            else:
                bearish_exhaustion += momentum_loss * 0.4
        
        # OI-price divergence
        oi_price_corr = oi.get('oi_price_correlation', 0)
        position_intent = oi.get('position_intent', 'neutral')
        
        # Divergence = OI and price moving in different directions
        if abs(oi_price_corr) < 0.3:
            divergence_score = 1 - abs(oi_price_corr)
            components['oi_divergence'] = min(1.0, divergence_score)
            available += 1
            
            if position_intent in ('long_liquidation', 'short_covering'):
                bullish_exhaustion += divergence_score * 0.3
            elif position_intent in ('short_accumulation', 'long_accumulation'):
                bearish_exhaustion += divergence_score * 0.3
        
        # Liquidation exhaustion
        liq_exhaustion = liquidations.get('exhaustion_signal', False)
        liq_imbalance = liquidations.get('liquidation_imbalance', 0)
        
        if liq_exhaustion:
            components['liquidation_exhaustion'] = 0.8
            available += 1
            
            if liq_imbalance > 0.3:  # More longs liquidated
                bearish_exhaustion += 0.5  # Bearish trend exhausting longs
            elif liq_imbalance < -0.3:
                bullish_exhaustion += 0.5
        
        # Funding reversal
        funding_momentum = funding.get('funding_momentum', 0)
        funding_zscore = funding.get('funding_zscore', 0)
        
        # Funding momentum reversing from extremes
        if abs(funding_zscore) > 2 and abs(funding_momentum) > 0:
            if funding_zscore > 2 and funding_momentum < 0:  # High funding dropping
                reversal_score = 0.7
                bullish_exhaustion += reversal_score * 0.3
            elif funding_zscore < -2 and funding_momentum > 0:  # Low funding rising
                reversal_score = 0.7
                bearish_exhaustion += reversal_score * 0.3
            else:
                reversal_score = 0.3
            
            components['funding_reversal'] = reversal_score
            available += 1
        
        if available < 2:
            return None
        
        # Calculate overall exhaustion
        value = max(bullish_exhaustion, bearish_exhaustion)
        confidence = available / 4
        exhaustion_detected = value > 0.5
        
        # Determine trend direction being exhausted
        if bullish_exhaustion > bearish_exhaustion:
            trend_direction = 'bullish'
        else:
            trend_direction = 'bearish'
        
        # Reversal probability
        reversal_prob = value * 0.8  # Scale to max 80%
        
        self._exhaustion_buffer.append(value)
        if len(self._exhaustion_buffer) > self.window_size:
            self._exhaustion_buffer.pop(0)
        
        return {
            'value': min(1.0, value),
            'confidence': confidence,
            'components': components,
            'exhaustion': exhaustion_detected,
            'trend_direction': trend_direction,
            'reversal_probability': reversal_prob,
        }
    
    def _calculate_smart_money_flow(self) -> Optional[Dict[str, Any]]:
        """
        Calculate Smart Money Flow Direction.
        
        Determines institutional order flow direction.
        
        Components:
        - Whale side bias
        - Passive vs aggressive flow
        - Depth imbalance trend
        - OI direction correlation
        
        Returns:
            {
                'value': -1.0 to 1.0 (negative=selling, positive=buying),
                'confidence': 0.0-1.0,
                'components': {...},
                'direction': 'BUYING'|'SELLING'|'NEUTRAL',
                'strength': 0.0-1.0,
                'volume_estimate_btc': float
            }
        """
        trades = self._feature_cache.get('trades', {})
        orderbook = self._feature_cache.get('orderbook', {})
        oi = self._feature_cache.get('oi', {})
        ticker = self._feature_cache.get('ticker', {})
        
        components = {}
        available = 0
        flow_direction = 0.0  # -1 to 1
        
        # Whale side bias
        whale_detected = trades.get('whale_trade_detected', False)
        whale_ratio = trades.get('whale_volume_ratio', 0)
        buy_pressure = trades.get('buy_pressure', 0.5)
        sell_pressure = trades.get('sell_pressure', 0.5)
        
        if whale_detected or whale_ratio > 0.05:
            # Determine whale direction from buy/sell pressure
            whale_direction = (buy_pressure - sell_pressure) * whale_ratio * 5
            components['whale_bias'] = whale_direction
            flow_direction += whale_direction * 0.4
            available += 1
        
        # Passive vs aggressive flow
        aggressive_ratio = trades.get('aggressive_ratio', 0.5)
        cvd_slope = trades.get('cvd_slope', 0)
        
        if aggressive_ratio != 0.5 or cvd_slope != 0:
            # Aggressive buyers = positive, aggressive sellers = negative
            passive_aggressive = (aggressive_ratio - 0.5) * 2 + cvd_slope * 10
            components['flow_aggressiveness'] = np.clip(passive_aggressive, -1, 1)
            flow_direction += passive_aggressive * 0.3
            available += 1
        
        # Depth imbalance trend
        depth_5 = orderbook.get('depth_imbalance_5', 0)
        depth_10 = orderbook.get('depth_imbalance_10', 0)
        
        if depth_5 != 0 or depth_10 != 0:
            depth_signal = (depth_5 + depth_10) / 2
            components['depth_bias'] = depth_signal
            flow_direction += depth_signal * 0.2
            available += 1
        
        # OI direction
        oi_delta_pct = oi.get('oi_delta_pct', 0)
        long_short_ratio = oi.get('long_short_ratio', 1.0)
        
        if abs(oi_delta_pct) > 0.5 or abs(long_short_ratio - 1) > 0.1:
            oi_signal = np.sign(oi_delta_pct) * min(1, abs(oi_delta_pct) / 5)
            oi_signal += (long_short_ratio - 1) * 0.5
            components['oi_signal'] = np.clip(oi_signal, -1, 1)
            flow_direction += oi_signal * 0.2
            available += 1
        
        if available < 2:
            return None
        
        # Normalize flow direction
        flow_direction = np.clip(flow_direction, -1, 1)
        
        # Determine direction
        if flow_direction > 0.2:
            direction = 'BUYING'
        elif flow_direction < -0.2:
            direction = 'SELLING'
        else:
            direction = 'NEUTRAL'
        
        strength = abs(flow_direction)
        confidence = available / 4
        
        # Volume estimate (based on whale activity and volume)
        volume_24h = ticker.get('volume', 0)
        institutional_idx = ticker.get('institutional_interest_idx', 0.5)
        volume_estimate = volume_24h * institutional_idx * whale_ratio if whale_ratio > 0 else 0
        
        self._smart_flow_buffer.append(flow_direction)
        if len(self._smart_flow_buffer) > self.window_size:
            self._smart_flow_buffer.pop(0)
        
        return {
            'value': flow_direction,
            'confidence': confidence,
            'components': components,
            'direction': direction,
            'strength': strength,
            'volume_estimate_btc': volume_estimate,
        }
    
    def _detect_arbitrage_opportunity(self) -> Optional[Dict[str, Any]]:
        """
        Detect Arbitrage Opportunity.
        
        Identifies funding/basis arbitrage setups.
        
        Components:
        - Funding rate yield
        - Basis opportunity
        - Execution feasibility
        
        Returns:
            {
                'value': 0.0-1.0 (opportunity score),
                'confidence': 0.0-1.0,
                'components': {...},
                'opportunity': bool,
                'type': 'funding'|'basis'|'combined'|'none',
                'expected_return_pct_annual': float,
                'risk_level': 'LOW'|'MEDIUM'|'HIGH'
            }
        """
        funding = self._feature_cache.get('funding', {})
        mark_prices = self._feature_cache.get('mark_prices', {})
        ticker = self._feature_cache.get('ticker', {})
        orderbook = self._feature_cache.get('orderbook', {})
        
        components = {}
        available = 0
        funding_arb = 0.0
        basis_arb = 0.0
        
        # Funding rate opportunity
        funding_rate = funding.get('funding_rate', 0)
        carry_yield = funding.get('funding_carry_yield', 0)
        
        # Annualized funding yield threshold (> 10% is attractive)
        if abs(carry_yield) > 0.1:
            funding_score = min(1.0, abs(carry_yield) / 0.5)  # Cap at 50% yield
            components['funding_yield'] = funding_score
            funding_arb = abs(carry_yield)
            available += 1
        
        # Basis opportunity
        annualized_basis = mark_prices.get('annualized_basis', 0)
        basis_zscore = mark_prices.get('basis_zscore', 0)
        
        # High basis = cash-and-carry opportunity
        if abs(annualized_basis) > 0.1 or abs(basis_zscore) > 2:
            basis_score = min(1.0, abs(annualized_basis) / 0.5)
            components['basis_opportunity'] = basis_score
            basis_arb = abs(annualized_basis)
            available += 1
        
        # Execution feasibility (liquidity check)
        spread_bps = 0
        bid_ask_spread = orderbook.get('spread', 0)
        mid_price = mark_prices.get('mark_price', 1)
        if mid_price > 0:
            spread_bps = (bid_ask_spread / mid_price) * 10000
        
        volume = ticker.get('volume', 0)
        
        execution_score = 1.0
        if spread_bps > 5:  # More than 5 bps spread hurts arb
            execution_score -= (spread_bps - 5) * 0.1
        if volume < 100:  # Low volume
            execution_score -= 0.3
        
        execution_score = max(0, execution_score)
        components['execution_feasibility'] = execution_score
        available += 1
        
        if available < 2:
            return None
        
        # Calculate opportunity
        total_arb = funding_arb + basis_arb
        value = min(1.0, total_arb * 2) * execution_score
        
        # Determine type
        if funding_arb > basis_arb and funding_arb > 0.1:
            arb_type = 'funding'
            expected_return = funding_arb
        elif basis_arb > funding_arb and basis_arb > 0.1:
            arb_type = 'basis'
            expected_return = basis_arb
        elif funding_arb > 0.05 and basis_arb > 0.05:
            arb_type = 'combined'
            expected_return = funding_arb + basis_arb
        else:
            arb_type = 'none'
            expected_return = 0
        
        opportunity = value > 0.3 and arb_type != 'none'
        confidence = available / 3
        
        # Risk level
        if execution_score > 0.8 and abs(basis_zscore) < 3:
            risk_level = 'LOW'
        elif execution_score > 0.5:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'
        
        self._arbitrage_buffer.append(value)
        if len(self._arbitrage_buffer) > self.window_size:
            self._arbitrage_buffer.pop(0)
        
        return {
            'value': min(1.0, value),
            'confidence': confidence,
            'components': components,
            'opportunity': opportunity,
            'type': arb_type,
            'expected_return_pct_annual': expected_return * 100,
            'risk_level': risk_level,
        }
    
    def _predict_regime_transition(self) -> Optional[Dict[str, Any]]:
        """
        Predict Market Regime Transition.
        
        Identifies probability of regime changes (bull/bear/range).
        
        Components:
        - Volatility regime change
        - Flow toxicity spike
        - Funding regime divergence
        - Leverage cycle position
        
        Returns:
            {
                'value': 0.0-1.0 (transition probability),
                'confidence': 0.0-1.0,
                'components': {...},
                'current_regime': 'BULL'|'BEAR'|'RANGING',
                'predicted_regime': 'BULL'|'BEAR'|'RANGING',
                'transition_probability': 0.0-1.0,
                'timeframe_hours': int
            }
        """
        ticker = self._feature_cache.get('ticker', {})
        trades = self._feature_cache.get('trades', {})
        funding = self._feature_cache.get('funding', {})
        oi = self._feature_cache.get('oi', {})
        prices = self._feature_cache.get('prices', {})
        
        components = {}
        available = 0
        transition_score = 0.0
        
        # Current regime detection
        price_change = ticker.get('price_change_pct', 0)
        volatility = ticker.get('realized_volatility', 0)
        market_strength = ticker.get('market_strength', 0.5)
        
        # Determine current regime
        if price_change > 3 and market_strength > 0.6:
            current_regime = 'BULL'
        elif price_change < -3 and market_strength < 0.4:
            current_regime = 'BEAR'
        else:
            current_regime = 'RANGING'
        
        # Volatility regime change
        vol_compression = ticker.get('volatility_compression_idx', 0)
        vol_expansion = ticker.get('volatility_expansion_idx', 0)
        
        if vol_compression > 0.5 or vol_expansion > 0.5:
            vol_change = max(vol_compression, vol_expansion)
            components['volatility_regime_change'] = vol_change
            transition_score += vol_change * 0.3
            available += 1
        
        # Flow toxicity spike
        flow_toxicity = trades.get('flow_toxicity', 0)
        
        # Check for toxicity changes
        toxicity_spike = 0
        if len(self._smart_money_buffer) > 5:
            recent_toxicity = flow_toxicity
            baseline = np.mean(self._smart_money_buffer[-10:]) if len(self._smart_money_buffer) > 10 else 0.3
            toxicity_spike = max(0, recent_toxicity - baseline)
        
        if toxicity_spike > 0.2 or flow_toxicity > 0.6:
            components['flow_toxicity_spike'] = min(1.0, toxicity_spike + flow_toxicity * 0.5)
            transition_score += components['flow_toxicity_spike'] * 0.3
            available += 1
        
        # Funding regime divergence
        funding_regime = funding.get('funding_regime', 'neutral')
        funding_zscore = funding.get('funding_zscore', 0)
        
        # Extreme funding often precedes regime change
        if abs(funding_zscore) > 2:
            fund_divergence = abs(funding_zscore) / 3
            components['funding_divergence'] = min(1.0, fund_divergence)
            transition_score += fund_divergence * 0.2
            available += 1
        
        # Leverage cycle position
        leverage_idx = oi.get('leverage_index', 1.0)
        crowding = oi.get('position_crowding_score', 0)
        
        # High leverage = cycle peak = regime change likely
        if leverage_idx > 1.5 or crowding > 0.7:
            leverage_extreme = (leverage_idx - 1) * 0.5 + crowding * 0.5
            components['leverage_cycle_extreme'] = min(1.0, leverage_extreme)
            transition_score += leverage_extreme * 0.2
            available += 1
        
        if available < 2:
            return None
        
        confidence = available / 4
        value = min(1.0, transition_score)
        
        # Predict next regime
        if current_regime == 'BULL':
            if transition_score > 0.5:
                predicted_regime = 'BEAR' if funding_zscore > 2 else 'RANGING'
            else:
                predicted_regime = 'BULL'
        elif current_regime == 'BEAR':
            if transition_score > 0.5:
                predicted_regime = 'BULL' if funding_zscore < -2 else 'RANGING'
            else:
                predicted_regime = 'BEAR'
        else:  # RANGING
            if transition_score > 0.5:
                predicted_regime = 'BULL' if market_strength > 0.5 else 'BEAR'
            else:
                predicted_regime = 'RANGING'
        
        # Timeframe based on volatility
        if vol_compression > 0.7:
            timeframe = 6  # Hours
        elif transition_score > 0.7:
            timeframe = 12
        else:
            timeframe = 24
        
        self._regime_buffer.append(current_regime)
        if len(self._regime_buffer) > self.window_size:
            self._regime_buffer.pop(0)
        
        return {
            'value': value,
            'confidence': confidence,
            'components': components,
            'current_regime': current_regime,
            'predicted_regime': predicted_regime,
            'transition_probability': value,
            'timeframe_hours': timeframe,
        }
    
    def _calculate_execution_quality(self) -> Optional[Dict[str, Any]]:
        """
        Calculate Execution Quality Score.
        
        Assesses optimal execution conditions.
        
        Components:
        - Liquidity score
        - Spread quality
        - Market impact estimate
        - Volume conditions
        
        Returns:
            {
                'value': 0.0-1.0 (execution quality),
                'confidence': 0.0-1.0,
                'components': {...},
                'quality': 'EXCELLENT'|'GOOD'|'FAIR'|'POOR',
                'slippage_estimate_bps': float,
                'max_size_no_impact_btc': float,
                'optimal_order_splits': int
            }
        """
        orderbook = self._feature_cache.get('orderbook', {})
        prices = self._feature_cache.get('prices', {})
        ticker = self._feature_cache.get('ticker', {})
        trades = self._feature_cache.get('trades', {})
        
        components = {}
        available = 0
        
        # Liquidity score
        bid_depth = orderbook.get('bid_depth_10', 0)
        ask_depth = orderbook.get('ask_depth_10', 0)
        total_depth = bid_depth + ask_depth
        liquidity_persistence = orderbook.get('liquidity_persistence_score', 0.5)
        
        if total_depth > 0:
            # Normalize depth (assuming typical depth)
            depth_score = min(1.0, total_depth / 100)  # Normalize to ~100 BTC depth
            liquidity_score = depth_score * 0.6 + liquidity_persistence * 0.4
            components['liquidity'] = liquidity_score
            available += 1
        
        # Spread quality
        spread_bps = prices.get('spread_bps', 10)
        spread_zscore = prices.get('spread_zscore', 0)
        
        # Lower spread = better execution
        spread_score = max(0, 1 - spread_bps / 20)  # 20 bps = score of 0
        if spread_zscore < 0:  # Compressed spread = better
            spread_score = min(1.0, spread_score + abs(spread_zscore) * 0.1)
        
        components['spread_quality'] = spread_score
        available += 1
        
        # Market impact estimate
        trade_clustering = trades.get('trade_clustering_index', 0.5)
        aggressive_ratio = trades.get('aggressive_ratio', 0.5)
        
        # High clustering = higher impact
        impact_score = 1 - (trade_clustering * 0.5 + abs(aggressive_ratio - 0.5))
        components['low_impact'] = max(0, impact_score)
        available += 1
        
        # Volume conditions
        volume = ticker.get('volume', 0)
        volume_percentile = ticker.get('relative_volume_percentile', 50)
        
        # Higher volume = better execution
        volume_score = min(1.0, volume_percentile / 100)
        components['volume_conditions'] = volume_score
        available += 1
        
        if available < 2:
            return None
        
        # Calculate overall score
        value = sum(components.values()) / len(components)
        confidence = available / 4
        
        # Determine quality tier
        if value > 0.8:
            quality = 'EXCELLENT'
        elif value > 0.6:
            quality = 'GOOD'
        elif value > 0.4:
            quality = 'FAIR'
        else:
            quality = 'POOR'
        
        # Slippage estimate
        base_slippage = spread_bps / 2
        impact_slippage = (1 - components.get('low_impact', 0.5)) * 10
        slippage_estimate = base_slippage + impact_slippage
        
        # Max size without impact
        if total_depth > 0:
            max_size = total_depth * value * 0.1  # 10% of depth at quality factor
        else:
            max_size = 0.5  # Default
        
        # Optimal splits (more splits for larger size or lower quality)
        if value > 0.8:
            optimal_splits = 1
        elif value > 0.6:
            optimal_splits = 3
        elif value > 0.4:
            optimal_splits = 5
        else:
            optimal_splits = 10
        
        self._execution_buffer.append(value)
        if len(self._execution_buffer) > self.window_size:
            self._execution_buffer.pop(0)
        
        return {
            'value': min(1.0, value),
            'confidence': confidence,
            'components': components,
            'quality': quality,
            'slippage_estimate_bps': slippage_estimate,
            'max_size_no_impact_btc': max_size,
            'optimal_order_splits': optimal_splits,
        }
    
    def get_signal_interpretation(self, signal_name: str, value: float) -> str:
        """
        Get human-readable interpretation of a signal value.
        
        Args:
            signal_name: Name of the signal
            value: Signal value (0-1)
        
        Returns:
            Interpretation string
        """
        interpretations = {
            'smart_money_index': {
                (0.0, 0.2): 'Low institutional activity',
                (0.2, 0.4): 'Some institutional presence',
                (0.4, 0.6): 'Moderate institutional activity',
                (0.6, 0.8): 'High institutional activity',
                (0.8, 1.0): 'Very high institutional activity - potential large move',
            },
            'squeeze_probability': {
                (0.0, 0.2): 'Low squeeze risk',
                (0.2, 0.4): 'Elevated squeeze risk',
                (0.4, 0.6): 'Moderate squeeze risk',
                (0.6, 0.8): 'High squeeze risk - caution',
                (0.8, 1.0): 'Extreme squeeze risk - imminent',
            },
            'stop_hunt_probability': {
                (0.0, 0.2): 'Clean price action',
                (0.2, 0.4): 'Some manipulation signs',
                (0.4, 0.6): 'Potential stop hunting',
                (0.6, 0.8): 'Likely stop hunting in progress',
                (0.8, 1.0): 'Active manipulation detected',
            },
            'momentum_quality': {
                (0.0, 0.2): 'Poor momentum - likely reversal',
                (0.2, 0.4): 'Weak momentum',
                (0.4, 0.6): 'Neutral momentum',
                (0.6, 0.8): 'Strong momentum',
                (0.8, 1.0): 'Excellent momentum - trend confirmed',
            },
            'composite_risk': {
                (0.0, 0.2): 'Low risk environment',
                (0.2, 0.4): 'Normal risk',
                (0.4, 0.6): 'Elevated risk',
                (0.6, 0.8): 'High risk - reduce exposure',
                (0.8, 1.0): 'Extreme risk - defensive positioning',
            },
            # Phase 3 signals
            'market_maker_activity': {
                (0.0, 0.2): 'Low market maker activity',
                (0.2, 0.4): 'Moderate MM presence',
                (0.4, 0.6): 'Active market making',
                (0.6, 0.8): 'High MM activity - potential inventory management',
                (0.8, 1.0): 'Aggressive MM activity - watch for manipulation',
            },
            'liquidation_cascade_risk': {
                (0.0, 0.2): 'Low cascade risk',
                (0.2, 0.4): 'Elevated cascade risk',
                (0.4, 0.6): 'Moderate cascade risk - monitor closely',
                (0.6, 0.8): 'High cascade risk - reduce leverage',
                (0.8, 1.0): 'Critical cascade risk - imminent',
            },
            'institutional_phase': {
                (0.0, 0.2): 'No clear institutional activity',
                (0.2, 0.4): 'Weak institutional signals',
                (0.4, 0.6): 'Moderate institutional activity',
                (0.6, 0.8): 'Clear institutional phase detected',
                (0.8, 1.0): 'Strong institutional accumulation/distribution',
            },
            'volatility_breakout': {
                (0.0, 0.2): 'Low breakout probability',
                (0.2, 0.4): 'Breakout possible',
                (0.4, 0.6): 'Moderate breakout probability',
                (0.6, 0.8): 'High breakout probability - prepare',
                (0.8, 1.0): 'Breakout imminent',
            },
            'mean_reversion_signal': {
                (0.0, 0.2): 'No reversion signal',
                (0.2, 0.4): 'Weak reversion setup',
                (0.4, 0.6): 'Moderate reversion opportunity',
                (0.6, 0.8): 'Strong reversion signal',
                (0.8, 1.0): 'Extreme reversion opportunity',
            },
            'momentum_exhaustion': {
                (0.0, 0.2): 'Strong momentum - trend intact',
                (0.2, 0.4): 'Slight momentum weakness',
                (0.4, 0.6): 'Momentum slowing',
                (0.6, 0.8): 'Significant exhaustion - reversal likely',
                (0.8, 1.0): 'Severe exhaustion - reversal imminent',
            },
            'arbitrage_opportunity': {
                (0.0, 0.2): 'No arbitrage opportunity',
                (0.2, 0.4): 'Marginal arbitrage',
                (0.4, 0.6): 'Moderate arbitrage opportunity',
                (0.6, 0.8): 'Good arbitrage opportunity',
                (0.8, 1.0): 'Excellent arbitrage opportunity',
            },
            'regime_transition': {
                (0.0, 0.2): 'Stable regime',
                (0.2, 0.4): 'Regime stable but watch',
                (0.4, 0.6): 'Potential regime change',
                (0.6, 0.8): 'Likely regime transition',
                (0.8, 1.0): 'Regime change imminent',
            },
            'execution_quality': {
                (0.0, 0.2): 'Poor execution conditions - avoid',
                (0.2, 0.4): 'Fair execution - use limit orders',
                (0.4, 0.6): 'Decent execution conditions',
                (0.6, 0.8): 'Good execution conditions',
                (0.8, 1.0): 'Excellent execution - optimal entry',
            },
        }
        
        signal_interp = interpretations.get(signal_name, {})
        for (low, high), text in signal_interp.items():
            if low <= value < high:
                return text
        
        return f"Signal value: {value:.2f}"
    
    def reset(self):
        """Reset all buffers and caches."""
        # Phase 1 buffers
        self._smart_money_buffer = []
        self._squeeze_buffer = []
        self._stop_hunt_buffer = []
        self._momentum_buffer = []
        self._risk_buffer = []
        
        # Phase 3 buffers
        self._market_maker_buffer = []
        self._cascade_risk_buffer = []
        self._institutional_phase_buffer = []
        self._volatility_breakout_buffer = []
        self._mean_reversion_buffer = []
        self._exhaustion_buffer = []
        self._smart_flow_buffer = []
        self._arbitrage_buffer = []
        self._regime_buffer = []
        self._execution_buffer = []
        
        for stream in self._feature_cache:
            self._feature_cache[stream] = {}
