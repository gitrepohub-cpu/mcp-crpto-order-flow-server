"""
🎯 Signal Aggregator - Phase 3 Week 3-4
=======================================

Aggregates, ranks, and resolves composite signals into actionable recommendations.

Components:
- SignalAggregator: Main aggregation engine
- SignalRanker: Weighted ranking algorithm
- ConflictResolver: Bullish/Bearish conflict detection
- RecommendationGenerator: Trade recommendation builder

Flow:
1. Collect all signals from CompositeSignalCalculator
2. Rank signals by weighted importance
3. Detect and resolve conflicting signals
4. Generate actionable trade recommendations
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from .composite import CompositeSignal, EnhancedSignal, CompositeSignalCalculator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class SignalDirection(Enum):
    """Signal directional bias."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    

class SignalCategory(Enum):
    """Signal category for grouping."""
    MOMENTUM = "momentum"
    FLOW = "flow"
    STRUCTURE = "structure"
    RISK = "risk"
    TIMING = "timing"
    EXECUTION = "execution"


class RecommendationStrength(Enum):
    """Recommendation conviction level."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    CONFLICTED = "conflicted"
    NO_ACTION = "no_action"


class ConflictSeverity(Enum):
    """Severity of signal conflicts."""
    NONE = "none"
    MINOR = "minor"          # < 25% disagreement
    MODERATE = "moderate"    # 25-50% disagreement
    SEVERE = "severe"        # > 50% disagreement
    CRITICAL = "critical"    # > 75% disagreement


# Signal metadata for categorization and direction mapping
SIGNAL_METADATA = {
    # Phase 1 Signals
    'smart_money_index': {
        'category': SignalCategory.FLOW,
        'direction_threshold': 0.6,  # > 0.6 = bullish
        'weight': 1.2,
        'description': 'Institutional activity level'
    },
    'squeeze_probability': {
        'category': SignalCategory.STRUCTURE,
        'direction_threshold': 0.5,  # context dependent
        'weight': 1.1,
        'description': 'Short/long squeeze likelihood'
    },
    'stop_hunt_probability': {
        'category': SignalCategory.RISK,
        'direction_threshold': 0.5,  # > 0.5 = bearish (manipulation)
        'weight': 0.9,
        'description': 'Stop hunt manipulation probability'
    },
    'momentum_quality': {
        'category': SignalCategory.MOMENTUM,
        'direction_threshold': 0.6,  # > 0.6 = bullish
        'weight': 1.3,
        'description': 'Trend sustainability'
    },
    'composite_risk': {
        'category': SignalCategory.RISK,
        'direction_threshold': 0.5,  # < 0.5 = bullish (low risk)
        'weight': 1.0,
        'inverted': True,  # Lower is better
        'description': 'Overall market risk'
    },
    
    # Phase 3 Signals
    'market_maker_activity': {
        'category': SignalCategory.FLOW,
        'direction_threshold': 0.5,  # Uses metadata for direction
        'weight': 1.1,
        'description': 'Market maker positioning'
    },
    'liquidation_cascade_risk': {
        'category': SignalCategory.RISK,
        'direction_threshold': 0.5,
        'weight': 1.4,  # High importance
        'inverted': True,
        'description': 'Cascade liquidation probability'
    },
    'institutional_phase': {
        'category': SignalCategory.FLOW,
        'direction_threshold': 0.5,  # Uses metadata
        'weight': 1.3,
        'description': 'Accumulation/Distribution phase'
    },
    'volatility_breakout': {
        'category': SignalCategory.TIMING,
        'direction_threshold': 0.5,  # Uses metadata
        'weight': 1.2,
        'description': 'Breakout probability and direction'
    },
    'mean_reversion': {
        'category': SignalCategory.TIMING,
        'direction_threshold': 0.5,  # Uses metadata
        'weight': 1.0,
        'description': 'Mean reversion signal'
    },
    'momentum_exhaustion': {
        'category': SignalCategory.MOMENTUM,
        'direction_threshold': 0.7,  # > 0.7 = exhausted (bearish for longs)
        'weight': 1.2,
        'inverted': True,
        'description': 'Trend exhaustion level'
    },
    'smart_money_flow': {
        'category': SignalCategory.FLOW,
        'direction_threshold': 0.0,  # Positive = bullish
        'weight': 1.4,
        'description': 'Institutional flow direction'
    },
    'arbitrage_opportunity': {
        'category': SignalCategory.EXECUTION,
        'direction_threshold': 0.5,
        'weight': 0.8,
        'description': 'Arbitrage opportunity'
    },
    'regime_transition': {
        'category': SignalCategory.STRUCTURE,
        'direction_threshold': 0.5,  # Uses metadata
        'weight': 1.1,
        'description': 'Market regime transition'
    },
    'execution_quality': {
        'category': SignalCategory.EXECUTION,
        'direction_threshold': 0.6,  # > 0.6 = good
        'weight': 0.9,
        'description': 'Execution conditions'
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RankedSignal:
    """A signal with its ranking score and direction."""
    name: str
    value: float
    confidence: float
    direction: SignalDirection
    category: SignalCategory
    rank_score: float
    weight: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'confidence': self.confidence,
            'direction': self.direction.value,
            'category': self.category.value,
            'rank_score': self.rank_score,
            'weight': self.weight,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata,
        }


@dataclass
class SignalConflict:
    """Detected conflict between signals."""
    bullish_signals: List[RankedSignal]
    bearish_signals: List[RankedSignal]
    severity: ConflictSeverity
    resolution: SignalDirection
    resolution_confidence: float
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'bullish_count': len(self.bullish_signals),
            'bearish_count': len(self.bearish_signals),
            'bullish_signals': [s.name for s in self.bullish_signals],
            'bearish_signals': [s.name for s in self.bearish_signals],
            'severity': self.severity.value,
            'resolution': self.resolution.value,
            'resolution_confidence': self.resolution_confidence,
            'explanation': self.explanation,
        }


@dataclass
class TradeRecommendation:
    """Actionable trade recommendation."""
    direction: SignalDirection
    strength: RecommendationStrength
    confidence: float
    entry_bias: str  # "aggressive", "conservative", "wait"
    
    # Supporting signals
    primary_signals: List[str]
    supporting_signals: List[str]
    conflicting_signals: List[str]
    
    # Risk assessment
    risk_level: str  # "low", "medium", "high", "extreme"
    risk_factors: List[str]
    
    # Timing
    urgency: str  # "immediate", "soon", "patient", "wait"
    timeframe: str  # "scalp", "intraday", "swing"
    
    # Metadata
    timestamp: datetime
    explanation: str
    alerts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'direction': self.direction.value,
            'strength': self.strength.value,
            'confidence': round(self.confidence, 4),
            'entry_bias': self.entry_bias,
            'primary_signals': self.primary_signals,
            'supporting_signals': self.supporting_signals,
            'conflicting_signals': self.conflicting_signals,
            'risk_level': self.risk_level,
            'risk_factors': self.risk_factors,
            'urgency': self.urgency,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'explanation': self.explanation,
            'alerts': self.alerts,
        }


@dataclass
class AggregatedIntelligence:
    """Complete aggregated market intelligence."""
    symbol: str
    timestamp: datetime
    
    # Overall assessment
    market_bias: SignalDirection
    bias_confidence: float
    
    # Ranked signals
    ranked_signals: List[RankedSignal]
    top_signals: List[RankedSignal]  # Top 5
    
    # Conflicts
    conflict: Optional[SignalConflict]
    
    # Recommendation
    recommendation: TradeRecommendation
    
    # Category summaries
    category_scores: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'market_bias': self.market_bias.value,
            'bias_confidence': round(self.bias_confidence, 4),
            'ranked_signals': [s.to_dict() for s in self.ranked_signals],
            'top_signals': [s.to_dict() for s in self.top_signals],
            'conflict': self.conflict.to_dict() if self.conflict else None,
            'recommendation': self.recommendation.to_dict(),
            'category_scores': {k: round(v, 4) for k, v in self.category_scores.items()},
        }


# =============================================================================
# SIGNAL AGGREGATOR
# =============================================================================

class SignalAggregator:
    """
    Aggregates, ranks, and resolves composite signals into actionable intelligence.
    
    Features:
    - Signal ranking with weighted importance
    - Conflict detection and resolution
    - Trade recommendation generation
    - Category-based analysis
    
    Usage:
        aggregator = SignalAggregator()
        
        # From composite calculator signals
        signals = composite_calc.calculate_all(features)
        
        # Aggregate and analyze
        intelligence = aggregator.aggregate(
            symbol="BTCUSDT",
            signals=signals,
            phase3_signals=composite_calc.get_phase3_signals()
        )
        
        # Get recommendation
        print(intelligence.recommendation.to_dict())
    """
    
    def __init__(
        self,
        recency_decay: float = 0.95,
        confidence_weight: float = 0.3,
        correlation_penalty: float = 0.1,
        conflict_threshold: float = 0.25,
    ):
        """
        Initialize signal aggregator.
        
        Args:
            recency_decay: Decay factor for older signals (0-1)
            confidence_weight: Weight given to signal confidence
            correlation_penalty: Penalty for correlated signals
            conflict_threshold: Threshold for conflict detection
        """
        self.recency_decay = recency_decay
        self.confidence_weight = confidence_weight
        self.correlation_penalty = correlation_penalty
        self.conflict_threshold = conflict_threshold
        
        # Signal history for recency weighting
        self._signal_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self._max_history = 100
        
        # Correlation groups (signals that tend to move together)
        self._correlation_groups = {
            'flow': {'smart_money_index', 'smart_money_flow', 'institutional_phase', 'market_maker_activity'},
            'momentum': {'momentum_quality', 'momentum_exhaustion', 'volatility_breakout'},
            'risk': {'composite_risk', 'liquidation_cascade_risk', 'stop_hunt_probability'},
            'timing': {'mean_reversion', 'volatility_breakout'},
            'structure': {'squeeze_probability', 'regime_transition'},
        }
        
        logger.info("SignalAggregator initialized")
    
    # =========================================================================
    # MAIN AGGREGATION
    # =========================================================================
    
    def aggregate(
        self,
        symbol: str,
        signals: Dict[str, CompositeSignal],
        phase3_signals: Optional[Dict[str, EnhancedSignal]] = None,
    ) -> AggregatedIntelligence:
        """
        Aggregate all signals into unified market intelligence.
        
        Args:
            symbol: Trading symbol
            signals: Phase 1 signals from CompositeSignalCalculator
            phase3_signals: Phase 3 enhanced signals (dict from get_phase3_signals())
            
        Returns:
            AggregatedIntelligence with ranked signals, conflicts, and recommendations
        """
        timestamp = datetime.now(timezone.utc)
        
        # Step 1: Convert all signals to RankedSignal format
        ranked_signals = self._rank_signals(signals, phase3_signals, timestamp)
        
        # Step 2: Detect and resolve conflicts
        conflict = self._detect_conflicts(ranked_signals)
        
        # Step 3: Calculate overall market bias
        market_bias, bias_confidence = self._calculate_market_bias(ranked_signals, conflict)
        
        # Step 4: Calculate category scores
        category_scores = self._calculate_category_scores(ranked_signals)
        
        # Step 5: Generate trade recommendation
        recommendation = self._generate_recommendation(
            ranked_signals=ranked_signals,
            conflict=conflict,
            market_bias=market_bias,
            bias_confidence=bias_confidence,
            category_scores=category_scores,
            timestamp=timestamp,
        )
        
        # Sort by rank score
        ranked_signals.sort(key=lambda s: s.rank_score, reverse=True)
        top_signals = ranked_signals[:5]
        
        return AggregatedIntelligence(
            symbol=symbol,
            timestamp=timestamp,
            market_bias=market_bias,
            bias_confidence=bias_confidence,
            ranked_signals=ranked_signals,
            top_signals=top_signals,
            conflict=conflict,
            recommendation=recommendation,
            category_scores=category_scores,
        )
    
    # =========================================================================
    # SIGNAL RANKING
    # =========================================================================
    
    def _rank_signals(
        self,
        signals: Dict[str, CompositeSignal],
        phase3_signals: Optional[Dict[str, EnhancedSignal]],
        timestamp: datetime,
    ) -> List[RankedSignal]:
        """Rank all signals by weighted importance."""
        ranked = []
        
        # Process Phase 1 signals (filter out EnhancedSignals that may be in signals dict)
        for name, signal in signals.items():
            # Skip EnhancedSignals - they'll be processed separately
            if isinstance(signal, EnhancedSignal):
                continue
            if not isinstance(signal, CompositeSignal):
                continue
                
            ranked_signal = self._create_ranked_signal(
                name=name,
                value=signal.value,
                confidence=signal.confidence,
                timestamp=signal.timestamp or timestamp,
                metadata={},
            )
            if ranked_signal:
                ranked.append(ranked_signal)
        
        # Process Phase 3 enhanced signals (dict format from get_phase3_signals())
        if phase3_signals:
            for name, signal in phase3_signals.items():
                if not isinstance(signal, EnhancedSignal):
                    continue
                ranked_signal = self._create_ranked_signal(
                    name=name,
                    value=signal.value,
                    confidence=signal.confidence,
                    timestamp=signal.timestamp or timestamp,
                    metadata=signal.metadata if signal.metadata else {},
                )
                if ranked_signal:
                    ranked.append(ranked_signal)
        
        # Apply correlation penalty
        ranked = self._apply_correlation_penalty(ranked)
        
        return ranked
    
    def _create_ranked_signal(
        self,
        name: str,
        value: float,
        confidence: float,
        timestamp: datetime,
        metadata: Dict[str, Any],
    ) -> Optional[RankedSignal]:
        """Create a ranked signal with direction and score."""
        # Get signal metadata
        signal_meta = SIGNAL_METADATA.get(name)
        if not signal_meta:
            logger.warning(f"Unknown signal: {name}")
            return None
        
        # Determine direction
        direction = self._determine_direction(name, value, metadata, signal_meta)
        
        # Calculate rank score
        rank_score = self._calculate_rank_score(
            value=value,
            confidence=confidence,
            weight=signal_meta['weight'],
            timestamp=timestamp,
            name=name,
        )
        
        # Update history
        self._update_history(name, timestamp, value)
        
        return RankedSignal(
            name=name,
            value=value,
            confidence=confidence,
            direction=direction,
            category=signal_meta['category'],
            rank_score=rank_score,
            weight=signal_meta['weight'],
            timestamp=timestamp,
            metadata=metadata,
        )
    
    def _determine_direction(
        self,
        name: str,
        value: float,
        metadata: Dict[str, Any],
        signal_meta: Dict[str, Any],
    ) -> SignalDirection:
        """Determine signal direction (bullish/bearish/neutral)."""
        
        # Check for explicit direction in metadata (Phase 3 signals)
        if 'direction' in metadata:
            dir_str = str(metadata['direction']).lower()
            if dir_str in ['up', 'long', 'bullish', 'buying', 'accumulation']:
                return SignalDirection.BULLISH
            elif dir_str in ['down', 'short', 'bearish', 'selling', 'distribution']:
                return SignalDirection.BEARISH
            else:
                return SignalDirection.NEUTRAL
        
        # Check for phase in metadata
        if 'phase' in metadata:
            phase = str(metadata['phase']).lower()
            if phase == 'accumulation':
                return SignalDirection.BULLISH
            elif phase == 'distribution':
                return SignalDirection.BEARISH
        
        # Check for signal type
        if 'signal' in metadata:
            sig = str(metadata['signal']).lower()
            if sig in ['oversold', 'bullish']:
                return SignalDirection.BULLISH
            elif sig in ['overbought', 'bearish']:
                return SignalDirection.BEARISH
        
        # Use threshold-based direction
        threshold = signal_meta.get('direction_threshold', 0.5)
        inverted = signal_meta.get('inverted', False)
        
        if inverted:
            # For inverted signals (risk), low is bullish
            if value < threshold * 0.8:
                return SignalDirection.BULLISH
            elif value > threshold * 1.2:
                return SignalDirection.BEARISH
        else:
            # Normal signals, high is bullish
            if value > threshold * 1.2:
                return SignalDirection.BULLISH
            elif value < threshold * 0.8:
                return SignalDirection.BEARISH
        
        return SignalDirection.NEUTRAL
    
    def _calculate_rank_score(
        self,
        value: float,
        confidence: float,
        weight: float,
        timestamp: datetime,
        name: str,
    ) -> float:
        """
        Calculate ranking score for a signal.
        
        Score = (abs(value - 0.5) * weight) * (confidence_factor) * (recency_factor)
        """
        # Significance: how far from neutral (0.5)
        significance = abs(value - 0.5) * 2  # Scale to 0-1
        
        # Confidence factor
        confidence_factor = 1.0 + (confidence - 0.5) * self.confidence_weight
        
        # Recency factor (decay older signals)
        recency_factor = self._calculate_recency_factor(name, timestamp)
        
        # Final score
        rank_score = significance * weight * confidence_factor * recency_factor
        
        return min(max(rank_score, 0.0), 2.0)  # Cap at 2.0
    
    def _calculate_recency_factor(self, name: str, timestamp: datetime) -> float:
        """Calculate recency factor based on signal history."""
        history = self._signal_history.get(name, [])
        if len(history) < 2:
            return 1.0
        
        # Check if signal has been stable (reduce importance) or volatile (increase)
        recent_values = [v for _, v in history[-10:]]
        if len(recent_values) < 2:
            return 1.0
        
        volatility = np.std(recent_values) if len(recent_values) > 1 else 0
        
        # Higher volatility = more recent change = higher importance
        return 0.8 + min(volatility * 0.4, 0.4)  # Range: 0.8 - 1.2
    
    def _apply_correlation_penalty(self, signals: List[RankedSignal]) -> List[RankedSignal]:
        """Apply penalty to correlated signals to avoid double-counting."""
        signal_names = {s.name for s in signals}
        
        for group_name, group_signals in self._correlation_groups.items():
            # Find signals in this correlation group
            matching = [s for s in signals if s.name in group_signals]
            
            if len(matching) > 1:
                # Sort by rank score, penalize lower-ranked correlated signals
                matching.sort(key=lambda s: s.rank_score, reverse=True)
                
                for i, signal in enumerate(matching[1:], 1):
                    penalty = 1.0 - (self.correlation_penalty * i)
                    signal.rank_score *= max(penalty, 0.5)  # Min 50% of original
        
        return signals
    
    def _update_history(self, name: str, timestamp: datetime, value: float):
        """Update signal history."""
        self._signal_history[name].append((timestamp, value))
        
        # Trim history
        if len(self._signal_history[name]) > self._max_history:
            self._signal_history[name] = self._signal_history[name][-self._max_history:]
    
    # =========================================================================
    # CONFLICT DETECTION
    # =========================================================================
    
    def _detect_conflicts(self, signals: List[RankedSignal]) -> Optional[SignalConflict]:
        """Detect conflicts between bullish and bearish signals."""
        bullish = [s for s in signals if s.direction == SignalDirection.BULLISH]
        bearish = [s for s in signals if s.direction == SignalDirection.BEARISH]
        
        # No conflict if all same direction
        if not bullish or not bearish:
            return None
        
        # Calculate weighted scores for each direction
        bullish_score = sum(s.rank_score * s.confidence for s in bullish)
        bearish_score = sum(s.rank_score * s.confidence for s in bearish)
        total_score = bullish_score + bearish_score
        
        if total_score == 0:
            return None
        
        # Calculate disagreement ratio
        disagreement = min(bullish_score, bearish_score) / total_score
        
        # Determine severity
        if disagreement < 0.15:
            severity = ConflictSeverity.MINOR
        elif disagreement < 0.30:
            severity = ConflictSeverity.MODERATE
        elif disagreement < 0.45:
            severity = ConflictSeverity.SEVERE
        else:
            severity = ConflictSeverity.CRITICAL
        
        # Resolve conflict
        if bullish_score > bearish_score:
            resolution = SignalDirection.BULLISH
            resolution_confidence = (bullish_score - bearish_score) / total_score
        elif bearish_score > bullish_score:
            resolution = SignalDirection.BEARISH
            resolution_confidence = (bearish_score - bullish_score) / total_score
        else:
            resolution = SignalDirection.NEUTRAL
            resolution_confidence = 0.0
        
        # Generate explanation
        explanation = self._generate_conflict_explanation(
            bullish, bearish, severity, resolution, resolution_confidence
        )
        
        return SignalConflict(
            bullish_signals=bullish,
            bearish_signals=bearish,
            severity=severity,
            resolution=resolution,
            resolution_confidence=resolution_confidence,
            explanation=explanation,
        )
    
    def _generate_conflict_explanation(
        self,
        bullish: List[RankedSignal],
        bearish: List[RankedSignal],
        severity: ConflictSeverity,
        resolution: SignalDirection,
        confidence: float,
    ) -> str:
        """Generate human-readable conflict explanation."""
        top_bullish = sorted(bullish, key=lambda s: s.rank_score, reverse=True)[:2]
        top_bearish = sorted(bearish, key=lambda s: s.rank_score, reverse=True)[:2]
        
        bull_names = ', '.join(s.name.replace('_', ' ') for s in top_bullish)
        bear_names = ', '.join(s.name.replace('_', ' ') for s in top_bearish)
        
        if severity == ConflictSeverity.MINOR:
            prefix = "Minor conflict"
        elif severity == ConflictSeverity.MODERATE:
            prefix = "Moderate conflict"
        elif severity == ConflictSeverity.SEVERE:
            prefix = "Severe conflict"
        else:
            prefix = "Critical conflict"
        
        res_str = resolution.value.upper()
        conf_pct = round(confidence * 100, 1)
        
        return f"{prefix}: Bullish ({bull_names}) vs Bearish ({bear_names}). Resolution: {res_str} ({conf_pct}% confidence)"
    
    # =========================================================================
    # MARKET BIAS CALCULATION
    # =========================================================================
    
    def _calculate_market_bias(
        self,
        signals: List[RankedSignal],
        conflict: Optional[SignalConflict],
    ) -> Tuple[SignalDirection, float]:
        """Calculate overall market bias from signals."""
        if not signals:
            return SignalDirection.NEUTRAL, 0.0
        
        # Weighted vote
        bullish_weight = sum(
            s.rank_score * s.confidence 
            for s in signals 
            if s.direction == SignalDirection.BULLISH
        )
        bearish_weight = sum(
            s.rank_score * s.confidence 
            for s in signals 
            if s.direction == SignalDirection.BEARISH
        )
        neutral_weight = sum(
            s.rank_score * s.confidence * 0.5  # Neutral signals count half
            for s in signals 
            if s.direction == SignalDirection.NEUTRAL
        )
        
        total = bullish_weight + bearish_weight + neutral_weight
        if total == 0:
            return SignalDirection.NEUTRAL, 0.0
        
        # Determine bias
        if bullish_weight > bearish_weight * 1.2:  # 20% threshold
            bias = SignalDirection.BULLISH
            confidence = (bullish_weight - bearish_weight) / total
        elif bearish_weight > bullish_weight * 1.2:
            bias = SignalDirection.BEARISH
            confidence = (bearish_weight - bullish_weight) / total
        else:
            bias = SignalDirection.NEUTRAL
            confidence = 1.0 - abs(bullish_weight - bearish_weight) / total
        
        # Reduce confidence if there's a severe conflict
        if conflict and conflict.severity in [ConflictSeverity.SEVERE, ConflictSeverity.CRITICAL]:
            confidence *= 0.7
        
        return bias, min(max(confidence, 0.0), 1.0)
    
    def _calculate_category_scores(self, signals: List[RankedSignal]) -> Dict[str, float]:
        """Calculate average score per category."""
        category_signals: Dict[SignalCategory, List[RankedSignal]] = defaultdict(list)
        
        for signal in signals:
            category_signals[signal.category].append(signal)
        
        scores = {}
        for category, cat_signals in category_signals.items():
            if cat_signals:
                avg_score = sum(s.value for s in cat_signals) / len(cat_signals)
                scores[category.value] = avg_score
        
        return scores
    
    # =========================================================================
    # RECOMMENDATION GENERATION
    # =========================================================================
    
    def _generate_recommendation(
        self,
        ranked_signals: List[RankedSignal],
        conflict: Optional[SignalConflict],
        market_bias: SignalDirection,
        bias_confidence: float,
        category_scores: Dict[str, float],
        timestamp: datetime,
    ) -> TradeRecommendation:
        """Generate actionable trade recommendation."""
        
        # Determine recommendation strength
        strength = self._determine_strength(bias_confidence, conflict)
        
        # Categorize signals
        primary, supporting, conflicting = self._categorize_signals(ranked_signals, market_bias)
        
        # Assess risk
        risk_level, risk_factors = self._assess_risk(ranked_signals, category_scores, conflict)
        
        # Determine entry bias
        entry_bias = self._determine_entry_bias(strength, risk_level, conflict)
        
        # Determine timing
        urgency, timeframe = self._determine_timing(ranked_signals, category_scores)
        
        # Generate explanation
        explanation = self._generate_explanation(
            market_bias, strength, primary, risk_level, conflict
        )
        
        # Generate alerts
        alerts = self._generate_alerts(ranked_signals, conflict, risk_level)
        
        return TradeRecommendation(
            direction=market_bias,
            strength=strength,
            confidence=bias_confidence,
            entry_bias=entry_bias,
            primary_signals=primary,
            supporting_signals=supporting,
            conflicting_signals=conflicting,
            risk_level=risk_level,
            risk_factors=risk_factors,
            urgency=urgency,
            timeframe=timeframe,
            timestamp=timestamp,
            explanation=explanation,
            alerts=alerts,
        )
    
    def _determine_strength(
        self,
        confidence: float,
        conflict: Optional[SignalConflict],
    ) -> RecommendationStrength:
        """Determine recommendation strength."""
        if conflict and conflict.severity == ConflictSeverity.CRITICAL:
            return RecommendationStrength.CONFLICTED
        
        if confidence > 0.7:
            if conflict and conflict.severity == ConflictSeverity.SEVERE:
                return RecommendationStrength.MODERATE
            return RecommendationStrength.STRONG
        elif confidence > 0.4:
            return RecommendationStrength.MODERATE
        elif confidence > 0.2:
            return RecommendationStrength.WEAK
        else:
            return RecommendationStrength.NO_ACTION
    
    def _categorize_signals(
        self,
        signals: List[RankedSignal],
        market_bias: SignalDirection,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Categorize signals into primary, supporting, and conflicting."""
        primary = []
        supporting = []
        conflicting = []
        
        # Sort by rank score
        sorted_signals = sorted(signals, key=lambda s: s.rank_score, reverse=True)
        
        for signal in sorted_signals:
            if signal.direction == market_bias:
                if len(primary) < 3:
                    primary.append(signal.name)
                else:
                    supporting.append(signal.name)
            elif signal.direction != SignalDirection.NEUTRAL:
                conflicting.append(signal.name)
        
        return primary, supporting[:5], conflicting[:3]
    
    def _assess_risk(
        self,
        signals: List[RankedSignal],
        category_scores: Dict[str, float],
        conflict: Optional[SignalConflict],
    ) -> Tuple[str, List[str]]:
        """Assess overall risk level."""
        risk_factors = []
        risk_score = 0.0
        
        # Check risk category
        risk_cat_score = category_scores.get('risk', 0.5)
        if risk_cat_score > 0.7:
            risk_score += 0.3
            risk_factors.append("High risk signals detected")
        
        # Check for liquidation cascade risk
        cascade_signal = next((s for s in signals if s.name == 'liquidation_cascade_risk'), None)
        if cascade_signal and cascade_signal.value > 0.7:
            risk_score += 0.3
            severity = cascade_signal.metadata.get('severity', 'unknown')
            risk_factors.append(f"Liquidation cascade risk: {severity}")
        
        # Check for stop hunt
        stop_hunt = next((s for s in signals if s.name == 'stop_hunt_probability'), None)
        if stop_hunt and stop_hunt.value > 0.6:
            risk_score += 0.2
            risk_factors.append("Stop hunt probability elevated")
        
        # Check for conflict
        if conflict:
            if conflict.severity == ConflictSeverity.CRITICAL:
                risk_score += 0.3
                risk_factors.append("Critical signal conflict")
            elif conflict.severity == ConflictSeverity.SEVERE:
                risk_score += 0.2
                risk_factors.append("Severe signal conflict")
        
        # Check momentum exhaustion
        exhaustion = next((s for s in signals if s.name == 'momentum_exhaustion'), None)
        if exhaustion and exhaustion.value > 0.7:
            risk_score += 0.15
            risk_factors.append("Momentum exhaustion detected")
        
        # Determine risk level
        if risk_score > 0.7:
            return "extreme", risk_factors
        elif risk_score > 0.5:
            return "high", risk_factors
        elif risk_score > 0.3:
            return "medium", risk_factors
        else:
            return "low", risk_factors if risk_factors else ["Normal market conditions"]
    
    def _determine_entry_bias(
        self,
        strength: RecommendationStrength,
        risk_level: str,
        conflict: Optional[SignalConflict],
    ) -> str:
        """Determine entry bias (aggressive/conservative/wait)."""
        if strength == RecommendationStrength.CONFLICTED:
            return "wait"
        if strength == RecommendationStrength.NO_ACTION:
            return "wait"
        
        if risk_level == "extreme":
            return "wait"
        elif risk_level == "high":
            return "conservative"
        elif strength == RecommendationStrength.STRONG and risk_level == "low":
            return "aggressive"
        else:
            return "conservative"
    
    def _determine_timing(
        self,
        signals: List[RankedSignal],
        category_scores: Dict[str, float],
    ) -> Tuple[str, str]:
        """Determine urgency and timeframe."""
        # Check timing signals
        timing_score = category_scores.get('timing', 0.5)
        
        # Check volatility breakout
        breakout = next((s for s in signals if s.name == 'volatility_breakout'), None)
        if breakout and breakout.value > 0.7:
            timeframe_meta = breakout.metadata.get('timeframe', 'intraday')
            return "soon", timeframe_meta if timeframe_meta else "intraday"
        
        # Check mean reversion
        reversion = next((s for s in signals if s.name == 'mean_reversion'), None)
        if reversion and abs(reversion.value - 0.5) > 0.3:
            return "patient", "swing"
        
        # Check execution quality
        execution = next((s for s in signals if s.name == 'execution_quality'), None)
        if execution and execution.value > 0.7:
            return "immediate", "scalp"
        
        # Default
        if timing_score > 0.6:
            return "soon", "intraday"
        else:
            return "patient", "swing"
    
    def _generate_explanation(
        self,
        bias: SignalDirection,
        strength: RecommendationStrength,
        primary_signals: List[str],
        risk_level: str,
        conflict: Optional[SignalConflict],
    ) -> str:
        """Generate human-readable explanation."""
        bias_str = bias.value.upper()
        strength_str = strength.value.replace('_', ' ')
        primary_str = ', '.join(s.replace('_', ' ') for s in primary_signals[:2])
        
        explanation = f"{strength_str.title()} {bias_str} bias"
        
        if primary_signals:
            explanation += f" supported by {primary_str}"
        
        explanation += f". Risk: {risk_level}."
        
        if conflict and conflict.severity in [ConflictSeverity.SEVERE, ConflictSeverity.CRITICAL]:
            explanation += f" Warning: {conflict.severity.value} signal conflict detected."
        
        return explanation
    
    def _generate_alerts(
        self,
        signals: List[RankedSignal],
        conflict: Optional[SignalConflict],
        risk_level: str,
    ) -> List[str]:
        """Generate alert messages."""
        alerts = []
        
        # Risk alerts
        if risk_level == "extreme":
            alerts.append("⚠️ EXTREME RISK: Consider reducing exposure")
        elif risk_level == "high":
            alerts.append("⚠️ High risk environment")
        
        # Conflict alerts
        if conflict:
            if conflict.severity == ConflictSeverity.CRITICAL:
                alerts.append("🔴 CRITICAL: Conflicting signals - wait for clarity")
            elif conflict.severity == ConflictSeverity.SEVERE:
                alerts.append("🟠 Severe signal conflict detected")
        
        # Specific signal alerts
        cascade = next((s for s in signals if s.name == 'liquidation_cascade_risk'), None)
        if cascade and cascade.value > 0.8:
            direction = cascade.metadata.get('direction', 'unknown')
            alerts.append(f"🔴 CASCADE ALERT: High liquidation risk ({direction})")
        
        arbitrage = next((s for s in signals if s.name == 'arbitrage_opportunity'), None)
        if arbitrage and arbitrage.value > 0.7:
            arb_type = arbitrage.metadata.get('type', 'unknown')
            alerts.append(f"💰 Arbitrage opportunity: {arb_type}")
        
        return alerts
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def reset(self):
        """Reset aggregator state."""
        self._signal_history.clear()
        logger.info("SignalAggregator reset")
    
    def get_signal_history(self, name: str) -> List[Tuple[datetime, float]]:
        """Get history for a specific signal."""
        return list(self._signal_history.get(name, []))
    
    def get_all_history(self) -> Dict[str, List[Tuple[datetime, float]]]:
        """Get all signal history."""
        return dict(self._signal_history)
