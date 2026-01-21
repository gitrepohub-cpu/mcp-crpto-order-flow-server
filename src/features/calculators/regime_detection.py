"""
Regime Detection Calculator

Detect market regimes and provide transition analysis.
"""

import logging
from typing import Dict, List, Optional, Any

from src.features.base import FeatureCalculator, FeatureResult
from src.features.utils import generate_signal
from src.analytics.timeseries_engine import (
    get_timeseries_engine,
    TimeSeriesData,
    MarketRegime
)

logger = logging.getLogger(__name__)


class RegimeDetectionCalculator(FeatureCalculator):
    """
    Detect market regimes using volatility, momentum, and trend analysis.
    
    Regime Types:
        - TRENDING_UP: Strong upward momentum
        - TRENDING_DOWN: Strong downward momentum  
        - RANGING: Sideways consolidation
        - HIGH_VOLATILITY: Elevated volatility
        - LOW_VOLATILITY: Compressed volatility
        - BREAKOUT: Volatility expansion with upward movement
        - BREAKDOWN: Volatility expansion with downward movement
    """
    
    name = "regime_detection"
    description = "Detect market regimes and transition probabilities"
    category = "regimes"
    version = "1.0.0"
    
    async def calculate(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        hours: int = 168,
        lookback: int = 20,
        resample_freq: str = "1h",
        volatility_threshold: float = 0.02,
        **params
    ) -> FeatureResult:
        """
        Detect current market regime.
        
        Args:
            symbol: Trading pair
            exchange: Exchange
            hours: Hours of data (default 7 days)
            lookback: Lookback period for regime calculation
            resample_freq: Resample frequency
            volatility_threshold: Threshold for high/low volatility
        """
        symbol_lower = symbol.lower()
        exc = (exchange or 'binance').lower()
        
        engine = get_timeseries_engine()
        
        table_name = self.get_table_name(symbol_lower, exc, 'futures', 'prices')
        
        if not self.table_exists(table_name):
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={},
                errors=[f"No price data found for {symbol}"]
            )
        
        try:
            query = f"""
                SELECT timestamp, mid_price as value
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
                ORDER BY timestamp
            """
            
            results = self.execute_query(query)
            
            if len(results) < lookback * 2:
                return self.create_result(
                    symbol=symbol,
                    exchanges=[exc],
                    data={},
                    errors=["Insufficient data for regime detection"]
                )
            
            ts = TimeSeriesData.from_duckdb_result(results, name='price')
            ts = ts.resample(resample_freq)
            
            # Detect regime
            regime_result = engine.detect_regime(
                ts,
                lookback=lookback,
                volatility_threshold=volatility_threshold
            )
            
            # Analyze regime history
            history_analysis = self._analyze_regime_history(regime_result.regime_history)
            
            # Generate signals
            signals = self._generate_regime_signals(regime_result)
            
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={
                    'current_regime': regime_result.current_regime.value,
                    'regime_confidence': regime_result.confidence,
                    'regime_history': [r.value for r in regime_result.regime_history],
                    'transition_matrix': self._format_transition_matrix(
                        regime_result.transition_matrix
                    ),
                    'history_analysis': history_analysis,
                    'regime_characteristics': self._get_regime_characteristics(
                        regime_result.current_regime
                    ),
                    'data_stats': {
                        'points_analyzed': len(ts),
                        'lookback': lookback,
                        'time_range_hours': hours
                    }
                },
                metadata={
                    'volatility_threshold': volatility_threshold,
                    'resample_freq': resample_freq
                },
                signals=signals
            )
            
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={},
                errors=[str(e)]
            )
    
    def _analyze_regime_history(self, history: List[MarketRegime]) -> Dict:
        """Analyze regime history for patterns."""
        if not history:
            return {}
        
        # Count regimes
        regime_counts = {}
        for r in history:
            regime_counts[r.value] = regime_counts.get(r.value, 0) + 1
        
        total = len(history)
        regime_distribution = {k: v / total for k, v in regime_counts.items()}
        
        # Find dominant regime
        dominant = max(regime_counts.items(), key=lambda x: x[1])
        
        # Count transitions
        transitions = 0
        for i in range(1, len(history)):
            if history[i] != history[i-1]:
                transitions += 1
        
        # Find longest streak
        current_streak = 1
        max_streak = 1
        max_streak_regime = history[0]
        
        for i in range(1, len(history)):
            if history[i] == history[i-1]:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
                    max_streak_regime = history[i]
            else:
                current_streak = 1
        
        return {
            'regime_distribution': regime_distribution,
            'dominant_regime': dominant[0],
            'dominant_regime_pct': dominant[1] / total,
            'total_transitions': transitions,
            'transition_frequency': transitions / (total - 1) if total > 1 else 0,
            'longest_streak': max_streak,
            'longest_streak_regime': max_streak_regime.value,
            'stability_score': 1 - (transitions / (total - 1)) if total > 1 else 1
        }
    
    def _format_transition_matrix(self, matrix: Optional[Dict]) -> Dict:
        """Format transition matrix for output."""
        if not matrix:
            return {}
        
        formatted = {}
        for from_regime, transitions in matrix.items():
            formatted[from_regime.value] = {
                to_regime.value: prob 
                for to_regime, prob in transitions.items()
            }
        
        return formatted
    
    def _get_regime_characteristics(self, regime: MarketRegime) -> Dict:
        """Get characteristics for each regime type."""
        characteristics = {
            MarketRegime.TRENDING_UP: {
                'description': 'Strong upward momentum',
                'strategy_hints': ['Trend following', 'Buy dips', 'Trail stops'],
                'risk_level': 'Medium',
                'typical_duration': 'Days to weeks'
            },
            MarketRegime.TRENDING_DOWN: {
                'description': 'Strong downward momentum',
                'strategy_hints': ['Avoid longs', 'Short rallies', 'Hedge positions'],
                'risk_level': 'High',
                'typical_duration': 'Days to weeks'
            },
            MarketRegime.RANGING: {
                'description': 'Sideways consolidation',
                'strategy_hints': ['Range trading', 'Fade extremes', 'Reduce size'],
                'risk_level': 'Low',
                'typical_duration': 'Hours to days'
            },
            MarketRegime.HIGH_VOLATILITY: {
                'description': 'Elevated volatility',
                'strategy_hints': ['Reduce position size', 'Wider stops', 'Options'],
                'risk_level': 'Very High',
                'typical_duration': 'Hours to days'
            },
            MarketRegime.LOW_VOLATILITY: {
                'description': 'Compressed volatility',
                'strategy_hints': ['Prepare for breakout', 'Long options', 'Straddles'],
                'risk_level': 'Low',
                'typical_duration': 'Days'
            },
            MarketRegime.BREAKOUT: {
                'description': 'Volatility expansion upward',
                'strategy_hints': ['Follow breakout', 'Momentum trading', 'Tight stops'],
                'risk_level': 'High',
                'typical_duration': 'Hours'
            },
            MarketRegime.BREAKDOWN: {
                'description': 'Volatility expansion downward',
                'strategy_hints': ['Avoid catching knife', 'Wait for stabilization'],
                'risk_level': 'Very High',
                'typical_duration': 'Hours'
            }
        }
        
        return characteristics.get(regime, {})
    
    def _generate_regime_signals(self, result) -> List[Dict]:
        """Generate signals from regime detection."""
        signals = []
        regime = result.current_regime
        confidence = result.confidence
        
        # Signal based on current regime
        regime_signals = {
            MarketRegime.TRENDING_UP: ('BULLISH', 'Bullish trending regime'),
            MarketRegime.TRENDING_DOWN: ('BEARISH', 'Bearish trending regime'),
            MarketRegime.RANGING: ('NEUTRAL', 'Ranging/Consolidation regime'),
            MarketRegime.HIGH_VOLATILITY: ('WARNING', 'High volatility regime'),
            MarketRegime.LOW_VOLATILITY: ('INFO', 'Low volatility compression'),
            MarketRegime.BREAKOUT: ('BULLISH', 'Breakout regime detected'),
            MarketRegime.BREAKDOWN: ('BEARISH', 'Breakdown regime detected')
        }
        
        if regime in regime_signals:
            signal_type, message = regime_signals[regime]
            signals.append(generate_signal(
                signal_type, confidence,
                f"{message} (confidence: {confidence:.1%})",
                {'regime': regime.value, 'confidence': confidence}
            ))
        
        # Check for regime instability
        if result.regime_history:
            recent = result.regime_history[-10:] if len(result.regime_history) >= 10 else result.regime_history
            unique_regimes = len(set(recent))
            
            if unique_regimes >= 4:
                signals.append(generate_signal(
                    'WARNING', 0.7,
                    'Unstable regime: frequent changes detected',
                    {'unique_regimes_recent': unique_regimes}
                ))
        
        return signals
    
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'symbol': {
                'type': 'str',
                'default': 'BTCUSDT',
                'description': 'Trading pair',
                'required': True
            },
            'exchange': {
                'type': 'str',
                'default': 'binance',
                'description': 'Exchange',
                'required': False
            },
            'hours': {
                'type': 'int',
                'default': 168,
                'description': 'Hours of data (default 7 days)',
                'required': False
            },
            'lookback': {
                'type': 'int',
                'default': 20,
                'description': 'Lookback period for regime calculation',
                'required': False
            },
            'resample_freq': {
                'type': 'str',
                'default': '1h',
                'description': 'Resample frequency',
                'required': False
            },
            'volatility_threshold': {
                'type': 'float',
                'default': 0.02,
                'description': 'Threshold for high/low volatility',
                'required': False
            }
        }
