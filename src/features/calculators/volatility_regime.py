"""
Volatility Regime Detector

Identifies market volatility regimes and regime changes
for adaptive trading strategies.

Uses TimeSeriesEngine for:
- Regime detection with transition probabilities
- Seasonality analysis in volatility
- Feature extraction for regime characteristics
"""

import logging
from typing import Dict, List, Optional, Any
import statistics

from src.features.base import FeatureCalculator, FeatureResult
from src.features.utils import (
    generate_signal,
    calculate_volatility,
    exponential_moving_average
)

logger = logging.getLogger(__name__)


class VolatilityRegimeCalculator(FeatureCalculator):
    """
    Detects volatility regimes and regime transitions using TimeSeriesEngine.
    
    Uses TimeSeriesEngine for:
        - Market regime detection (7 regime types)
        - Transition probability matrix
        - Seasonality analysis in volatility patterns
        - Feature extraction for regime characteristics
    
    Metrics:
        - Current volatility state (LOW, NORMAL, HIGH, EXTREME)
        - Regime transition probability
        - Historical regime distribution
        - Volatility term structure
        - Regime classification (TimeSeriesEngine)
        - Volatility seasonality (TimeSeriesEngine)
    """
    
    name = "volatility_regime"
    description = "Detect market volatility regimes with time series analysis and regime transitions"
    category = "volatility"
    version = "2.0.0"
    
    async def calculate(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        hours: int = 168,  # 7 days default
        short_window: int = 60,  # 1 hour
        long_window: int = 1440,  # 24 hours
        **params
    ) -> FeatureResult:
        """
        Calculate volatility regime.
        
        Args:
            symbol: Trading pair
            exchange: Specific exchange (required for volatility)
            hours: Hours of history
            short_window: Short-term vol window in minutes
            long_window: Long-term vol window in minutes
        """
        symbol_lower = symbol.lower()
        exc = (exchange or 'binance').lower()
        
        table_name = self.get_table_name(symbol_lower, exc, 'futures', 'prices')
        
        if not self.table_exists(table_name):
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={},
                errors=[f"No price data found for {symbol} on {exc}"]
            )
        
        try:
            # Get price data
            query = f"""
                SELECT timestamp, mid_price
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
                ORDER BY timestamp
            """
            
            results = self.execute_query(query)
            
            if len(results) < long_window:
                return self.create_result(
                    symbol=symbol,
                    exchanges=[exc],
                    data={},
                    errors=["Insufficient data for volatility calculation"]
                )
            
            prices = [r[1] for r in results]
            timestamps = [r[0] for r in results]
            
            # Calculate volatilities
            short_vol = self._calculate_rolling_volatility(prices, short_window)
            long_vol = self._calculate_rolling_volatility(prices, long_window)
            
            # Current volatility metrics
            current_short_vol = short_vol[-1] if short_vol else 0
            current_long_vol = long_vol[-1] if long_vol else 0
            
            # Volatility ratio (term structure)
            vol_ratio = current_short_vol / current_long_vol if current_long_vol > 0 else 1
            
            # Determine regime
            regime = self._classify_regime(current_short_vol, short_vol)
            
            # Calculate regime transitions
            regimes_history = [
                self._classify_regime(v, short_vol[:i+1])
                for i, v in enumerate(short_vol)
            ]
            
            transitions = self._detect_transitions(regimes_history)
            
            # Volatility percentile
            vol_percentile = self._calculate_percentile(current_short_vol, short_vol)
            
            signals = []
            
            # === USE TIME SERIES ENGINE FOR ADVANCED REGIME ANALYSIS ===
            timeseries_analysis = {}
            try:
                # Create time series from prices
                ts_data = self.create_timeseries_data(results, name='price')
                
                # Use TimeSeriesEngine regime detection
                regime_result = self.timeseries_engine.detect_regime(
                    ts_data,
                    lookback=min(long_window // 60, 20),  # Convert to hourly lookback
                    volatility_threshold=0.02
                )
                
                # Get seasonality in volatility
                vol_ts = self.create_timeseries_data(
                    [(timestamps[i + long_window], v) for i, v in enumerate(short_vol)],
                    name='volatility'
                )
                seasonality = self.timeseries_engine.detect_seasonality(vol_ts, top_n=3)
                
                # Extract volatility features
                vol_features = self.timeseries_engine.extract_features(vol_ts)
                
                timeseries_analysis = {
                    'engine_regime': {
                        'current': regime_result.current_regime.value,
                        'confidence': regime_result.confidence,
                        'history_length': len(regime_result.regime_history)
                    },
                    'transition_matrix': {
                        from_regime.value: {
                            to_regime.value: prob 
                            for to_regime, prob in transitions.items()
                        }
                        for from_regime, transitions in (regime_result.transition_matrix or {}).items()
                    },
                    'seasonality': {
                        'has_pattern': seasonality.get('has_seasonality', False),
                        'dominant_period': seasonality.get('dominant_period'),
                        'interpretation': self._interpret_seasonality(seasonality.get('dominant_period'))
                    },
                    'volatility_features': {
                        'trend': 'expanding' if vol_features.get('trend_slope', 0) > 0 else 'contracting',
                        'hurst': vol_features.get('hurst_exponent', 0.5),
                        'autocorrelation': vol_features.get('autocorr_lag1', 0)
                    }
                }
                
                # Generate signals from TimeSeriesEngine analysis
                if regime_result.current_regime.value in ['BREAKOUT', 'BREAKDOWN']:
                    signals.append(generate_signal(
                        'WARNING', regime_result.confidence,
                        f"TimeSeriesEngine: {regime_result.current_regime.value} regime detected",
                        {'regime': regime_result.current_regime.value, 'confidence': regime_result.confidence}
                    ))
                
                if vol_features.get('hurst_exponent', 0.5) > 0.65:
                    signals.append(generate_signal(
                        'INFO', 0.7,
                        f"Volatility is trending (Hurst: {vol_features.get('hurst_exponent', 0):.2f})",
                        {'hurst': vol_features.get('hurst_exponent')}
                    ))
                    
            except Exception as e:
                logger.warning(f"TimeSeriesEngine analysis failed: {e}")
            
            # Generate signals
            if regime == 'EXTREME':
                signals.append(generate_signal(
                    'WARNING', 0.9,
                    f"Extreme volatility detected on {exc}",
                    {'regime': regime, 'vol_percentile': vol_percentile}
                ))
            
            if vol_ratio > 1.5:
                signals.append(generate_signal(
                    'WARNING', min(vol_ratio / 2, 1.0),
                    f"Volatility expansion (short/long ratio: {vol_ratio:.2f})",
                    {'vol_ratio': vol_ratio}
                ))
            elif vol_ratio < 0.7:
                signals.append(generate_signal(
                    'NEUTRAL', 0.5,
                    f"Volatility compression (short/long ratio: {vol_ratio:.2f})",
                    {'vol_ratio': vol_ratio}
                ))
            
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={
                    'current_state': {
                        'regime': regime,
                        'short_term_volatility': current_short_vol,
                        'long_term_volatility': current_long_vol,
                        'volatility_ratio': vol_ratio,
                        'percentile': vol_percentile
                    },
                    'historical': {
                        'min_volatility': min(short_vol) if short_vol else 0,
                        'max_volatility': max(short_vol) if short_vol else 0,
                        'mean_volatility': statistics.mean(short_vol) if short_vol else 0,
                        'regime_distribution': self._regime_distribution(regimes_history)
                    },
                    'transitions': {
                        'recent_transitions': transitions[-5:],
                        'transition_count': len(transitions),
                        'avg_regime_duration': self._avg_regime_duration(regimes_history)
                    },
                    'timeseries_analysis': timeseries_analysis,
                    'time_series': [
                        {
                            'timestamp': str(timestamps[i + long_window]),
                            'volatility': v,
                            'regime': regimes_history[i] if i < len(regimes_history) else 'UNKNOWN'
                        }
                        for i, v in enumerate(short_vol[-48:])  # Last 48 data points
                    ]
                },
                metadata={
                    'exchange': exc,
                    'hours': hours,
                    'short_window': short_window,
                    'long_window': long_window,
                    'uses_timeseries_engine': True
                },
                signals=signals
            )
            
        except Exception as e:
            logger.error(f"Error calculating volatility regime: {e}")
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={},
                errors=[str(e)]
            )
    
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'symbol': {
                'type': 'str',
                'default': 'BTCUSDT',
                'description': 'Trading pair to analyze',
                'required': True
            },
            'exchange': {
                'type': 'str',
                'default': 'binance',
                'description': 'Exchange to analyze (required)',
                'required': False
            },
            'hours': {
                'type': 'int',
                'default': 168,
                'description': 'Hours of historical data (default 7 days)',
                'required': False
            },
            'short_window': {
                'type': 'int',
                'default': 60,
                'description': 'Short-term volatility window in minutes',
                'required': False
            },
            'long_window': {
                'type': 'int',
                'default': 1440,
                'description': 'Long-term volatility window in minutes',
                'required': False
            }
        }
    
    def _calculate_rolling_volatility(
        self,
        prices: List[float],
        window: int
    ) -> List[float]:
        """Calculate rolling volatility."""
        if len(prices) < window + 1:
            return []
        
        vols = []
        for i in range(window, len(prices)):
            window_prices = prices[i-window:i+1]
            vol = calculate_volatility(window_prices, annualize=True)
            vols.append(vol)
        
        return vols
    
    def _classify_regime(
        self,
        current_vol: float,
        historical_vols: List[float]
    ) -> str:
        """Classify volatility regime."""
        if not historical_vols:
            return 'UNKNOWN'
        
        percentile = self._calculate_percentile(current_vol, historical_vols)
        
        if percentile >= 95:
            return 'EXTREME'
        elif percentile >= 75:
            return 'HIGH'
        elif percentile >= 25:
            return 'NORMAL'
        else:
            return 'LOW'
    
    def _calculate_percentile(self, value: float, values: List[float]) -> float:
        """Calculate percentile rank."""
        if not values:
            return 50.0
        
        count_below = sum(1 for v in values if v < value)
        return (count_below / len(values)) * 100
    
    def _detect_transitions(self, regimes: List[str]) -> List[Dict]:
        """Detect regime transitions."""
        transitions = []
        
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i-1]:
                transitions.append({
                    'index': i,
                    'from': regimes[i-1],
                    'to': regimes[i]
                })
        
        return transitions
    
    def _regime_distribution(self, regimes: List[str]) -> Dict[str, float]:
        """Calculate regime distribution."""
        if not regimes:
            return {}
        
        counts = {}
        for r in regimes:
            counts[r] = counts.get(r, 0) + 1
        
        total = len(regimes)
        return {k: v / total * 100 for k, v in counts.items()}
    
    def _avg_regime_duration(self, regimes: List[str]) -> float:
        """Calculate average regime duration."""
        if not regimes:
            return 0
        
        durations = []
        current_regime = regimes[0]
        current_duration = 1
        
        for r in regimes[1:]:
            if r == current_regime:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_regime = r
                current_duration = 1
        
        durations.append(current_duration)
        
        return statistics.mean(durations) if durations else 0
    
    def _interpret_seasonality(self, period: Optional[float]) -> str:
        """Interpret volatility seasonality period."""
        if period is None:
            return "No clear seasonality"
        
        # Assuming minute-level data
        if 50 <= period <= 70:  # ~60 minutes
            return "Hourly volatility cycle"
        elif 1400 <= period <= 1500:  # ~1440 minutes = 1 day
            return "Daily volatility cycle"
        elif 9500 <= period <= 10500:  # ~10080 minutes = 1 week
            return "Weekly volatility cycle"
        elif 20 <= period <= 40:
            return "30-minute volatility bursts"
        else:
            return f"Cycle of ~{period:.0f} minutes"
