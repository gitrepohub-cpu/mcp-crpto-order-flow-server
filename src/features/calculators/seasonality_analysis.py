"""
Seasonality Analysis Calculator

Detect and analyze seasonal patterns in market data.
"""

import logging
from typing import Dict, List, Optional, Any

from src.features.base import FeatureCalculator, FeatureResult
from src.features.utils import generate_signal
from src.analytics.timeseries_engine import (
    get_timeseries_engine,
    TimeSeriesData
)

logger = logging.getLogger(__name__)


class SeasonalityCalculator(FeatureCalculator):
    """
    Detect seasonal patterns and decompose time series.
    
    Analyzes:
        - Hourly patterns (intraday)
        - Daily patterns (day-of-week)
        - FFT-based frequency detection
        - Trend/Seasonal/Residual decomposition
    """
    
    name = "seasonality_analysis"
    description = "Detect seasonal patterns and decompose time series"
    category = "seasonality"
    version = "1.0.0"
    
    async def calculate(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        data_type: str = "prices",
        hours: int = 336,
        resample_freq: str = "1h",
        decomposition_period: int = 24,
        top_frequencies: int = 5,
        **params
    ) -> FeatureResult:
        """
        Analyze seasonality in data.
        
        Args:
            symbol: Trading pair
            exchange: Exchange
            data_type: prices, funding_rates, trades, open_interest
            hours: Hours of data (default 14 days)
            resample_freq: Resample frequency
            decomposition_period: Period for seasonal decomposition
            top_frequencies: Number of top frequencies to return
        """
        symbol_lower = symbol.lower()
        exc = (exchange or 'binance').lower()
        
        engine = get_timeseries_engine()
        
        value_col = self._get_value_column(data_type)
        table_name = self.get_table_name(symbol_lower, exc, 'futures', data_type)
        
        if not self.table_exists(table_name):
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={},
                errors=[f"No {data_type} data found"]
            )
        
        try:
            query = f"""
                SELECT timestamp, {value_col} as value
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
                ORDER BY timestamp
            """
            
            results = self.execute_query(query)
            
            if len(results) < decomposition_period * 3:
                return self.create_result(
                    symbol=symbol,
                    exchanges=[exc],
                    data={},
                    errors=["Insufficient data for seasonality analysis"]
                )
            
            ts = TimeSeriesData.from_duckdb_result(results, name=data_type)
            ts = ts.resample(resample_freq)
            
            # Detect seasonality
            seasonality_result = engine.detect_seasonality(
                ts,
                top_n=top_frequencies
            )
            
            # Decompose time series
            decomp_result = engine.decompose_seasonality(
                ts,
                period=decomposition_period
            )
            
            # Analyze patterns
            pattern_analysis = self._analyze_patterns(
                seasonality_result,
                decomp_result,
                resample_freq
            )
            
            # Generate signals
            signals = self._generate_seasonality_signals(
                seasonality_result,
                decomp_result
            )
            
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={
                    'seasonality': {
                        'has_seasonality': seasonality_result.get('has_seasonality', False),
                        'dominant_period': seasonality_result.get('dominant_period'),
                        'dominant_frequency': seasonality_result.get('dominant_frequency'),
                        'top_frequencies': seasonality_result.get('top_frequencies', []),
                        'spectral_density': seasonality_result.get('spectral_density')
                    },
                    'decomposition': {
                        'trend_strength': decomp_result.get('trend_strength'),
                        'seasonal_strength': decomp_result.get('seasonal_strength'),
                        'trend_direction': decomp_result.get('trend_direction'),
                        'residual_variance': decomp_result.get('residual_variance'),
                        'period': decomposition_period
                    },
                    'pattern_analysis': pattern_analysis,
                    'data_stats': {
                        'points_analyzed': len(ts),
                        'resample_freq': resample_freq,
                        'time_range_hours': hours
                    }
                },
                metadata={
                    'data_type': data_type,
                    'decomposition_period': decomposition_period
                },
                signals=signals
            )
            
        except Exception as e:
            logger.error(f"Seasonality analysis failed: {e}")
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={},
                errors=[str(e)]
            )
    
    def _get_value_column(self, data_type: str) -> str:
        columns = {
            'prices': 'mid_price',
            'funding_rates': 'funding_rate',
            'trades': 'price',
            'open_interest': 'open_interest'
        }
        return columns.get(data_type, 'mid_price')
    
    def _analyze_patterns(
        self,
        seasonality: Dict,
        decomp: Dict,
        resample_freq: str
    ) -> Dict:
        """Analyze detected patterns for trading insights."""
        analysis = {
            'trading_implications': [],
            'pattern_type': 'unknown'
        }
        
        # Interpret dominant period
        dom_period = seasonality.get('dominant_period')
        if dom_period:
            if resample_freq == '1h':
                if 23 <= dom_period <= 25:
                    analysis['pattern_type'] = 'daily'
                    analysis['trading_implications'].append(
                        "Daily cycle detected - consider time-of-day strategies"
                    )
                elif 6 <= dom_period <= 8:
                    analysis['pattern_type'] = 'asian_session'
                    analysis['trading_implications'].append(
                        "8-hour cycle detected - likely session-based pattern"
                    )
                elif 166 <= dom_period <= 170:
                    analysis['pattern_type'] = 'weekly'
                    analysis['trading_implications'].append(
                        "Weekly cycle detected - consider day-of-week effects"
                    )
            elif resample_freq == '4h':
                if 5 <= dom_period <= 7:
                    analysis['pattern_type'] = 'daily'
                    analysis['trading_implications'].append(
                        "Daily cycle detected in 4H data"
                    )
        
        # Trend strength implications
        trend_strength = decomp.get('trend_strength', 0)
        if trend_strength > 0.7:
            analysis['trading_implications'].append(
                f"Strong trend component ({trend_strength:.1%}) - trend following preferred"
            )
        elif trend_strength < 0.3:
            analysis['trading_implications'].append(
                f"Weak trend ({trend_strength:.1%}) - mean reversion may work better"
            )
        
        # Seasonal strength implications
        seasonal_strength = decomp.get('seasonal_strength', 0)
        if seasonal_strength > 0.5:
            analysis['trading_implications'].append(
                f"Strong seasonality ({seasonal_strength:.1%}) - time-based entries valuable"
            )
        
        return analysis
    
    def _generate_seasonality_signals(
        self,
        seasonality: Dict,
        decomp: Dict
    ) -> List[Dict]:
        """Generate signals from seasonality analysis."""
        signals = []
        
        # Strong seasonality signal
        if seasonality.get('has_seasonality'):
            seasonal_strength = decomp.get('seasonal_strength', 0)
            signals.append(generate_signal(
                'INFO', seasonal_strength,
                f"Seasonal pattern detected (strength: {seasonal_strength:.1%})",
                {'dominant_period': seasonality.get('dominant_period')}
            ))
        
        # Strong trend signal
        trend_strength = decomp.get('trend_strength', 0)
        trend_direction = decomp.get('trend_direction', 'flat')
        
        if trend_strength > 0.6:
            signal_type = 'BULLISH' if trend_direction == 'up' else 'BEARISH' if trend_direction == 'down' else 'NEUTRAL'
            signals.append(generate_signal(
                signal_type, trend_strength,
                f"Strong underlying trend: {trend_direction} ({trend_strength:.1%})",
                {'trend_direction': trend_direction, 'trend_strength': trend_strength}
            ))
        
        # High residual variance (unpredictable)
        residual_var = decomp.get('residual_variance', 0)
        if residual_var > 0.5:
            signals.append(generate_signal(
                'WARNING', residual_var,
                f"High unpredictable component ({residual_var:.1%})",
                {'residual_variance': residual_var}
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
            'data_type': {
                'type': 'str',
                'default': 'prices',
                'description': 'Data type: prices, funding_rates, trades, open_interest',
                'required': False
            },
            'hours': {
                'type': 'int',
                'default': 336,
                'description': 'Hours of data (default 14 days)',
                'required': False
            },
            'resample_freq': {
                'type': 'str',
                'default': '1h',
                'description': 'Resample frequency',
                'required': False
            },
            'decomposition_period': {
                'type': 'int',
                'default': 24,
                'description': 'Period for decomposition (e.g., 24 for daily)',
                'required': False
            },
            'top_frequencies': {
                'type': 'int',
                'default': 5,
                'description': 'Number of top frequencies to return',
                'required': False
            }
        }
