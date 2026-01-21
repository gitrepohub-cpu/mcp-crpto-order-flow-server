"""
Feature Extraction Calculator

Extract comprehensive statistical features from time series data
for machine learning and pattern recognition.
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


class FeatureExtractionCalculator(FeatureCalculator):
    """
    Extract 40+ statistical features from market time series.
    
    Features include:
        - Statistical moments (mean, std, skew, kurtosis)
        - Trend features (slope, R², direction changes)
        - Volatility features (ATR-like, range ratios)
        - Complexity features (sample entropy, Hurst exponent)
        - Autocorrelation features
    """
    
    name = "feature_extraction"
    description = "Extract comprehensive statistical features for ML and pattern recognition"
    category = "features"
    version = "1.0.0"
    
    async def calculate(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        data_type: str = "prices",
        hours: int = 24,
        resample_freq: str = "5m",
        include_advanced: bool = True,
        **params
    ) -> FeatureResult:
        """
        Extract features from time series.
        
        Args:
            symbol: Trading pair
            exchange: Exchange
            data_type: prices, funding_rates, trades, open_interest
            hours: Hours of data
            resample_freq: Resample frequency
            include_advanced: Include entropy/Hurst (slower)
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
            
            if len(results) < 30:
                return self.create_result(
                    symbol=symbol,
                    exchanges=[exc],
                    data={},
                    errors=["Insufficient data for feature extraction"]
                )
            
            # Convert and resample
            ts = TimeSeriesData.from_duckdb_result(results, name=data_type)
            ts = ts.resample(resample_freq)
            
            # Extract features
            features = engine.extract_features(ts, include_advanced=include_advanced)
            
            # Generate signals based on features
            signals = self._generate_feature_signals(features)
            
            # Categorize features
            categorized = self._categorize_features(features)
            
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={
                    'features': features,
                    'categorized_features': categorized,
                    'feature_count': len(features),
                    'data_stats': {
                        'points_analyzed': len(ts),
                        'resample_freq': resample_freq,
                        'time_range_hours': hours
                    }
                },
                metadata={
                    'data_type': data_type,
                    'include_advanced': include_advanced
                },
                signals=signals
            )
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
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
    
    def _categorize_features(self, features: Dict) -> Dict:
        """Categorize features by type."""
        categories = {
            'statistical': ['mean', 'std', 'var', 'skew', 'kurtosis', 'median', 
                          'q25', 'q75', 'iqr', 'min', 'max', 'range'],
            'trend': ['trend_slope', 'trend_r2', 'direction_changes', 
                     'direction_change_rate', 'total_return', 'cagr'],
            'volatility': ['volatility', 'mean_abs_change', 'max_abs_change',
                          'mean_abs_deviation', 'range_to_mean_ratio'],
            'autocorrelation': ['autocorr_lag1', 'autocorr_lag5', 'autocorr_lag10'],
            'complexity': ['sample_entropy', 'hurst_exponent'],
            'distribution': ['coeff_variation', 'above_mean_pct', 'below_mean_pct']
        }
        
        categorized = {}
        for cat, keys in categories.items():
            categorized[cat] = {k: features.get(k) for k in keys if k in features}
        
        return categorized
    
    def _generate_feature_signals(self, features: Dict) -> List[Dict]:
        """Generate signals from extracted features."""
        signals = []
        
        # High volatility
        volatility = features.get('volatility', 0)
        if volatility > 0.05:  # 5% volatility
            signals.append(generate_signal(
                'WARNING', min(volatility * 10, 1.0),
                f"High volatility: {volatility:.2%}",
                {'volatility': volatility}
            ))
        
        # Strong trend
        r2 = features.get('trend_r2', 0)
        slope = features.get('trend_slope', 0)
        if r2 > 0.7 and abs(slope) > 0:
            direction = "bullish" if slope > 0 else "bearish"
            signals.append(generate_signal(
                'BULLISH' if slope > 0 else 'BEARISH',
                r2,
                f"Strong {direction} trend (R²={r2:.2f})",
                {'slope': slope, 'r2': r2}
            ))
        
        # High autocorrelation (mean reversion opportunity)
        ac_lag1 = features.get('autocorr_lag1', 0)
        if ac_lag1 < -0.3:
            signals.append(generate_signal(
                'INFO', abs(ac_lag1),
                f"Negative autocorrelation: mean reversion potential",
                {'autocorr_lag1': ac_lag1}
            ))
        
        # Trending market (Hurst > 0.5)
        hurst = features.get('hurst_exponent', 0.5)
        if hurst > 0.65:
            signals.append(generate_signal(
                'INFO', hurst,
                f"Trending market (Hurst={hurst:.2f})",
                {'hurst': hurst}
            ))
        elif hurst < 0.35:
            signals.append(generate_signal(
                'INFO', 1 - hurst,
                f"Mean-reverting market (Hurst={hurst:.2f})",
                {'hurst': hurst}
            ))
        
        # High skewness
        skew = features.get('skew', 0)
        if abs(skew) > 1.0:
            direction = "positive" if skew > 0 else "negative"
            signals.append(generate_signal(
                'INFO', min(abs(skew) / 2, 1.0),
                f"High {direction} skewness: {skew:.2f}",
                {'skew': skew}
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
                'default': 24,
                'description': 'Hours of data',
                'required': False
            },
            'resample_freq': {
                'type': 'str',
                'default': '5m',
                'description': 'Resample frequency',
                'required': False
            },
            'include_advanced': {
                'type': 'bool',
                'default': True,
                'description': 'Include entropy and Hurst (slower)',
                'required': False
            }
        }
