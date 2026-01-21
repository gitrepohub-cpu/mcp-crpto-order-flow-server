"""
Advanced Anomaly Detection Calculator

Uses multiple anomaly detection algorithms to identify unusual
market behavior across different data streams.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.features.base import FeatureCalculator, FeatureResult
from src.features.utils import generate_signal
from src.analytics.timeseries_engine import (
    get_timeseries_engine,
    TimeSeriesData,
    AnomalyResult
)

logger = logging.getLogger(__name__)


class AnomalyDetectionCalculator(FeatureCalculator):
    """
    Detect anomalies in market data using multiple algorithms.
    
    Algorithms:
        - Z-score (global and rolling)
        - IQR (Interquartile Range)
        - Isolation Forest
        - CUSUM (Cumulative Sum)
    """
    
    name = "anomaly_detection"
    description = "Detect market anomalies using Z-score, IQR, Isolation Forest, and CUSUM"
    category = "detection"
    version = "1.0.0"
    
    async def calculate(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        data_type: str = "prices",
        hours: int = 24,
        method: str = "ensemble",
        zscore_threshold: float = 3.0,
        rolling_window: Optional[int] = 60,
        **params
    ) -> FeatureResult:
        """
        Detect anomalies in market data.
        
        Args:
            symbol: Trading pair
            exchange: Exchange (default: binance)
            data_type: prices, trades, funding_rates, liquidations
            hours: Hours of data to analyze
            method: zscore, iqr, isolation_forest, cusum, or ensemble
            zscore_threshold: Threshold for z-score method
            rolling_window: Window for rolling calculations
        """
        symbol_lower = symbol.lower()
        exc = (exchange or 'binance').lower()
        
        engine = get_timeseries_engine()
        
        # Build query based on data type
        table_suffix = data_type
        value_col = self._get_value_column(data_type)
        
        table_name = self.get_table_name(symbol_lower, exc, 'futures', table_suffix)
        
        if not self.table_exists(table_name):
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={},
                errors=[f"No {data_type} data found for {symbol} on {exc}"]
            )
        
        try:
            query = f"""
                SELECT timestamp, {value_col} as value
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
                ORDER BY timestamp
            """
            
            results = self.execute_query(query)
            
            if len(results) < 20:
                return self.create_result(
                    symbol=symbol,
                    exchanges=[exc],
                    data={},
                    errors=["Insufficient data for anomaly detection"]
                )
            
            # Convert to TimeSeriesData
            ts = TimeSeriesData.from_duckdb_result(results, name=data_type)
            
            # Run anomaly detection
            all_results = {}
            signals = []
            
            if method in ['zscore', 'ensemble']:
                result = engine.detect_anomalies_zscore(
                    ts, threshold=zscore_threshold, window=rolling_window
                )
                all_results['zscore'] = result.to_dict()
                
                if result.is_anomaly.iloc[-10:].any():
                    recent_anomalies = result.is_anomaly.iloc[-10:].sum()
                    signals.append(generate_signal(
                        'WARNING', min(recent_anomalies / 5, 1.0),
                        f"Z-score anomalies in recent {data_type}: {recent_anomalies} detected",
                        {'method': 'zscore', 'count': int(recent_anomalies)}
                    ))
            
            if method in ['iqr', 'ensemble']:
                result = engine.detect_anomalies_iqr(ts, window=rolling_window)
                all_results['iqr'] = result.to_dict()
            
            if method in ['isolation_forest', 'ensemble']:
                result = engine.detect_anomalies_isolation_forest(ts)
                all_results['isolation_forest'] = result.to_dict()
            
            if method in ['cusum', 'ensemble']:
                result = engine.detect_anomalies_cusum(ts)
                all_results['cusum'] = result.to_dict()
                
                if result.is_anomaly.iloc[-10:].any():
                    signals.append(generate_signal(
                        'WARNING', 0.8,
                        f"CUSUM detected mean shift in {data_type}",
                        {'method': 'cusum'}
                    ))
            
            # Ensemble summary
            if method == 'ensemble':
                total_anomalies = sum(r.get('anomaly_count', 0) for r in all_results.values())
                avg_rate = sum(r.get('anomaly_rate', 0) for r in all_results.values()) / len(all_results)
                
                all_results['ensemble_summary'] = {
                    'total_anomalies_across_methods': total_anomalies,
                    'average_anomaly_rate': avg_rate,
                    'consensus_level': self._calculate_consensus(all_results)
                }
            
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={
                    'data_type': data_type,
                    'detection_results': all_results,
                    'data_stats': {
                        'points_analyzed': len(ts),
                        'time_range_hours': hours,
                        'current_value': float(ts.value.iloc[-1])
                    }
                },
                metadata={
                    'method': method,
                    'zscore_threshold': zscore_threshold,
                    'rolling_window': rolling_window
                },
                signals=signals
            )
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={},
                errors=[str(e)]
            )
    
    def _get_value_column(self, data_type: str) -> str:
        """Get the appropriate value column for each data type."""
        columns = {
            'prices': 'mid_price',
            'trades': 'value',
            'funding_rates': 'funding_rate',
            'liquidations': 'value',
            'open_interest': 'open_interest'
        }
        return columns.get(data_type, 'value')
    
    def _calculate_consensus(self, results: Dict) -> str:
        """Calculate consensus across methods."""
        rates = [r.get('anomaly_rate', 0) for k, r in results.items() if k != 'ensemble_summary']
        
        if all(r > 0.05 for r in rates):
            return 'HIGH'  # All methods agree on anomalies
        elif any(r > 0.05 for r in rates):
            return 'MEDIUM'  # Some methods detect anomalies
        else:
            return 'LOW'  # No significant anomalies
    
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
                'description': 'Exchange for data',
                'required': False
            },
            'data_type': {
                'type': 'str',
                'default': 'prices',
                'description': 'Data type: prices, trades, funding_rates, liquidations',
                'required': False
            },
            'hours': {
                'type': 'int',
                'default': 24,
                'description': 'Hours of data to analyze',
                'required': False
            },
            'method': {
                'type': 'str',
                'default': 'ensemble',
                'description': 'Method: zscore, iqr, isolation_forest, cusum, ensemble',
                'required': False
            },
            'zscore_threshold': {
                'type': 'float',
                'default': 3.0,
                'description': 'Z-score threshold for anomaly',
                'required': False
            },
            'rolling_window': {
                'type': 'int',
                'default': 60,
                'description': 'Rolling window size',
                'required': False
            }
        }
