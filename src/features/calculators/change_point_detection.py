"""
Change Point Detection Calculator

Detects structural breaks and regime changes in market data
using CUSUM and Binary Segmentation algorithms.
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


class ChangePointCalculator(FeatureCalculator):
    """
    Detect change points (structural breaks) in market data.
    
    Identifies:
        - Mean shifts
        - Trend changes
        - Volatility regime changes
    """
    
    name = "change_point_detection"
    description = "Detect structural breaks and regime changes in market data"
    category = "detection"
    version = "1.0.0"
    
    async def calculate(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        data_type: str = "prices",
        hours: int = 168,
        method: str = "both",
        resample_freq: str = "1h",
        **params
    ) -> FeatureResult:
        """
        Detect change points.
        
        Args:
            symbol: Trading pair
            exchange: Exchange (default: binance)
            data_type: prices, funding_rates, open_interest
            hours: Hours of data (default 7 days)
            method: cusum, binary_segmentation, or both
            resample_freq: Resample frequency
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
            
            if len(results) < 50:
                return self.create_result(
                    symbol=symbol,
                    exchanges=[exc],
                    data={},
                    errors=["Insufficient data for change point detection"]
                )
            
            # Convert and resample
            ts = TimeSeriesData.from_duckdb_result(results, name=data_type)
            ts = ts.resample(resample_freq)
            
            all_results = {}
            signals = []
            
            # CUSUM method
            if method in ['cusum', 'both']:
                cp_result = engine.detect_change_points_cusum(ts)
                all_results['cusum'] = cp_result.to_dict()
                
                # Check for recent change points
                recent_cps = [
                    cp for cp, ts_cp in zip(cp_result.change_points, cp_result.timestamps)
                    if cp > len(ts) - 24  # Last 24 periods
                ]
                
                if recent_cps:
                    signals.append(generate_signal(
                        'WARNING', 0.8,
                        f"Recent regime change detected via CUSUM",
                        {'method': 'cusum', 'recent_changes': len(recent_cps)}
                    ))
            
            # Binary Segmentation method
            if method in ['binary_segmentation', 'both']:
                cp_result = engine.detect_change_points_binary_segmentation(ts)
                all_results['binary_segmentation'] = cp_result.to_dict()
            
            # Analyze segments
            segment_analysis = self._analyze_segments(all_results)
            
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={
                    'data_type': data_type,
                    'change_points': all_results,
                    'segment_analysis': segment_analysis,
                    'data_stats': {
                        'points_analyzed': len(ts),
                        'resample_freq': resample_freq,
                        'time_range_hours': hours
                    }
                },
                metadata={
                    'method': method,
                    'resample_freq': resample_freq
                },
                signals=signals
            )
            
        except Exception as e:
            logger.error(f"Change point detection failed: {e}")
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
            'open_interest': 'open_interest'
        }
        return columns.get(data_type, 'mid_price')
    
    def _analyze_segments(self, results: Dict) -> Dict:
        """Analyze detected segments."""
        analysis = {}
        
        for method, result in results.items():
            segments = result.get('segments', [])
            if not segments:
                continue
            
            # Find largest segment
            largest = max(segments, key=lambda x: x.get('length', 0))
            
            # Find segment with highest/lowest mean
            highest_mean = max(segments, key=lambda x: x.get('mean', 0))
            lowest_mean = min(segments, key=lambda x: x.get('mean', 0))
            
            analysis[method] = {
                'num_segments': len(segments),
                'largest_segment': largest,
                'highest_mean_segment': highest_mean,
                'lowest_mean_segment': lowest_mean,
                'mean_range': highest_mean.get('mean', 0) - lowest_mean.get('mean', 0)
            }
        
        return analysis
    
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
                'description': 'Data type: prices, funding_rates, open_interest',
                'required': False
            },
            'hours': {
                'type': 'int',
                'default': 168,
                'description': 'Hours of data (default 7 days)',
                'required': False
            },
            'method': {
                'type': 'str',
                'default': 'both',
                'description': 'Method: cusum, binary_segmentation, both',
                'required': False
            },
            'resample_freq': {
                'type': 'str',
                'default': '1h',
                'description': 'Resample frequency',
                'required': False
            }
        }
