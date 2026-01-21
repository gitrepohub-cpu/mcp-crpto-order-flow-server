"""
Order Flow Imbalance Calculator

Analyzes order flow to detect buying/selling pressure imbalances
that may indicate directional moves.
"""

import logging
from typing import Dict, List, Optional, Any

from src.features.base import FeatureCalculator, FeatureResult
from src.features.utils import (
    calculate_buy_sell_ratio,
    calculate_zscore,
    rolling_mean,
    generate_signal
)

logger = logging.getLogger(__name__)


class OrderFlowImbalanceCalculator(FeatureCalculator):
    """
    Calculates order flow imbalance metrics from trade data.
    
    Metrics:
        - Buy/sell volume ratio
        - Net flow (buy - sell volume)
        - Flow momentum (rate of change)
        - Imbalance z-score vs historical
        - Large trade clustering
    """
    
    name = "order_flow_imbalance"
    description = "Analyze order flow to detect buying/selling pressure imbalances"
    category = "order_flow"
    version = "1.0.0"
    
    async def calculate(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        hours: int = 24,
        bucket_minutes: int = 15,
        **params
    ) -> FeatureResult:
        """
        Calculate order flow imbalance.
        
        Args:
            symbol: Trading pair
            exchange: Specific exchange or None for all
            hours: Hours of history to analyze
            bucket_minutes: Time bucket size for aggregation
        """
        symbol_lower = symbol.lower()
        exchanges = [exchange.lower()] if exchange else self.get_exchanges('futures')
        
        all_data = {}
        all_signals = []
        errors = []
        
        for exc in exchanges:
            table_name = self.get_table_name(symbol_lower, exc, 'futures', 'trades')
            
            if not self.table_exists(table_name):
                continue
            
            try:
                # Get aggregated trade data by time bucket
                query = f"""
                    SELECT 
                        TIME_BUCKET(INTERVAL '{bucket_minutes} minutes', timestamp) as bucket,
                        SUM(CASE WHEN side = 'buy' THEN value ELSE 0 END) as buy_volume,
                        SUM(CASE WHEN side = 'sell' THEN value ELSE 0 END) as sell_volume,
                        COUNT(*) as trade_count,
                        AVG(price) as avg_price
                    FROM {table_name}
                    WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
                    GROUP BY bucket
                    ORDER BY bucket
                """
                
                results = self.execute_query(query)
                
                if not results:
                    continue
                
                buckets = []
                buy_volumes = []
                sell_volumes = []
                net_flows = []
                
                for row in results:
                    bucket_time, buy_vol, sell_vol, count, avg_price = row
                    net_flow = buy_vol - sell_vol
                    ratio = buy_vol / sell_vol if sell_vol > 0 else float('inf')
                    
                    buckets.append({
                        'timestamp': str(bucket_time),
                        'buy_volume': buy_vol,
                        'sell_volume': sell_vol,
                        'net_flow': net_flow,
                        'ratio': min(ratio, 100),  # Cap for display
                        'trade_count': count,
                        'avg_price': avg_price
                    })
                    
                    buy_volumes.append(buy_vol)
                    sell_volumes.append(sell_vol)
                    net_flows.append(net_flow)
                
                # Calculate summary metrics
                total_buy = sum(buy_volumes)
                total_sell = sum(sell_volumes)
                avg_net_flow = sum(net_flows) / len(net_flows) if net_flows else 0
                
                # Recent vs historical comparison
                recent_buckets = buckets[-4:] if len(buckets) >= 4 else buckets  # Last hour
                recent_net = sum(b['net_flow'] for b in recent_buckets)
                
                historical_mean = sum(net_flows[:-4]) / len(net_flows[:-4]) if len(net_flows) > 4 else 0
                historical_std = (sum((x - historical_mean)**2 for x in net_flows[:-4]) / len(net_flows[:-4]))**0.5 if len(net_flows) > 4 else 1
                
                zscore = calculate_zscore(recent_net, historical_mean, historical_std)
                
                all_data[exc] = {
                    'summary': {
                        'total_buy_volume': total_buy,
                        'total_sell_volume': total_sell,
                        'net_flow': total_buy - total_sell,
                        'buy_sell_ratio': total_buy / total_sell if total_sell > 0 else 0,
                        'avg_bucket_net_flow': avg_net_flow,
                        'bucket_count': len(buckets)
                    },
                    'recent_analysis': {
                        'recent_net_flow': recent_net,
                        'historical_mean': historical_mean,
                        'zscore': zscore,
                        'percentile': self._zscore_to_percentile(zscore)
                    },
                    'time_series': buckets[-20:]  # Last 20 buckets
                }
                
                # Generate signals
                if zscore > 2:
                    all_signals.append(generate_signal(
                        'BULLISH', min(zscore / 3, 1.0),
                        f"Strong buying pressure on {exc} (z-score: {zscore:.2f})",
                        {'exchange': exc, 'zscore': zscore}
                    ))
                elif zscore < -2:
                    all_signals.append(generate_signal(
                        'BEARISH', min(abs(zscore) / 3, 1.0),
                        f"Strong selling pressure on {exc} (z-score: {zscore:.2f})",
                        {'exchange': exc, 'zscore': zscore}
                    ))
                
            except Exception as e:
                logger.error(f"Error calculating order flow for {exc}: {e}")
                errors.append(f"{exc}: {str(e)}")
        
        # Cross-exchange analysis
        if len(all_data) > 1:
            cross_exchange = self._analyze_cross_exchange(all_data)
            all_data['cross_exchange'] = cross_exchange
        
        return self.create_result(
            symbol=symbol,
            exchanges=list(all_data.keys()),
            data=all_data,
            metadata={
                'hours': hours,
                'bucket_minutes': bucket_minutes,
                'calculation': 'order_flow_imbalance'
            },
            signals=all_signals,
            errors=errors
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
                'default': None,
                'description': 'Specific exchange or None for all',
                'required': False
            },
            'hours': {
                'type': 'int',
                'default': 24,
                'description': 'Hours of historical data to analyze',
                'required': False
            },
            'bucket_minutes': {
                'type': 'int',
                'default': 15,
                'description': 'Time bucket size in minutes for aggregation',
                'required': False
            }
        }
    
    def _zscore_to_percentile(self, zscore: float) -> float:
        """Approximate percentile from z-score."""
        # Simplified conversion
        if zscore >= 3:
            return 99.9
        elif zscore >= 2:
            return 97.7
        elif zscore >= 1:
            return 84.1
        elif zscore >= 0:
            return 50 + (zscore * 34.1)
        elif zscore >= -1:
            return 50 + (zscore * 34.1)
        elif zscore >= -2:
            return 2.3
        else:
            return 0.1
    
    def _analyze_cross_exchange(self, data: Dict) -> Dict:
        """Analyze order flow consistency across exchanges."""
        net_flows = {}
        for exc, exc_data in data.items():
            if exc == 'cross_exchange':
                continue
            summary = exc_data.get('summary', {})
            net_flows[exc] = summary.get('net_flow', 0)
        
        if not net_flows:
            return {}
        
        # Check if all exchanges agree on direction
        directions = [1 if nf > 0 else -1 for nf in net_flows.values()]
        consensus = sum(directions) / len(directions) if directions else 0
        
        return {
            'net_flows_by_exchange': net_flows,
            'consensus_score': abs(consensus),  # 0-1, 1 = all agree
            'dominant_direction': 'BUY' if consensus > 0 else 'SELL',
            'agreement': 'STRONG' if abs(consensus) > 0.8 else 'MIXED'
        }
