"""
Liquidation Cascade Detector

Analyzes liquidation patterns to detect potential cascade events
where liquidations trigger more liquidations.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from src.features.base import FeatureCalculator, FeatureResult
from src.features.utils import generate_signal, calculate_zscore

logger = logging.getLogger(__name__)


class LiquidationCascadeCalculator(FeatureCalculator):
    """
    Detects liquidation cascade patterns.
    
    Metrics:
        - Liquidation clustering (time-based)
        - Long vs short liquidation ratio
        - Cascade probability based on OI and price movement
        - Historical cascade pattern matching
    """
    
    name = "liquidation_cascade"
    description = "Detect liquidation cascade patterns and estimate cascade probability"
    category = "risk"
    version = "1.0.0"
    
    async def calculate(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        hours: int = 24,
        cascade_window_minutes: int = 5,
        min_liquidation_value: float = 10000,
        **params
    ) -> FeatureResult:
        """
        Detect liquidation cascades.
        
        Args:
            symbol: Trading pair
            exchange: Specific exchange or None for all
            hours: Hours of history
            cascade_window_minutes: Time window to detect clustering
            min_liquidation_value: Minimum USD value to include
        """
        symbol_lower = symbol.lower()
        exchanges = [exchange.lower()] if exchange else self.get_exchanges('futures')
        
        all_data = {}
        all_signals = []
        errors = []
        
        for exc in exchanges:
            table_name = self.get_table_name(symbol_lower, exc, 'futures', 'liquidations')
            
            if not self.table_exists(table_name):
                continue
            
            try:
                # Get liquidations
                query = f"""
                    SELECT timestamp, side, price, quantity, value
                    FROM {table_name}
                    WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
                    AND value >= {min_liquidation_value}
                    ORDER BY timestamp
                """
                
                liquidations = self.execute_query(query)
                
                if not liquidations:
                    continue
                
                # Convert to list of dicts
                liq_list = []
                for row in liquidations:
                    liq_list.append({
                        'timestamp': row[0],
                        'side': row[1],
                        'price': row[2],
                        'quantity': row[3],
                        'value': row[4]
                    })
                
                # Detect cascades (clusters of liquidations)
                cascades = self._detect_cascades(liq_list, cascade_window_minutes)
                
                # Summary statistics
                total_long = sum(l['value'] for l in liq_list if l['side'] == 'long')
                total_short = sum(l['value'] for l in liq_list if l['side'] == 'short')
                
                # Get current OI for context
                oi_data = self._get_current_oi(symbol_lower, exc)
                
                # Calculate cascade probability
                cascade_prob = self._calculate_cascade_probability(
                    cascades, liq_list, oi_data
                )
                
                all_data[exc] = {
                    'summary': {
                        'total_liquidations': len(liq_list),
                        'total_long_value': total_long,
                        'total_short_value': total_short,
                        'long_short_ratio': total_long / total_short if total_short > 0 else 0,
                        'dominant_side': 'LONG' if total_long > total_short else 'SHORT',
                        'avg_liquidation_size': sum(l['value'] for l in liq_list) / len(liq_list)
                    },
                    'cascades': {
                        'detected_count': len(cascades),
                        'total_cascade_value': sum(c['total_value'] for c in cascades),
                        'largest_cascade': max(cascades, key=lambda x: x['total_value']) if cascades else None,
                        'events': cascades[-5:]  # Last 5 cascades
                    },
                    'risk_metrics': {
                        'cascade_probability': cascade_prob,
                        'risk_level': self._get_risk_level(cascade_prob),
                        'current_oi': oi_data
                    }
                }
                
                # Generate signals
                if cascade_prob > 0.7:
                    all_signals.append(generate_signal(
                        'WARNING', cascade_prob,
                        f"High liquidation cascade risk on {exc} ({cascade_prob*100:.0f}%)",
                        {'exchange': exc, 'probability': cascade_prob}
                    ))
                
                # Check for active cascade
                recent_cascade = [c for c in cascades if c['is_recent']]
                if recent_cascade:
                    all_signals.append(generate_signal(
                        'WARNING', 0.9,
                        f"Active liquidation cascade detected on {exc}!",
                        {'exchange': exc, 'cascade': recent_cascade[-1]}
                    ))
                
            except Exception as e:
                logger.error(f"Error analyzing liquidations for {exc}: {e}")
                errors.append(f"{exc}: {str(e)}")
        
        return self.create_result(
            symbol=symbol,
            exchanges=list(all_data.keys()),
            data=all_data,
            metadata={
                'hours': hours,
                'cascade_window_minutes': cascade_window_minutes,
                'min_liquidation_value': min_liquidation_value
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
                'description': 'Hours of historical data',
                'required': False
            },
            'cascade_window_minutes': {
                'type': 'int',
                'default': 5,
                'description': 'Time window for cascade detection',
                'required': False
            },
            'min_liquidation_value': {
                'type': 'float',
                'default': 10000,
                'description': 'Minimum USD value to include',
                'required': False
            }
        }
    
    def _detect_cascades(
        self,
        liquidations: List[Dict],
        window_minutes: int
    ) -> List[Dict]:
        """Detect cascade events (clusters of liquidations)."""
        if not liquidations:
            return []
        
        cascades = []
        current_cascade = []
        window = timedelta(minutes=window_minutes)
        
        for liq in liquidations:
            if not current_cascade:
                current_cascade = [liq]
            else:
                time_diff = liq['timestamp'] - current_cascade[-1]['timestamp']
                if time_diff <= window:
                    current_cascade.append(liq)
                else:
                    # End current cascade if significant
                    if len(current_cascade) >= 3:  # At least 3 liquidations
                        cascades.append(self._summarize_cascade(current_cascade))
                    current_cascade = [liq]
        
        # Handle last cascade
        if len(current_cascade) >= 3:
            cascades.append(self._summarize_cascade(current_cascade))
        
        return cascades
    
    def _summarize_cascade(self, liquidations: List[Dict]) -> Dict:
        """Summarize a cascade event."""
        total_value = sum(l['value'] for l in liquidations)
        long_value = sum(l['value'] for l in liquidations if l['side'] == 'long')
        short_value = sum(l['value'] for l in liquidations if l['side'] == 'short')
        
        start_time = liquidations[0]['timestamp']
        end_time = liquidations[-1]['timestamp']
        duration = (end_time - start_time).total_seconds()
        
        # Check if recent (within last 30 minutes)
        now = datetime.utcnow()
        is_recent = (now - end_time.replace(tzinfo=None)).total_seconds() < 1800
        
        prices = [l['price'] for l in liquidations]
        
        return {
            'start_time': str(start_time),
            'end_time': str(end_time),
            'duration_seconds': duration,
            'liquidation_count': len(liquidations),
            'total_value': total_value,
            'long_value': long_value,
            'short_value': short_value,
            'dominant_side': 'LONG' if long_value > short_value else 'SHORT',
            'price_range': {
                'start': prices[0],
                'end': prices[-1],
                'low': min(prices),
                'high': max(prices)
            },
            'intensity': total_value / max(duration, 1),  # Value per second
            'is_recent': is_recent
        }
    
    def _get_current_oi(self, symbol: str, exchange: str) -> Dict:
        """Get current open interest data."""
        table_name = self.get_table_name(symbol, exchange, 'futures', 'open_interest')
        
        if not self.table_exists(table_name):
            return {}
        
        try:
            query = f"""
                SELECT open_interest, timestamp
                FROM {table_name}
                ORDER BY timestamp DESC
                LIMIT 1
            """
            result = self.execute_query(query)
            if result:
                return {
                    'value': result[0][0],
                    'timestamp': str(result[0][1])
                }
        except:
            pass
        
        return {}
    
    def _calculate_cascade_probability(
        self,
        cascades: List[Dict],
        all_liquidations: List[Dict],
        oi_data: Dict
    ) -> float:
        """Estimate probability of cascade based on current conditions."""
        probability = 0.0
        
        # Factor 1: Recent cascade activity (40% weight)
        recent_cascades = [c for c in cascades if c.get('is_recent')]
        if recent_cascades:
            probability += 0.4
        elif len(cascades) > 0:
            probability += 0.1 * min(len(cascades) / 5, 1)
        
        # Factor 2: Liquidation frequency (30% weight)
        if all_liquidations:
            hours_span = 24
            liqs_per_hour = len(all_liquidations) / hours_span
            # Higher frequency = higher risk
            probability += 0.3 * min(liqs_per_hour / 20, 1)  # 20 per hour = max
        
        # Factor 3: One-sided liquidations (30% weight)
        if all_liquidations:
            long_count = sum(1 for l in all_liquidations if l['side'] == 'long')
            short_count = len(all_liquidations) - long_count
            if long_count + short_count > 0:
                imbalance = abs(long_count - short_count) / (long_count + short_count)
                probability += 0.3 * imbalance
        
        return min(probability, 1.0)
    
    def _get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level."""
        if probability >= 0.8:
            return 'EXTREME'
        elif probability >= 0.6:
            return 'HIGH'
        elif probability >= 0.4:
            return 'ELEVATED'
        elif probability >= 0.2:
            return 'MODERATE'
        else:
            return 'LOW'
