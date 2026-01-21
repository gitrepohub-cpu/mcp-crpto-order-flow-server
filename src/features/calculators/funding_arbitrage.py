"""
Funding Rate Arbitrage Calculator

Identifies funding rate arbitrage opportunities across exchanges
for delta-neutral carry trades.
"""

import logging
from typing import Dict, List, Optional, Any

from src.features.base import FeatureCalculator, FeatureResult
from src.features.utils import generate_signal, calculate_zscore

logger = logging.getLogger(__name__)


class FundingArbitrageCalculator(FeatureCalculator):
    """
    Calculates funding rate arbitrage opportunities.
    
    Metrics:
        - Cross-exchange funding spreads
        - Historical funding patterns
        - Annualized carry yield
        - Entry timing signals
    """
    
    name = "funding_arbitrage"
    description = "Identify funding rate arbitrage opportunities across exchanges"
    category = "arbitrage"
    version = "1.0.0"
    
    async def calculate(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        hours: int = 72,
        min_spread_bps: float = 5,
        **params
    ) -> FeatureResult:
        """
        Calculate funding arbitrage opportunities.
        
        Args:
            symbol: Trading pair
            exchange: Specific exchange or None for all
            hours: Hours of history
            min_spread_bps: Minimum spread in basis points to flag
        """
        symbol_lower = symbol.lower()
        exchanges = [exchange.lower()] if exchange else self.get_exchanges('futures')
        
        exchange_data = {}
        errors = []
        
        # Collect funding data from all exchanges
        for exc in exchanges:
            table_name = self.get_table_name(symbol_lower, exc, 'futures', 'funding_rates')
            
            if not self.table_exists(table_name):
                continue
            
            try:
                query = f"""
                    SELECT 
                        timestamp,
                        funding_rate,
                        next_funding_time
                    FROM {table_name}
                    WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
                    ORDER BY timestamp DESC
                """
                
                results = self.execute_query(query)
                
                if not results:
                    continue
                
                rates = [r[1] for r in results]
                
                exchange_data[exc] = {
                    'current_rate': rates[0] if rates else 0,
                    'avg_rate': sum(rates) / len(rates) if rates else 0,
                    'min_rate': min(rates) if rates else 0,
                    'max_rate': max(rates) if rates else 0,
                    'rate_volatility': self._calculate_rate_volatility(rates),
                    'cumulative_rate': sum(rates),
                    'sample_count': len(rates),
                    'annualized_yield': self._annualize_rate(sum(rates) / len(rates) if rates else 0),
                    'history': [
                        {'timestamp': str(r[0]), 'rate': r[1]}
                        for r in results[:24]  # Last 24 data points
                    ]
                }
                
            except Exception as e:
                logger.error(f"Error getting funding for {exc}: {e}")
                errors.append(f"{exc}: {str(e)}")
        
        # Find arbitrage opportunities
        opportunities = self._find_opportunities(exchange_data, min_spread_bps)
        signals = []
        
        for opp in opportunities:
            if opp['annualized_spread'] > 10:  # >10% annualized
                signals.append(generate_signal(
                    'BULLISH', min(opp['annualized_spread'] / 50, 1.0),
                    f"Funding arb: Long {opp['long_exchange']} / Short {opp['short_exchange']} "
                    f"({opp['annualized_spread']:.1f}% annualized)",
                    opp
                ))
        
        # Historical pattern analysis
        patterns = self._analyze_patterns(exchange_data)
        
        return self.create_result(
            symbol=symbol,
            exchanges=list(exchange_data.keys()),
            data={
                'exchange_funding': exchange_data,
                'opportunities': opportunities,
                'patterns': patterns,
                'best_opportunity': opportunities[0] if opportunities else None
            },
            metadata={
                'hours': hours,
                'min_spread_bps': min_spread_bps,
                'exchanges_analyzed': len(exchange_data)
            },
            signals=signals,
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
                'default': 72,
                'description': 'Hours of funding history to analyze',
                'required': False
            },
            'min_spread_bps': {
                'type': 'float',
                'default': 5,
                'description': 'Minimum spread in basis points to flag',
                'required': False
            }
        }
    
    def _calculate_rate_volatility(self, rates: List[float]) -> float:
        """Calculate volatility of funding rates."""
        if len(rates) < 2:
            return 0.0
        
        mean = sum(rates) / len(rates)
        variance = sum((r - mean) ** 2 for r in rates) / len(rates)
        return variance ** 0.5
    
    def _annualize_rate(self, rate: float) -> float:
        """Annualize funding rate (assumes 8-hour funding)."""
        # 3 funding periods per day * 365 days
        return rate * 3 * 365 * 100
    
    def _find_opportunities(
        self,
        exchange_data: Dict,
        min_spread_bps: float
    ) -> List[Dict]:
        """Find funding arbitrage opportunities."""
        opportunities = []
        exchanges = list(exchange_data.keys())
        
        for i, exc1 in enumerate(exchanges):
            for exc2 in exchanges[i+1:]:
                rate1 = exchange_data[exc1]['current_rate']
                rate2 = exchange_data[exc2]['current_rate']
                
                spread = abs(rate1 - rate2)
                spread_bps = spread * 10000
                
                if spread_bps >= min_spread_bps:
                    # Determine direction
                    if rate1 < rate2:
                        long_exc, short_exc = exc1, exc2
                    else:
                        long_exc, short_exc = exc2, exc1
                    
                    annualized = self._annualize_rate(spread)
                    
                    # Check historical consistency
                    avg1 = exchange_data[exc1]['avg_rate']
                    avg2 = exchange_data[exc2]['avg_rate']
                    historical_spread = abs(avg1 - avg2)
                    
                    opportunities.append({
                        'long_exchange': long_exc,
                        'short_exchange': short_exc,
                        'spread': spread,
                        'spread_bps': spread_bps,
                        'annualized_spread': annualized,
                        'historical_spread': historical_spread * 10000,
                        'historical_annualized': self._annualize_rate(historical_spread),
                        'confidence': 'HIGH' if spread_bps > historical_spread * 10000 else 'MEDIUM',
                        'strategy': f"Long {long_exc} perp + Short {short_exc} perp"
                    })
        
        # Sort by annualized return
        opportunities.sort(key=lambda x: x['annualized_spread'], reverse=True)
        return opportunities
    
    def _analyze_patterns(self, exchange_data: Dict) -> Dict:
        """Analyze historical funding patterns."""
        patterns = {
            'persistently_positive': [],
            'persistently_negative': [],
            'volatile': [],
            'stable': []
        }
        
        for exc, data in exchange_data.items():
            avg = data['avg_rate']
            vol = data['rate_volatility']
            
            # Classify exchange
            if avg > 0.0001:  # Consistently positive
                patterns['persistently_positive'].append(exc)
            elif avg < -0.0001:  # Consistently negative
                patterns['persistently_negative'].append(exc)
            
            if vol > 0.0002:  # High volatility
                patterns['volatile'].append(exc)
            else:
                patterns['stable'].append(exc)
        
        return {
            'classifications': patterns,
            'recommendation': self._generate_recommendation(patterns, exchange_data)
        }
    
    def _generate_recommendation(self, patterns: Dict, data: Dict) -> str:
        """Generate trading recommendation."""
        positive = patterns['persistently_positive']
        negative = patterns['persistently_negative']
        
        if positive and negative:
            return f"Strong arb setup: Long on {', '.join(negative)} (negative funding), Short on {', '.join(positive)} (positive funding)"
        elif positive:
            return f"Consider shorting perpetuals on {', '.join(positive)} to collect funding"
        elif negative:
            return f"Consider longing perpetuals on {', '.join(negative)} to collect funding"
        else:
            return "No clear funding bias detected across exchanges"
