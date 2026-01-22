"""
💰 Funding Feature Calculator
============================

Calculates funding rate features including momentum, skew, carry.

Features:
- Funding rate analysis
- Cross-exchange funding spread
- Funding momentum
- Carry yield calculation
- Funding regime detection
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from ..base import InstitutionalFeatureCalculator, FeatureBuffer

logger = logging.getLogger(__name__)


class FundingFeatureCalculator(InstitutionalFeatureCalculator):
    """
    Real-time calculator for funding rate features.
    
    Input: Funding rate data (rate, timestamp, interval)
    Output: 13 institutional features for funding analysis
    
    Features:
    - Funding rate and derivatives
    - Cross-exchange spread
    - Funding momentum
    - Carry yield estimation
    - Funding regime (positive/negative/neutral)
    """
    
    feature_type = "funding"
    feature_count = 13
    requires_history = 24  # ~24 funding periods (8hr each = 3 days)
    
    def _init_buffers(self):
        """Initialize funding-specific buffers."""
        self.rate_buffer = self._create_buffer('rate', window_size=48)
        self.predicted_rate_buffer = self._create_buffer('predicted_rate', window_size=48)
        
        # Multi-exchange tracking
        self._exchange_rates: Dict[str, float] = {}
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate funding features.
        
        Args:
            data: Dict with funding rate info (rate, predicted_rate, exchange)
        
        Returns:
            Dict of funding features
        """
        # Extract funding data
        rate = float(data.get('funding_rate', data.get('rate', 0)))
        predicted_rate = float(data.get('predicted_rate', data.get('predicted_funding_rate', rate)))
        exchange = data.get('exchange', 'unknown')
        
        # Convert to percentage if needed (some exchanges use 0.0001 = 0.01%)
        if abs(rate) > 1:
            rate = rate / 10000
        if abs(predicted_rate) > 1:
            predicted_rate = predicted_rate / 10000
        
        # Update exchange rates
        self._exchange_rates[exchange] = rate
        
        # Update buffers
        self.rate_buffer.append(rate)
        self.predicted_rate_buffer.append(predicted_rate)
        
        # Calculate features
        rate_zscore = self.rate_buffer.zscore()
        rate_percentile = self.rate_buffer.percentile(rate)
        
        # Funding momentum
        momentum = self._calculate_funding_momentum()
        acceleration = self._calculate_funding_acceleration()
        
        # Funding skew (prediction vs actual)
        skew_index = predicted_rate - rate
        
        # Regime detection
        regime = self._detect_funding_regime()
        regime_score = self._calculate_regime_score(regime)
        
        # Cross-exchange spread
        funding_spread = self._calculate_cross_exchange_spread()
        
        # Carry yield (annualized)
        carry_yield = self._calculate_carry_yield(rate)
        carry_zscore = self._calculate_carry_zscore()
        
        # Extreme funding detection
        is_extreme = abs(rate_zscore) > 2.0
        extreme_direction = 1 if rate_zscore > 2.0 else (-1 if rate_zscore < -2.0 else 0)
        
        # Funding volatility
        funding_volatility = self.rate_buffer.std() if len(self.rate_buffer) >= 5 else 0.0
        
        return {
            # Core Funding
            'funding_rate': rate,
            'predicted_funding_rate': predicted_rate,
            'funding_zscore': rate_zscore,
            'funding_percentile': rate_percentile,
            
            # Momentum
            'funding_momentum': momentum,
            'funding_acceleration': acceleration,
            
            # Skew
            'funding_skew_index': skew_index,
            
            # Regime
            'funding_regime': regime,
            'funding_regime_score': regime_score,
            
            # Cross-Exchange
            'funding_cross_exchange_spread': funding_spread,
            
            # Carry
            'funding_carry_yield': carry_yield,
            'funding_carry_zscore': carry_zscore,
            
            # Volatility
            'funding_volatility': funding_volatility,
        }
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            'funding_rate', 'predicted_funding_rate', 'funding_zscore', 'funding_percentile',
            'funding_momentum', 'funding_acceleration',
            'funding_skew_index',
            'funding_regime', 'funding_regime_score',
            'funding_cross_exchange_spread',
            'funding_carry_yield', 'funding_carry_zscore',
            'funding_volatility',
        ]
    
    def update_exchange_rate(self, exchange: str, rate: float):
        """Update funding rate for a specific exchange."""
        if abs(rate) > 1:
            rate = rate / 10000
        self._exchange_rates[exchange] = rate
    
    def _calculate_funding_momentum(self) -> float:
        """
        Calculate funding momentum (rate of change).
        
        Positive momentum = funding getting more positive
        Negative momentum = funding getting more negative
        """
        return self.rate_buffer.velocity(periods=8)
    
    def _calculate_funding_acceleration(self) -> float:
        """Calculate funding acceleration (second derivative)."""
        if len(self.rate_buffer) < 16:
            return 0.0
        
        rates = self.rate_buffer.to_array()
        
        # First derivative (momentum)
        velocities = np.diff(rates[-16:])
        
        # Second derivative (acceleration)
        if len(velocities) < 2:
            return 0.0
        
        return velocities[-1] - velocities[-2]
    
    def _detect_funding_regime(self) -> str:
        """
        Detect funding regime.
        
        Returns: 'positive_extreme', 'positive', 'neutral', 'negative', 'negative_extreme'
        """
        if len(self.rate_buffer) < 5:
            return 'neutral'
        
        avg_rate = self.rate_buffer.mean()
        std_rate = self.rate_buffer.std()
        current_rate = self.rate_buffer.values[-1]
        
        if std_rate == 0:
            if current_rate > 0.001:
                return 'positive'
            elif current_rate < -0.001:
                return 'negative'
            return 'neutral'
        
        zscore = (current_rate - avg_rate) / std_rate
        
        if zscore > 2:
            return 'positive_extreme'
        elif zscore > 0.5:
            return 'positive'
        elif zscore < -2:
            return 'negative_extreme'
        elif zscore < -0.5:
            return 'negative'
        return 'neutral'
    
    def _calculate_regime_score(self, regime: str) -> float:
        """Convert regime to numeric score (-2 to 2)."""
        scores = {
            'positive_extreme': 2.0,
            'positive': 1.0,
            'neutral': 0.0,
            'negative': -1.0,
            'negative_extreme': -2.0,
        }
        return scores.get(regime, 0.0)
    
    def _calculate_cross_exchange_spread(self) -> float:
        """
        Calculate cross-exchange funding spread.
        
        Large spread = arbitrage opportunity
        """
        if len(self._exchange_rates) < 2:
            return 0.0
        
        rates = list(self._exchange_rates.values())
        return max(rates) - min(rates)
    
    def _calculate_carry_yield(self, rate: float) -> float:
        """
        Calculate annualized carry yield from funding rate.
        
        Assuming 8-hour funding intervals (3x per day):
        Annual = rate * 3 * 365
        """
        # Standard funding is 3x per day
        daily_rate = rate * 3
        annual_rate = daily_rate * 365
        
        return annual_rate
    
    def _calculate_carry_zscore(self) -> float:
        """Calculate z-score of current carry yield."""
        if len(self.rate_buffer) < 10:
            return 0.0
        
        rates = self.rate_buffer.to_array()
        carry_yields = [r * 3 * 365 for r in rates]
        
        current = carry_yields[-1]
        mean = np.mean(carry_yields)
        std = np.std(carry_yields)
        
        if std == 0:
            return 0.0
        
        return (current - mean) / std
    
    def get_cross_exchange_summary(self) -> Dict[str, Any]:
        """Get summary of cross-exchange funding rates."""
        if not self._exchange_rates:
            return {}
        
        rates = self._exchange_rates
        values = list(rates.values())
        
        return {
            'exchanges': dict(rates),
            'max': max(values),
            'min': min(values),
            'spread': max(values) - min(values),
            'mean': np.mean(values),
            'best_long': min(rates.items(), key=lambda x: x[1])[0],  # Lowest funding = best to long
            'best_short': max(rates.items(), key=lambda x: x[1])[0],  # Highest funding = best to short
        }


class FundingArbitrageCalculator(InstitutionalFeatureCalculator):
    """
    Calculator specifically for funding rate arbitrage signals.
    
    Monitors multiple exchanges for arbitrage opportunities.
    """
    
    feature_type = "funding_arbitrage"
    feature_count = 5
    requires_history = 10
    
    def _init_buffers(self):
        """Initialize arbitrage-specific buffers."""
        self.spread_buffer = self._create_buffer('spread')
        self._exchange_history: Dict[str, List[float]] = {}
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate funding arbitrage features."""
        exchange_rates = data.get('exchange_rates', {})
        
        if len(exchange_rates) < 2:
            return {
                'arbitrage_spread': 0.0,
                'arbitrage_opportunity': False,
                'long_exchange': '',
                'short_exchange': '',
                'expected_profit_bps': 0.0,
            }
        
        # Find best arbitrage pair
        rates_list = [(ex, r) for ex, r in exchange_rates.items()]
        rates_sorted = sorted(rates_list, key=lambda x: x[1])
        
        lowest = rates_sorted[0]  # Best to long (pay less funding)
        highest = rates_sorted[-1]  # Best to short (receive more funding)
        
        spread = highest[1] - lowest[1]
        self.spread_buffer.append(spread)
        
        # Arbitrage opportunity threshold (must cover trading costs)
        min_spread = 0.0001  # 0.01% minimum spread
        opportunity = spread > min_spread
        
        # Expected profit (spread * 3 periods per day * 365 days)
        expected_annual_profit_bps = spread * 3 * 365 * 10000  # In basis points
        
        return {
            'arbitrage_spread': spread,
            'arbitrage_opportunity': opportunity,
            'long_exchange': lowest[0],
            'short_exchange': highest[0],
            'expected_profit_bps': expected_annual_profit_bps,
        }
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            'arbitrage_spread', 'arbitrage_opportunity',
            'long_exchange', 'short_exchange', 'expected_profit_bps',
        ]
