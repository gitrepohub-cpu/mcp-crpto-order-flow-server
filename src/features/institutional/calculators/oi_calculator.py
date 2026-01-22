"""
📈 Open Interest Feature Calculator
===================================

Calculates open interest features including leverage index, cascade risk.

Features:
- OI delta and derivatives
- Leverage index
- Position intent inference
- Liquidation cascade risk
- OI-price divergence
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from ..base import InstitutionalFeatureCalculator, FeatureBuffer

logger = logging.getLogger(__name__)


class OIFeatureCalculator(InstitutionalFeatureCalculator):
    """
    Real-time calculator for open interest features.
    
    Input: OI data (oi_value, price, timestamp)
    Output: 17 institutional features for positioning analysis
    
    Features:
    - OI delta and momentum
    - Leverage index
    - Position intent (accumulation vs distribution)
    - Liquidation cascade risk
    - OI-price divergence
    """
    
    feature_type = "oi"
    feature_count = 17
    requires_history = 50
    
    def _init_buffers(self):
        """Initialize OI-specific buffers."""
        self.oi_buffer = self._create_buffer('oi', window_size=100)
        self.oi_delta_buffer = self._create_buffer('oi_delta')
        self.price_buffer = self._create_buffer('price')
        self.leverage_buffer = self._create_buffer('leverage')
        
        # Previous OI for delta calculation
        self._prev_oi: float = 0.0
        self._prev_price: float = 0.0
        
        # Market cap reference for leverage index
        self._market_cap_estimate: float = 0.0
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate OI features.
        
        Args:
            data: Dict with OI info (oi, oi_value, price)
        
        Returns:
            Dict of OI features
        """
        # Extract OI data (different exchanges use different field names)
        oi = float(data.get('oi', data.get('open_interest', data.get('openInterest', 0))))
        oi_value = float(data.get('oi_value', data.get('openInterestValue', oi)))
        price = float(data.get('price', data.get('mark_price', data.get('last_price', 0))))
        
        if oi <= 0:
            return {}
        
        # Calculate OI delta
        oi_delta = oi - self._prev_oi if self._prev_oi > 0 else 0
        oi_delta_pct = (oi_delta / self._prev_oi * 100) if self._prev_oi > 0 else 0
        
        # Price delta
        price_delta = price - self._prev_price if self._prev_price > 0 else 0
        price_delta_pct = (price_delta / self._prev_price * 100) if self._prev_price > 0 else 0
        
        # Update buffers
        self.oi_buffer.append(oi)
        self.oi_delta_buffer.append(oi_delta)
        if price > 0:
            self.price_buffer.append(price)
        
        # OI-Price delta product (position intent signal)
        oi_price_product = oi_delta * price_delta
        
        # Calculate features
        oi_zscore = self.oi_buffer.zscore()
        oi_momentum = self.oi_buffer.velocity(periods=10)
        oi_acceleration = self._calculate_oi_acceleration()
        
        # Leverage index
        leverage_index = self._calculate_leverage_index(oi_value, price)
        self.leverage_buffer.append(leverage_index)
        leverage_zscore = self.leverage_buffer.zscore()
        
        # Position intent
        intent = self._classify_position_intent(oi_delta, price_delta)
        intent_score = self._calculate_intent_score(intent)
        
        # Liquidation cascade risk
        cascade_risk = self._calculate_cascade_risk()
        
        # OI-price divergence
        divergence = self._calculate_oi_price_divergence()
        
        # OI concentration
        concentration = self._calculate_oi_concentration()
        
        # OI volatility
        oi_volatility = self.oi_buffer.std() / self.oi_buffer.mean() if self.oi_buffer.mean() > 0 else 0
        
        # Long/short ratio estimation (requires additional data)
        ls_ratio = data.get('long_short_ratio', data.get('ls_ratio', 1.0))
        ls_ratio = float(ls_ratio) if ls_ratio else 1.0
        
        # Position crowding score
        crowding_score = self._calculate_crowding_score(ls_ratio)
        
        # Store current values for next calculation
        self._prev_oi = oi
        if price > 0:
            self._prev_price = price
        
        return {
            # Core OI
            'oi': oi,
            'oi_value': oi_value,
            'oi_delta': oi_delta,
            'oi_delta_pct': oi_delta_pct,
            'oi_zscore': oi_zscore,
            
            # Momentum
            'oi_momentum': oi_momentum,
            'oi_acceleration': oi_acceleration,
            
            # OI-Price Relationship
            'oi_price_delta_product': oi_price_product,
            'oi_price_divergence': divergence,
            
            # Leverage
            'leverage_index': leverage_index,
            'leverage_zscore': leverage_zscore,
            
            # Position Analysis
            'position_intent': intent,
            'position_intent_score': intent_score,
            'long_short_ratio': ls_ratio,
            'position_crowding_score': crowding_score,
            
            # Risk
            'liquidation_cascade_risk': cascade_risk,
            
            # Quality
            'oi_concentration': concentration,
        }
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            'oi', 'oi_value', 'oi_delta', 'oi_delta_pct', 'oi_zscore',
            'oi_momentum', 'oi_acceleration',
            'oi_price_delta_product', 'oi_price_divergence',
            'leverage_index', 'leverage_zscore',
            'position_intent', 'position_intent_score', 'long_short_ratio', 'position_crowding_score',
            'liquidation_cascade_risk',
            'oi_concentration',
        ]
    
    def _calculate_oi_acceleration(self) -> float:
        """Calculate OI acceleration (second derivative)."""
        if len(self.oi_delta_buffer) < 10:
            return 0.0
        
        deltas = self.oi_delta_buffer.to_array()[-10:]
        
        if len(deltas) < 3:
            return 0.0
        
        # Second derivative
        return deltas[-1] - deltas[-2]
    
    def _calculate_leverage_index(self, oi_value: float, price: float) -> float:
        """
        Calculate leverage index.
        
        Estimates average leverage based on OI relative to estimated notional.
        Higher index = more leveraged market
        """
        if oi_value <= 0 or price <= 0:
            return 0.0
        
        # Simple proxy: OI value relative to recent OI average
        if len(self.oi_buffer) < 10:
            return 1.0
        
        avg_oi = self.oi_buffer.mean()
        if avg_oi == 0:
            return 1.0
        
        # Leverage index = current OI / average OI
        # > 1 means more leveraged than usual
        return self.oi_buffer.values[-1] / avg_oi
    
    def _classify_position_intent(self, oi_delta: float, price_delta: float) -> str:
        """
        Classify position intent based on OI and price changes.
        
        OI↑ Price↑ = Long accumulation
        OI↑ Price↓ = Short accumulation
        OI↓ Price↓ = Long liquidation
        OI↓ Price↑ = Short liquidation
        """
        oi_up = oi_delta > 0
        price_up = price_delta > 0
        
        # Thresholds for significant changes
        oi_significant = abs(oi_delta) > 0
        price_significant = abs(price_delta) > 0
        
        if not oi_significant or not price_significant:
            return 'neutral'
        
        if oi_up and price_up:
            return 'long_accumulation'
        elif oi_up and not price_up:
            return 'short_accumulation'
        elif not oi_up and not price_up:
            return 'long_liquidation'
        elif not oi_up and price_up:
            return 'short_liquidation'
        
        return 'neutral'
    
    def _calculate_intent_score(self, intent: str) -> float:
        """Convert intent to numeric score."""
        scores = {
            'long_accumulation': 1.0,
            'short_accumulation': -1.0,
            'long_liquidation': -0.5,  # Bearish but positions closing
            'short_liquidation': 0.5,  # Bullish but positions closing
            'neutral': 0.0,
        }
        return scores.get(intent, 0.0)
    
    def _calculate_cascade_risk(self) -> float:
        """
        Calculate liquidation cascade risk.
        
        High risk = large OI + high leverage + recent price volatility
        """
        if len(self.oi_buffer) < 10 or len(self.price_buffer) < 10:
            return 0.0
        
        # OI component (normalized)
        oi_zscore = abs(self.oi_buffer.zscore())
        
        # Leverage component
        leverage_zscore = abs(self.leverage_buffer.zscore()) if len(self.leverage_buffer) >= 5 else 0
        
        # Price volatility component
        price_vol = self.price_buffer.std() / self.price_buffer.mean() if self.price_buffer.mean() > 0 else 0
        
        # Combined risk score (0-1)
        risk = (oi_zscore * 0.3 + leverage_zscore * 0.4 + price_vol * 10 * 0.3)
        
        return min(1.0, risk / 3)
    
    def _calculate_oi_price_divergence(self) -> float:
        """
        Calculate OI-price divergence.
        
        Divergence when OI and price move in opposite directions.
        Strong signal for potential reversal.
        """
        if len(self.oi_buffer) < 10 or len(self.price_buffer) < 10:
            return 0.0
        
        oi_values = self.oi_buffer.to_array()[-10:]
        price_values = self.price_buffer.to_array()[-10:]
        
        if len(oi_values) < 3 or len(price_values) < 3:
            return 0.0
        
        # Normalize
        oi_norm = (oi_values - np.mean(oi_values))
        price_norm = (price_values - np.mean(price_values))
        
        std_oi = np.std(oi_norm)
        std_price = np.std(price_norm)
        
        if std_oi == 0 or std_price == 0:
            return 0.0
        
        # Correlation
        correlation = np.mean(oi_norm * price_norm) / (std_oi * std_price)
        
        # Divergence = negative correlation
        return -correlation
    
    def _calculate_oi_concentration(self) -> float:
        """
        Estimate OI concentration (how concentrated positions are).
        
        Based on OI changes - erratic changes suggest dispersed positions.
        """
        if len(self.oi_delta_buffer) < 10:
            return 0.5
        
        deltas = self.oi_delta_buffer.to_array()
        
        # High variance in deltas = dispersed positions
        # Low variance = concentrated
        cv = np.std(deltas) / (np.mean(np.abs(deltas)) + 1e-9)
        
        # Invert: high CV = low concentration
        return max(0.0, 1.0 - cv)
    
    def _calculate_crowding_score(self, ls_ratio: float) -> float:
        """
        Calculate position crowding score.
        
        Extreme L/S ratios indicate crowded trades.
        Crowded = higher reversal risk.
        """
        if ls_ratio <= 0:
            return 0.0
        
        # Neutral is 1.0
        # Calculate deviation from neutral
        deviation = abs(np.log(ls_ratio))  # Log scale for symmetry
        
        # Normalize to 0-1
        return min(1.0, deviation / 2)
    
    def set_market_cap(self, market_cap: float):
        """Set market cap for improved leverage calculations."""
        self._market_cap_estimate = market_cap
