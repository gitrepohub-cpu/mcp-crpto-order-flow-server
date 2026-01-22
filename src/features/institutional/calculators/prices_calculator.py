"""
💰 Prices Feature Calculator
============================

Calculates microprice, spread dynamics, and pressure features from price stream.

Features:
- Microprice & deviation
- Spread dynamics (compression, expansion, z-score)
- Pressure ratio (bid vs ask)
- Price efficiency metrics
- Hurst exponent (trend persistence)
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from ..base import InstitutionalFeatureCalculator, FeatureBuffer

logger = logging.getLogger(__name__)


class PricesFeatureCalculator(InstitutionalFeatureCalculator):
    """
    Real-time calculator for price stream features.
    
    Input: Price data from IsolatedDataCollector (mid, bid, ask)
    Output: 15+ institutional features
    
    Features:
    - Microprice (volume-weighted fair price)
    - Microprice deviation from mid
    - Spread analysis (bps, z-score, velocity)
    - Pressure ratio (bid/ask asymmetry)
    - Price efficiency metrics
    - Hurst exponent (trend persistence)
    """
    
    feature_type = "prices"
    feature_count = 19
    requires_history = 20
    
    def _init_buffers(self):
        """Initialize price-specific buffers."""
        self.mid_buffer = self._create_buffer('mid')
        self.bid_buffer = self._create_buffer('bid')
        self.ask_buffer = self._create_buffer('ask')
        self.spread_buffer = self._create_buffer('spread')
        self.microprice_buffer = self._create_buffer('microprice')
        self.pressure_buffer = self._create_buffer('pressure')
        
        # For VWAP calculation
        self.price_volume_buffer = self._create_buffer('price_volume', window_size=1000)
        self.volume_buffer = self._create_buffer('volume', window_size=1000)
        
        # Track tick direction for reversal rate
        self.tick_directions = self._create_buffer('tick_dir')
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate price features from incoming data.
        
        Args:
            data: Dict with 'price'/'mid_price', 'bid', 'ask', optionally 'bid_qty', 'ask_qty'
        
        Returns:
            Dict of price features
        """
        # Extract prices
        mid = data.get('price') or data.get('mid_price')
        bid = data.get('bid') or data.get('bid_price')
        ask = data.get('ask') or data.get('ask_price')
        
        if mid is None and bid and ask:
            mid = (bid + ask) / 2
        
        if mid is None:
            return {}
        
        # Get quantities for microprice (if available)
        bid_qty = data.get('bid_qty') or data.get('bid_quantity') or 1.0
        ask_qty = data.get('ask_qty') or data.get('ask_quantity') or 1.0
        
        # Calculate spread
        spread = (ask - bid) if (bid and ask) else 0.0
        spread_bps = (spread / mid * 10000) if (spread and mid > 0) else 0.0
        
        # Calculate microprice (volume-weighted fair price)
        # microprice = (ask * bid_qty + bid * ask_qty) / (bid_qty + ask_qty)
        total_qty = bid_qty + ask_qty
        microprice = (ask * bid_qty + bid * ask_qty) / total_qty if total_qty > 0 else mid
        microprice_deviation = microprice - mid if mid else 0.0
        
        # Calculate pressure ratio
        # Positive = buy pressure, Negative = sell pressure
        if bid and ask and mid:
            ask_distance = ask - mid
            bid_distance = mid - bid
            if bid_distance > 0:
                pressure_ratio = ask_distance / bid_distance
            else:
                pressure_ratio = 1.0
        else:
            pressure_ratio = 1.0
        
        # Update buffers
        self.mid_buffer.append(mid)
        if bid:
            self.bid_buffer.append(bid)
        if ask:
            self.ask_buffer.append(ask)
        self.spread_buffer.append(spread_bps)
        self.microprice_buffer.append(microprice)
        self.pressure_buffer.append(pressure_ratio)
        
        # Track tick direction for reversal rate
        if len(self.mid_buffer) >= 2:
            prev_mid = self.mid_buffer.last(2)[0]
            tick_dir = 1 if mid > prev_mid else (-1 if mid < prev_mid else 0)
            self.tick_directions.append(tick_dir)
        
        # Calculate rolling features
        features = {
            # Raw values
            'mid_price': mid,
            'bid_price': bid,
            'ask_price': ask,
            
            # Microprice features
            'microprice': microprice,
            'microprice_deviation': microprice_deviation,
            'microprice_zscore': self.microprice_buffer.zscore(microprice) if len(self.microprice_buffer) >= self.requires_history else 0.0,
            
            # Spread features
            'spread': spread,
            'spread_bps': spread_bps,
            'spread_zscore': self.spread_buffer.zscore(spread_bps) if len(self.spread_buffer) >= self.requires_history else 0.0,
            'spread_compression_velocity': self._calculate_spread_velocity(),
            'spread_expansion_spike': self._detect_spread_spike(spread_bps),
            
            # Pressure features
            'pressure_ratio': pressure_ratio,
            'bid_pressure': self._calculate_bid_pressure(),
            'ask_pressure': self._calculate_ask_pressure(),
            
            # Efficiency features
            'price_efficiency': self._calculate_price_efficiency(),
            'tick_reversal_rate': self._calculate_tick_reversal_rate(),
            'price_vs_vwap': self._calculate_price_vs_vwap(mid),
            'mid_price_entropy': self._calculate_price_entropy(),
            
            # Trend persistence
            'hurst_exponent': self._calculate_hurst() if len(self.mid_buffer) >= 50 else 0.5,
        }
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            'mid_price', 'bid_price', 'ask_price',
            'microprice', 'microprice_deviation', 'microprice_zscore',
            'spread', 'spread_bps', 'spread_zscore', 'spread_compression_velocity', 'spread_expansion_spike',
            'pressure_ratio', 'bid_pressure', 'ask_pressure',
            'price_efficiency', 'tick_reversal_rate', 'price_vs_vwap', 'mid_price_entropy',
            'hurst_exponent',
        ]
    
    def _calculate_spread_velocity(self) -> float:
        """Calculate rate of spread change (compression/expansion)."""
        if len(self.spread_buffer) < 5:
            return 0.0
        return self.spread_buffer.velocity(periods=5)
    
    def _detect_spread_spike(self, current_spread: float) -> float:
        """Detect if current spread is a spike (expansion)."""
        if len(self.spread_buffer) < self.requires_history:
            return 0.0
        
        zscore = self.spread_buffer.zscore(current_spread)
        return max(0.0, zscore - 2.0)  # Only positive spikes above 2 std
    
    def _calculate_bid_pressure(self) -> float:
        """Calculate bid-side pressure (0-1 scale)."""
        if len(self.pressure_buffer) < 5:
            return 0.5
        
        recent = self.pressure_buffer.last(5)
        avg_pressure = np.mean(recent)
        
        # Normalize: pressure_ratio < 1 = bid pressure, > 1 = ask pressure
        # Convert to 0-1 where 1 = max bid pressure
        return self.clip(1.0 - (avg_pressure / 2.0), 0.0, 1.0)
    
    def _calculate_ask_pressure(self) -> float:
        """Calculate ask-side pressure (0-1 scale)."""
        return 1.0 - self._calculate_bid_pressure()
    
    def _calculate_price_efficiency(self) -> float:
        """
        Calculate price efficiency (how well price follows fundamentals).
        
        High efficiency = prices move smoothly
        Low efficiency = prices are noisy/mean-reverting
        """
        if len(self.mid_buffer) < 20:
            return 0.5
        
        prices = self.mid_buffer.to_array()[-20:]
        
        # Calculate variance ratio
        returns = np.diff(prices) / prices[:-1]
        
        if len(returns) < 10:
            return 0.5
        
        # Variance of 1-period returns
        var_1 = np.var(returns)
        
        # Variance of 5-period returns (should be 5x if efficient)
        returns_5 = (prices[5:] - prices[:-5]) / prices[:-5]
        var_5 = np.var(returns_5)
        
        if var_1 == 0:
            return 0.5
        
        # Variance ratio (1 = efficient, <1 = mean reverting, >1 = trending)
        vr = var_5 / (5 * var_1)
        
        return self.clip(vr, 0.0, 2.0) / 2.0
    
    def _calculate_tick_reversal_rate(self) -> float:
        """
        Calculate how often price reverses direction.
        
        High reversal rate = choppy/mean-reverting
        Low reversal rate = trending
        """
        if len(self.tick_directions) < 10:
            return 0.5
        
        ticks = list(self.tick_directions.values)[-20:]
        ticks = [t for t in ticks if t != 0]  # Remove no-change ticks
        
        if len(ticks) < 5:
            return 0.5
        
        reversals = 0
        for i in range(1, len(ticks)):
            if ticks[i] != ticks[i-1]:
                reversals += 1
        
        return reversals / (len(ticks) - 1)
    
    def _calculate_price_vs_vwap(self, current_price: float) -> float:
        """
        Calculate price deviation from VWAP.
        
        Positive = above VWAP (bullish)
        Negative = below VWAP (bearish)
        """
        if len(self.price_volume_buffer) < 10 or len(self.volume_buffer) < 10:
            return 0.0
        
        # Simple price average (without actual volume data)
        prices = self.mid_buffer.to_array()
        if len(prices) < 10:
            return 0.0
        
        vwap = np.mean(prices)
        
        if vwap == 0:
            return 0.0
        
        return ((current_price - vwap) / vwap) * 100
    
    def _calculate_price_entropy(self) -> float:
        """Calculate entropy of price distribution."""
        if len(self.mid_buffer) < 30:
            return 0.0
        
        prices = self.mid_buffer.to_array()
        return self.calculate_entropy(prices, bins=10)
    
    def _calculate_hurst(self) -> float:
        """Calculate Hurst exponent for trend persistence."""
        if len(self.mid_buffer) < 50:
            return 0.5
        
        prices = self.mid_buffer.to_array()
        return self.calculate_hurst_exponent(prices, max_lag=20)
