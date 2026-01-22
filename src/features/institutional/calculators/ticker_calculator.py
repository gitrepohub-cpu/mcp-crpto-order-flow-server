"""
📈 Ticker Feature Calculator
============================

Calculates 24h ticker features including volume analysis, range, volatility.

Features:
- Volume acceleration and profile
- Range expansion/compression
- Volatility metrics
- Institutional interest indicators
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from ..base import InstitutionalFeatureCalculator, FeatureBuffer

logger = logging.getLogger(__name__)


class TickerFeatureCalculator(InstitutionalFeatureCalculator):
    """
    Real-time calculator for 24h ticker features.
    
    Input: Ticker data (24h stats: volume, high, low, open, close, trades)
    Output: 15 institutional features for market structure analysis
    
    Features:
    - Volume analysis (acceleration, relative volume, profile)
    - Range analysis (expansion, ATR comparison)
    - Volatility metrics (compression, expansion)
    - Institutional interest indicators
    """
    
    feature_type = "ticker"
    feature_count = 15
    requires_history = 24  # ~24 hours of ticker updates
    
    def _init_buffers(self):
        """Initialize ticker-specific buffers."""
        self.volume_buffer = self._create_buffer('volume', window_size=48)
        self.range_buffer = self._create_buffer('range', window_size=48)
        self.trade_count_buffer = self._create_buffer('trade_count', window_size=48)
        self.price_change_buffer = self._create_buffer('price_change', window_size=48)
        
        # Historical ATR for comparison
        self._atr_history: List[float] = []
        
        # Previous values
        self._prev_volume: float = 0.0
        self._prev_trade_count: int = 0
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate ticker features.
        
        Args:
            data: Dict with 24h ticker stats
                  volume, high, low, open, close, trades/count
        
        Returns:
            Dict of ticker features
        """
        # Extract ticker data (handle different exchange formats)
        volume = float(data.get('volume', data.get('volume24h', data.get('quoteVolume', 0))))
        high = float(data.get('high', data.get('high24h', data.get('highPrice', 0))))
        low = float(data.get('low', data.get('low24h', data.get('lowPrice', 0))))
        open_price = float(data.get('open', data.get('openPrice', 0)))
        close_price = float(data.get('close', data.get('lastPrice', data.get('last', 0))))
        trade_count = int(data.get('trades', data.get('count', data.get('tradeCount', 0))))
        
        # Validate data
        if volume <= 0 or high <= 0 or low <= 0:
            return {}
        
        # Calculate range
        range_value = high - low
        range_pct = (range_value / low * 100) if low > 0 else 0.0
        
        # Calculate price change
        price_change = close_price - open_price if open_price > 0 else 0.0
        price_change_pct = (price_change / open_price * 100) if open_price > 0 else 0.0
        
        # Update buffers
        self.volume_buffer.append(volume)
        self.range_buffer.append(range_pct)
        if trade_count > 0:
            self.trade_count_buffer.append(trade_count)
        self.price_change_buffer.append(price_change_pct)
        
        # Calculate features
        
        # Volume features
        volume_acceleration = self._calculate_volume_acceleration(volume)
        relative_volume = self._calculate_relative_volume(volume)
        volume_profile_skew = self._calculate_volume_profile_skew()
        
        # Range features
        range_expansion_ratio = self._calculate_range_expansion(range_pct)
        range_vs_atr = self._calculate_range_vs_atr(range_pct)
        
        # Volatility features
        volatility_compression = self._calculate_volatility_compression()
        volatility_expansion = self._calculate_volatility_expansion()
        realized_volatility = self._calculate_realized_volatility()
        
        # Price change features
        price_change_zscore = self.price_change_buffer.zscore()
        
        # Participation features
        trade_count_percentile = self._calculate_trade_count_percentile(trade_count)
        institutional_interest = self._calculate_institutional_interest(volume, trade_count)
        
        # Market structure
        market_strength = self._calculate_market_strength(price_change_pct, volume)
        
        # Update previous values
        self._prev_volume = volume
        self._prev_trade_count = trade_count
        
        # Update ATR history
        if range_pct > 0:
            self._atr_history.append(range_pct)
            if len(self._atr_history) > 14:
                self._atr_history.pop(0)
        
        return {
            # Volume Features
            'volume_acceleration': volume_acceleration,
            'relative_volume_percentile': relative_volume,
            'volume_profile_skew': volume_profile_skew,
            
            # Range Features
            'range_expansion_ratio': range_expansion_ratio,
            'high_low_range': range_value,
            'high_low_range_pct': range_pct,
            'range_vs_atr': range_vs_atr,
            
            # Volatility Features
            'volatility_compression_idx': volatility_compression,
            'volatility_expansion_idx': volatility_expansion,
            'realized_volatility': realized_volatility,
            
            # Price Change
            'price_change_pct': price_change_pct,
            'price_change_zscore': price_change_zscore,
            
            # Participation
            'trade_count_percentile': trade_count_percentile,
            'institutional_interest_idx': institutional_interest,
            
            # Market Structure
            'market_strength': market_strength,
        }
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            'volume_acceleration', 'relative_volume_percentile', 'volume_profile_skew',
            'range_expansion_ratio', 'high_low_range', 'high_low_range_pct', 'range_vs_atr',
            'volatility_compression_idx', 'volatility_expansion_idx', 'realized_volatility',
            'price_change_pct', 'price_change_zscore',
            'trade_count_percentile', 'institutional_interest_idx',
            'market_strength',
        ]
    
    def _calculate_volume_acceleration(self, current_volume: float) -> float:
        """
        Calculate volume acceleration.
        
        How fast volume is changing compared to recent history.
        """
        if self._prev_volume <= 0:
            return 0.0
        
        if len(self.volume_buffer) < 5:
            return 0.0
        
        # Current velocity
        current_velocity = (current_volume - self._prev_volume) / self._prev_volume
        
        # Previous velocity (approximation)
        volumes = self.volume_buffer.to_array()[-5:]
        if len(volumes) >= 3:
            prev_velocity = (volumes[-2] - volumes[-3]) / volumes[-3] if volumes[-3] > 0 else 0
        else:
            prev_velocity = 0.0
        
        # Acceleration = change in velocity
        acceleration = current_velocity - prev_velocity
        
        return acceleration
    
    def _calculate_relative_volume(self, current_volume: float) -> float:
        """
        Calculate relative volume percentile.
        
        Where current volume sits relative to historical distribution.
        """
        if len(self.volume_buffer) < 10:
            return 50.0
        
        return self.volume_buffer.percentile(current_volume)
    
    def _calculate_volume_profile_skew(self) -> float:
        """
        Calculate volume profile skew.
        
        Positive skew = more volume recently (building interest)
        Negative skew = volume declining (waning interest)
        """
        if len(self.volume_buffer) < 10:
            return 0.0
        
        volumes = self.volume_buffer.to_array()
        
        # Compare recent half vs older half
        mid = len(volumes) // 2
        recent_avg = np.mean(volumes[mid:])
        older_avg = np.mean(volumes[:mid])
        
        if older_avg <= 0:
            return 0.0
        
        skew = (recent_avg - older_avg) / older_avg
        
        return np.clip(skew, -1.0, 1.0)
    
    def _calculate_range_expansion(self, current_range: float) -> float:
        """
        Calculate range expansion ratio.
        
        How current range compares to recent average.
        """
        if len(self.range_buffer) < 5:
            return 1.0
        
        avg_range = self.range_buffer.mean()
        if avg_range <= 0:
            return 1.0
        
        return current_range / avg_range
    
    def _calculate_range_vs_atr(self, current_range: float) -> float:
        """
        Calculate current range vs ATR (Average True Range).
        
        > 1 = above average volatility
        < 1 = below average volatility
        """
        if len(self._atr_history) < 7:
            return 1.0
        
        atr = np.mean(self._atr_history)
        if atr <= 0:
            return 1.0
        
        return current_range / atr
    
    def _calculate_volatility_compression(self) -> float:
        """
        Calculate volatility compression index.
        
        High = volatility is compressing (squeeze building)
        """
        if len(self.range_buffer) < 10:
            return 0.0
        
        ranges = self.range_buffer.to_array()[-10:]
        
        # Check if ranges are decreasing
        if len(ranges) >= 5:
            recent_avg = np.mean(ranges[-3:])
            older_avg = np.mean(ranges[:3])
            
            if older_avg > 0:
                compression = 1.0 - (recent_avg / older_avg)
                return max(0.0, compression)
        
        return 0.0
    
    def _calculate_volatility_expansion(self) -> float:
        """
        Calculate volatility expansion index.
        
        High = volatility is expanding (breakout potential)
        """
        if len(self.range_buffer) < 10:
            return 0.0
        
        ranges = self.range_buffer.to_array()[-10:]
        
        # Check if ranges are increasing
        if len(ranges) >= 5:
            recent_avg = np.mean(ranges[-3:])
            older_avg = np.mean(ranges[:3])
            
            if older_avg > 0:
                expansion = (recent_avg / older_avg) - 1.0
                return max(0.0, expansion)
        
        return 0.0
    
    def _calculate_realized_volatility(self) -> float:
        """
        Calculate realized volatility from price changes.
        
        Annualized standard deviation of returns.
        """
        if len(self.price_change_buffer) < 10:
            return 0.0
        
        changes = self.price_change_buffer.to_array()
        
        # Standard deviation of returns
        std = np.std(changes)
        
        # Annualize (assuming daily data, 365 days)
        annualized = std * np.sqrt(365)
        
        return annualized
    
    def _calculate_trade_count_percentile(self, current_count: int) -> float:
        """Calculate where current trade count sits in distribution."""
        if len(self.trade_count_buffer) < 10:
            return 50.0
        
        return self.trade_count_buffer.percentile(float(current_count))
    
    def _calculate_institutional_interest(self, volume: float, trade_count: int) -> float:
        """
        Calculate institutional interest index.
        
        High volume with low trade count = large orders = institutional
        """
        if trade_count <= 0 or volume <= 0:
            return 0.5
        
        avg_trade_size = volume / trade_count
        
        # Compare to historical average trade size
        if len(self.volume_buffer) >= 10 and len(self.trade_count_buffer) >= 10:
            hist_volume = np.mean(self.volume_buffer.to_array())
            hist_trades = np.mean(self.trade_count_buffer.to_array())
            
            if hist_trades > 0:
                hist_avg_size = hist_volume / hist_trades
                
                if hist_avg_size > 0:
                    # Higher than average trade size = more institutional
                    ratio = avg_trade_size / hist_avg_size
                    return min(1.0, ratio / 2)  # Normalize
        
        return 0.5
    
    def _calculate_market_strength(self, price_change_pct: float, volume: float) -> float:
        """
        Calculate market strength index.
        
        Combines price direction with volume confirmation.
        """
        if len(self.volume_buffer) < 5:
            return 0.5
        
        avg_volume = self.volume_buffer.mean()
        
        # Volume confirmation
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        # Direction
        direction = 1.0 if price_change_pct > 0 else -1.0 if price_change_pct < 0 else 0.0
        
        # Strength = direction * volume confirmation
        # Normalize to 0-1 range
        strength = 0.5 + (direction * min(1.0, volume_ratio - 1.0) * 0.5)
        
        return np.clip(strength, 0.0, 1.0)
