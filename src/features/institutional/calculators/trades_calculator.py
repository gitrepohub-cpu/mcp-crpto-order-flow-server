"""
📊 Trades Feature Calculator
============================

Calculates trade flow features including CVD, aggressor detection, whale trades.

Features:
- Cumulative Volume Delta (CVD)
- Aggressor analysis
- Whale detection
- Iceberg detection
- Market impact
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from collections import deque

from ..base import InstitutionalFeatureCalculator, FeatureBuffer

logger = logging.getLogger(__name__)


class TradesFeatureCalculator(InstitutionalFeatureCalculator):
    """
    Real-time calculator for trade flow features.
    
    Input: Trade data (price, quantity, side, timestamp)
    Output: 21 institutional features for flow analysis
    
    Features:
    - CVD (cumulative volume delta) + derivatives
    - Aggressor analysis (buy/sell classification)
    - Whale detection (large trade detection)
    - Iceberg detection (hidden order inference)
    - Sweep detection (multiple level fills)
    - Market impact estimation
    """
    
    feature_type = "trades"
    feature_count = 21
    requires_history = 100  # Need more history for trade analysis
    
    def _init_buffers(self):
        """Initialize trade-specific buffers."""
        self.cvd_buffer = self._create_buffer('cvd')
        self.volume_buffer = self._create_buffer('volume')
        self.buy_volume_buffer = self._create_buffer('buy_volume')
        self.sell_volume_buffer = self._create_buffer('sell_volume')
        self.price_buffer = self._create_buffer('price')
        self.trade_size_buffer = self._create_buffer('trade_size')
        
        # Running CVD
        self._cumulative_cvd: float = 0.0
        
        # Trade batching (aggregate multiple trades)
        self._trade_batch: List[Dict] = []
        self._batch_window_ms: int = 100  # Aggregate trades within 100ms
        self._last_batch_time: Optional[datetime] = None
        
        # Whale threshold (dynamically adjusted)
        self._whale_threshold: float = 0.0
        
        # Previous trade for sweep detection
        self._prev_trades: deque = deque(maxlen=50)
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate trade features.
        
        Args:
            data: Dict with trade info (price, quantity, side, timestamp)
                  OR list of trades for batch processing
        
        Returns:
            Dict of trade features
        """
        # Handle batch or single trade
        if isinstance(data, list):
            trades = data
        elif 'trades' in data:
            trades = data['trades']
        else:
            trades = [data]
        
        if not trades:
            return {}
        
        # Process trades
        total_volume = 0.0
        buy_volume = 0.0
        sell_volume = 0.0
        buy_count = 0
        sell_count = 0
        prices = []
        sizes = []
        whale_trades = []
        
        for trade in trades:
            price = float(trade.get('price', 0))
            qty = float(trade.get('quantity', trade.get('qty', trade.get('amount', 0))))
            side = trade.get('side', '').lower()
            
            if price <= 0 or qty <= 0:
                continue
            
            total_volume += qty
            prices.append(price)
            sizes.append(qty)
            
            # Classify aggressor side
            is_buy = side in ('buy', 'b', 'bid')
            is_sell = side in ('sell', 's', 'ask')
            
            if not is_buy and not is_sell:
                # Try to infer from taker/maker
                is_buy_maker = trade.get('is_buyer_maker', trade.get('isBuyerMaker', None))
                if is_buy_maker is not None:
                    is_buy = not is_buy_maker
                    is_sell = is_buy_maker
            
            if is_buy:
                buy_volume += qty
                buy_count += 1
            elif is_sell:
                sell_volume += qty
                sell_count += 1
            
            # Store for analysis
            self._prev_trades.append({
                'price': price,
                'qty': qty,
                'side': 'buy' if is_buy else 'sell',
                'value': price * qty,
            })
        
        # Update whale threshold (adaptive)
        if sizes:
            self._update_whale_threshold(sizes)
            whale_trades = [s for s in sizes if s >= self._whale_threshold]
        
        # Calculate CVD delta
        cvd_delta = buy_volume - sell_volume
        self._cumulative_cvd += cvd_delta
        
        # Calculate features
        aggressive_delta = buy_volume - sell_volume
        aggressive_ratio = buy_volume / sell_volume if sell_volume > 0 else (2.0 if buy_volume > 0 else 1.0)
        
        # Update buffers
        self.cvd_buffer.append(self._cumulative_cvd)
        self.volume_buffer.append(total_volume)
        self.buy_volume_buffer.append(buy_volume)
        self.sell_volume_buffer.append(sell_volume)
        if prices:
            self.price_buffer.append(prices[-1])
        if sizes:
            self.trade_size_buffer.append(np.mean(sizes))
        
        # Calculate derived features
        cvd_slope = self.cvd_buffer.velocity(periods=10)
        cvd_zscore = self.cvd_buffer.zscore()
        cvd_divergence = self._calculate_cvd_divergence()
        
        # Flow intensity
        flow_intensity = self._calculate_flow_intensity()
        flow_toxicity = self._calculate_flow_toxicity()
        
        # Whale detection
        whale_detected = len(whale_trades) > 0
        whale_volume_ratio = sum(whale_trades) / total_volume if total_volume > 0 else 0.0
        
        # Iceberg detection
        iceberg_prob = self._detect_iceberg()
        
        # Sweep detection
        sweep_detected = self._detect_sweep(prices, sizes)
        
        # Market impact
        market_impact = self._calculate_market_impact()
        
        # Trade frequency analysis
        avg_trade_size = np.mean(sizes) if sizes else 0.0
        trade_size_stddev = np.std(sizes) if len(sizes) > 1 else 0.0
        
        # Buy/sell pressure
        buy_pressure = buy_count / len(trades) if trades else 0.5
        sell_pressure = sell_count / len(trades) if trades else 0.5
        
        # VWAP
        vwap = self._calculate_vwap(prices, sizes)
        
        return {
            # CVD Family
            'cvd': self._cumulative_cvd,
            'cvd_delta': cvd_delta,
            'cvd_slope': cvd_slope,
            'cvd_zscore': cvd_zscore,
            'cvd_price_divergence': cvd_divergence,
            
            # Aggressor Analysis
            'aggressive_delta': aggressive_delta,
            'aggressive_ratio': aggressive_ratio,
            'buy_pressure': buy_pressure,
            'sell_pressure': sell_pressure,
            
            # Volume Analysis
            'total_volume': total_volume,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'avg_trade_size': avg_trade_size,
            'trade_size_stddev': trade_size_stddev,
            
            # Flow Quality
            'flow_intensity': flow_intensity,
            'flow_toxicity': flow_toxicity,
            
            # Whale Detection
            'whale_trade_detected': whale_detected,
            'whale_volume_ratio': whale_volume_ratio,
            
            # Pattern Detection
            'iceberg_probability': iceberg_prob,
            'sweep_detected': sweep_detected,
            
            # Market Impact
            'market_impact_per_volume': market_impact,
            
            # VWAP
            'vwap': vwap,
        }
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            'cvd', 'cvd_delta', 'cvd_slope', 'cvd_zscore', 'cvd_price_divergence',
            'aggressive_delta', 'aggressive_ratio', 'buy_pressure', 'sell_pressure',
            'total_volume', 'buy_volume', 'sell_volume', 'avg_trade_size', 'trade_size_stddev',
            'flow_intensity', 'flow_toxicity',
            'whale_trade_detected', 'whale_volume_ratio',
            'iceberg_probability', 'sweep_detected',
            'market_impact_per_volume', 'vwap',
        ]
    
    def _update_whale_threshold(self, sizes: List[float]):
        """Dynamically update whale threshold based on recent trade sizes."""
        if not sizes:
            return
        
        # Store in buffer
        for s in sizes:
            self.trade_size_buffer.append(s)
        
        # Whale = top 5% of trades
        if len(self.trade_size_buffer) >= 20:
            all_sizes = self.trade_size_buffer.to_array()
            self._whale_threshold = np.percentile(all_sizes, 95)
        else:
            # Initial estimate: 5x median
            self._whale_threshold = np.median(sizes) * 5
    
    def _calculate_cvd_divergence(self) -> float:
        """
        Calculate CVD-price divergence.
        
        Divergence = CVD moving one way while price moves another
        Strong predictive signal.
        """
        if len(self.cvd_buffer) < 10 or len(self.price_buffer) < 10:
            return 0.0
        
        cvd_values = self.cvd_buffer.to_array()[-10:]
        price_values = self.price_buffer.to_array()[-10:]
        
        if len(cvd_values) < 3 or len(price_values) < 3:
            return 0.0
        
        # Calculate correlation
        cvd_norm = (cvd_values - np.mean(cvd_values))
        price_norm = (price_values - np.mean(price_values))
        
        std_cvd = np.std(cvd_norm)
        std_price = np.std(price_norm)
        
        if std_cvd == 0 or std_price == 0:
            return 0.0
        
        correlation = np.mean(cvd_norm * price_norm) / (std_cvd * std_price)
        
        # Divergence = negative correlation
        return -correlation
    
    def _calculate_flow_intensity(self) -> float:
        """
        Calculate flow intensity (how aggressive the trading is).
        
        High intensity = large trades, directional
        Low intensity = small trades, balanced
        """
        if len(self.volume_buffer) < 5:
            return 0.5
        
        recent_volume = self.volume_buffer.to_array()[-5:]
        recent_buy = self.buy_volume_buffer.to_array()[-5:]
        recent_sell = self.sell_volume_buffer.to_array()[-5:]
        
        # Volume intensity
        vol_intensity = np.mean(recent_volume) / (np.std(recent_volume) + 1e-9)
        
        # Directional intensity
        deltas = [b - s for b, s in zip(recent_buy, recent_sell)]
        dir_intensity = abs(np.mean(deltas)) / (np.std(deltas) + 1e-9)
        
        # Combined
        return min(1.0, (vol_intensity + dir_intensity) / 4)
    
    def _calculate_flow_toxicity(self) -> float:
        """
        Calculate flow toxicity (informed trading indicator).
        
        Toxic flow = sequential same-direction trades pushing price
        Based on VPIN methodology.
        """
        if len(self._prev_trades) < 20:
            return 0.0
        
        trades = list(self._prev_trades)[-20:]
        
        # Count buy/sell streaks
        streaks = []
        current_streak = 1
        current_side = trades[0]['side']
        
        for i in range(1, len(trades)):
            if trades[i]['side'] == current_side:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
                current_side = trades[i]['side']
        streaks.append(current_streak)
        
        # High average streak length = toxic flow
        avg_streak = np.mean(streaks)
        
        # Normalize to 0-1
        return min(1.0, avg_streak / 5)
    
    def _detect_iceberg(self) -> float:
        """
        Detect iceberg orders (large hidden orders).
        
        Pattern: Multiple same-size trades at same price level
        Returns probability 0-1
        """
        if len(self._prev_trades) < 10:
            return 0.0
        
        trades = list(self._prev_trades)[-20:]
        
        # Look for same-size patterns
        sizes = [t['qty'] for t in trades]
        prices = [t['price'] for t in trades]
        
        # Count repeated sizes at similar prices
        size_price_pairs = [(round(s, 8), round(p, 8)) for s, p in zip(sizes, prices)]
        from collections import Counter
        pair_counts = Counter(size_price_pairs)
        
        # If same size appears 3+ times at same price, likely iceberg
        max_repeats = max(pair_counts.values()) if pair_counts else 0
        
        return min(1.0, max_repeats / 5)
    
    def _detect_sweep(self, prices: List[float], sizes: List[float]) -> bool:
        """
        Detect sweep orders (aggressive orders through multiple levels).
        
        Pattern: Multiple trades at ascending/descending prices in short time
        """
        if len(prices) < 3:
            return False
        
        # Check if prices are monotonically moving
        diffs = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
        
        # All positive or all negative diffs = sweep
        all_up = all(d > 0 for d in diffs)
        all_down = all(d < 0 for d in diffs)
        
        return all_up or all_down
    
    def _calculate_market_impact(self) -> float:
        """
        Calculate market impact per unit volume.
        
        How much price moves per unit of volume traded.
        """
        if len(self.price_buffer) < 5 or len(self.volume_buffer) < 5:
            return 0.0
        
        prices = self.price_buffer.to_array()[-5:]
        volumes = self.volume_buffer.to_array()[-5:]
        
        price_change = abs(prices[-1] - prices[0])
        total_volume = sum(volumes)
        
        if total_volume == 0 or prices[0] == 0:
            return 0.0
        
        # Normalized by starting price
        return (price_change / prices[0]) / total_volume * 1e6
    
    def _calculate_vwap(self, prices: List[float], sizes: List[float]) -> float:
        """Calculate volume-weighted average price."""
        if not prices or not sizes:
            return 0.0
        
        total_value = sum(p * s for p, s in zip(prices, sizes))
        total_qty = sum(sizes)
        
        if total_qty == 0:
            return 0.0
        
        return total_value / total_qty
    
    def reset_cvd(self):
        """Reset CVD to zero (for session boundaries)."""
        self._cumulative_cvd = 0.0
        self.cvd_buffer = self._create_buffer('cvd')
