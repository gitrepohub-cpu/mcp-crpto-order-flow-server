"""
📚 Orderbook Feature Calculator
==============================

Calculates liquidity structure, dynamics, and heatmap features from orderbook stream.

Features:
- Depth imbalance (multi-level)
- Liquidity gradient & concentration
- Absorption detection
- Queue position drift
- Wall detection (pull/push)
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from ..base import InstitutionalFeatureCalculator, FeatureBuffer

logger = logging.getLogger(__name__)


class OrderbookFeatureCalculator(InstitutionalFeatureCalculator):
    """
    Real-time calculator for orderbook features.
    
    Input: Orderbook data (bids/asks lists from IsolatedDataCollector)
    Output: 20+ institutional features for liquidity analysis
    
    Features:
    - Multi-level depth imbalance
    - Liquidity structure (gradient, concentration, VWAP depth)
    - Orderbook dynamics (absorption, replenishment, queue drift)
    - Wall detection (pull/push manipulation)
    - Support/resistance strength
    """
    
    feature_type = "orderbook"
    feature_count = 20
    requires_history = 10
    
    def _init_buffers(self):
        """Initialize orderbook-specific buffers."""
        self.imbalance_5_buffer = self._create_buffer('imbalance_5')
        self.imbalance_10_buffer = self._create_buffer('imbalance_10')
        self.bid_depth_buffer = self._create_buffer('bid_depth')
        self.ask_depth_buffer = self._create_buffer('ask_depth')
        self.total_depth_buffer = self._create_buffer('total_depth')
        self.absorption_buffer = self._create_buffer('absorption')
        
        # Previous orderbook state for dynamics
        self._prev_bids: List[List] = []
        self._prev_asks: List[List] = []
        self._prev_bid_depth: float = 0.0
        self._prev_ask_depth: float = 0.0
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate orderbook features.
        
        Args:
            data: Dict with 'bids' and 'asks' (lists of [price, qty] pairs)
        
        Returns:
            Dict of orderbook features
        """
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        if not bids or not asks:
            return {}
        
        # Ensure format is [[price, qty], ...]
        if isinstance(bids[0], dict):
            bids = [[b['price'], b['quantity']] for b in bids]
            asks = [[a['price'], a['quantity']] for a in asks]
        
        # Calculate depth at different levels
        bid_depth_5 = sum(b[1] for b in bids[:5]) if len(bids) >= 5 else sum(b[1] for b in bids)
        ask_depth_5 = sum(a[1] for a in asks[:5]) if len(asks) >= 5 else sum(a[1] for a in asks)
        bid_depth_10 = sum(b[1] for b in bids[:10]) if len(bids) >= 10 else sum(b[1] for b in bids)
        ask_depth_10 = sum(a[1] for a in asks[:10]) if len(asks) >= 10 else sum(a[1] for a in asks)
        
        total_bid_depth = sum(b[1] for b in bids)
        total_ask_depth = sum(a[1] for a in asks)
        
        # Calculate imbalances
        imbalance_5 = self._calculate_imbalance(bid_depth_5, ask_depth_5)
        imbalance_10 = self._calculate_imbalance(bid_depth_10, ask_depth_10)
        cumulative_imbalance = self._calculate_imbalance(total_bid_depth, total_ask_depth)
        
        # Calculate liquidity structure
        liquidity_gradient = self._calculate_liquidity_gradient(bids, asks)
        concentration_idx = self._calculate_concentration_index(bids, asks)
        vwap_depth = self._calculate_vwap_depth(bids, asks)
        
        # Calculate dynamics (compared to previous state)
        absorption_ratio = self._calculate_absorption(bids, asks)
        replenishment_speed = self._calculate_replenishment()
        queue_drift = self._calculate_queue_drift(bids, asks)
        add_cancel_ratio = self._calculate_add_cancel_ratio(bids, asks)
        
        # Wall detection
        pull_wall = self._detect_pull_wall(bids, asks)
        push_wall = self._detect_push_wall(bids, asks)
        
        # Liquidity persistence
        persistence_score = self._calculate_persistence_score()
        migration_velocity = self._calculate_migration_velocity()
        
        # Support/resistance
        support_strength = self._calculate_support_strength(bids)
        resistance_strength = self._calculate_resistance_strength(asks)
        
        # Update buffers
        self.imbalance_5_buffer.append(imbalance_5)
        self.imbalance_10_buffer.append(imbalance_10)
        self.bid_depth_buffer.append(total_bid_depth)
        self.ask_depth_buffer.append(total_ask_depth)
        self.total_depth_buffer.append(total_bid_depth + total_ask_depth)
        self.absorption_buffer.append(absorption_ratio)
        
        # Store current state for next calculation
        self._prev_bids = bids
        self._prev_asks = asks
        self._prev_bid_depth = total_bid_depth
        self._prev_ask_depth = total_ask_depth
        
        return {
            # Depth Imbalance
            'depth_imbalance_5': imbalance_5,
            'depth_imbalance_10': imbalance_10,
            'cumulative_depth_imbalance': cumulative_imbalance,
            
            # Liquidity Structure
            'liquidity_gradient': liquidity_gradient,
            'liquidity_concentration_idx': concentration_idx,
            'vwap_depth': vwap_depth,
            'bid_depth_5': bid_depth_5,
            'ask_depth_5': ask_depth_5,
            'bid_depth_10': bid_depth_10,
            'ask_depth_10': ask_depth_10,
            
            # Dynamics
            'queue_position_drift': queue_drift,
            'add_cancel_ratio': add_cancel_ratio,
            'absorption_ratio': absorption_ratio,
            'replenishment_speed': replenishment_speed,
            
            # Heatmap/Wall Detection
            'liquidity_persistence_score': persistence_score,
            'liquidity_migration_velocity': migration_velocity,
            'pull_wall_detected': pull_wall,
            'push_wall_detected': push_wall,
            
            # Support/Resistance
            'support_strength': support_strength,
            'resistance_strength': resistance_strength,
        }
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            'depth_imbalance_5', 'depth_imbalance_10', 'cumulative_depth_imbalance',
            'liquidity_gradient', 'liquidity_concentration_idx', 'vwap_depth',
            'bid_depth_5', 'ask_depth_5', 'bid_depth_10', 'ask_depth_10',
            'queue_position_drift', 'add_cancel_ratio', 'absorption_ratio', 'replenishment_speed',
            'liquidity_persistence_score', 'liquidity_migration_velocity',
            'pull_wall_detected', 'push_wall_detected',
            'support_strength', 'resistance_strength',
        ]
    
    def _calculate_imbalance(self, bid_depth: float, ask_depth: float) -> float:
        """Calculate bid/ask imbalance ratio (-1 to 1)."""
        total = bid_depth + ask_depth
        if total == 0:
            return 0.0
        return (bid_depth - ask_depth) / total
    
    def _calculate_liquidity_gradient(self, bids: List, asks: List) -> float:
        """
        Calculate liquidity gradient (how fast depth increases with distance).
        
        Steep gradient = liquidity concentrated near best price
        Flat gradient = liquidity spread evenly
        """
        if len(bids) < 5 or len(asks) < 5:
            return 0.0
        
        # Get best prices
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2
        
        # Calculate cumulative depth at each level
        bid_cum = []
        ask_cum = []
        cum_bid = 0
        cum_ask = 0
        
        for i, b in enumerate(bids[:10]):
            cum_bid += b[1]
            distance_pct = abs(b[0] - mid) / mid * 100
            bid_cum.append((distance_pct, cum_bid))
        
        for i, a in enumerate(asks[:10]):
            cum_ask += a[1]
            distance_pct = abs(a[0] - mid) / mid * 100
            ask_cum.append((distance_pct, cum_ask))
        
        # Calculate gradient (slope of depth vs distance)
        if len(bid_cum) < 3 or len(ask_cum) < 3:
            return 0.0
        
        # Simple linear regression slope
        bid_gradient = self._calculate_slope([x[0] for x in bid_cum], [x[1] for x in bid_cum])
        ask_gradient = self._calculate_slope([x[0] for x in ask_cum], [x[1] for x in ask_cum])
        
        # Average gradient (normalized)
        total_depth = bid_cum[-1][1] + ask_cum[-1][1]
        if total_depth == 0:
            return 0.0
        
        return (bid_gradient + ask_gradient) / 2 / total_depth * 100
    
    def _calculate_slope(self, x: List[float], y: List[float]) -> float:
        """Simple linear regression slope."""
        n = len(x)
        if n < 2:
            return 0.0
        
        x_arr = np.array(x)
        y_arr = np.array(y)
        
        sum_x = np.sum(x_arr)
        sum_y = np.sum(y_arr)
        sum_xy = np.sum(x_arr * y_arr)
        sum_xx = np.sum(x_arr ** 2)
        
        denom = n * sum_xx - sum_x ** 2
        if denom == 0:
            return 0.0
        
        return (n * sum_xy - sum_x * sum_y) / denom
    
    def _calculate_concentration_index(self, bids: List, asks: List) -> float:
        """
        Calculate liquidity concentration (Herfindahl-like index).
        
        High index = liquidity concentrated in few levels
        Low index = liquidity spread across many levels
        """
        all_qty = [b[1] for b in bids[:10]] + [a[1] for a in asks[:10]]
        
        if not all_qty:
            return 0.0
        
        total = sum(all_qty)
        if total == 0:
            return 0.0
        
        # Herfindahl index: sum of squared market shares
        shares = [q / total for q in all_qty]
        hhi = sum(s ** 2 for s in shares)
        
        return hhi
    
    def _calculate_vwap_depth(self, bids: List, asks: List) -> float:
        """Calculate volume-weighted average depth price."""
        all_levels = bids[:10] + asks[:10]
        
        if not all_levels:
            return 0.0
        
        total_value = sum(l[0] * l[1] for l in all_levels)
        total_qty = sum(l[1] for l in all_levels)
        
        if total_qty == 0:
            return 0.0
        
        return total_value / total_qty
    
    def _calculate_absorption(self, bids: List, asks: List) -> float:
        """
        Calculate absorption ratio.
        
        Measures how well the orderbook absorbs incoming orders.
        High absorption = strong support/resistance
        """
        if not self._prev_bids or not self._prev_asks:
            return 0.0
        
        # Compare current vs previous depth
        current_bid_depth = sum(b[1] for b in bids[:5])
        current_ask_depth = sum(a[1] for a in asks[:5])
        
        if self._prev_bid_depth == 0 or self._prev_ask_depth == 0:
            return 0.0
        
        # Absorption = how much depth was maintained despite trading
        bid_absorption = current_bid_depth / self._prev_bid_depth if self._prev_bid_depth > 0 else 0
        ask_absorption = current_ask_depth / self._prev_ask_depth if self._prev_ask_depth > 0 else 0
        
        return (bid_absorption + ask_absorption) / 2
    
    def _calculate_replenishment(self) -> float:
        """
        Calculate how fast liquidity replenishes.
        
        Based on depth buffer recovery after drops.
        """
        if len(self.total_depth_buffer) < 10:
            return 0.0
        
        depths = list(self.total_depth_buffer.values)[-10:]
        
        # Count recovery events (depth drop followed by increase)
        recoveries = 0
        for i in range(2, len(depths)):
            if depths[i-1] < depths[i-2] and depths[i] > depths[i-1]:
                recoveries += 1
        
        return recoveries / (len(depths) - 2)
    
    def _calculate_queue_drift(self, bids: List, asks: List) -> float:
        """
        Calculate queue position drift.
        
        Detects if liquidity is systematically moving (front-running).
        Positive = bids advancing, Negative = asks advancing
        """
        if not self._prev_bids or not self._prev_asks:
            return 0.0
        
        if not bids or not asks:
            return 0.0
        
        # Compare best bid/ask changes
        current_best_bid = bids[0][0]
        current_best_ask = asks[0][0]
        prev_best_bid = self._prev_bids[0][0] if self._prev_bids else current_best_bid
        prev_best_ask = self._prev_asks[0][0] if self._prev_asks else current_best_ask
        
        mid = (current_best_bid + current_best_ask) / 2
        if mid == 0:
            return 0.0
        
        bid_drift = (current_best_bid - prev_best_bid) / mid * 10000
        ask_drift = (current_best_ask - prev_best_ask) / mid * 10000
        
        return bid_drift - ask_drift
    
    def _calculate_add_cancel_ratio(self, bids: List, asks: List) -> float:
        """
        Estimate add/cancel ratio (spoofing indicator).
        
        High ratio = orders being placed and staying
        Low ratio = orders being cancelled quickly (spoofing)
        """
        if not self._prev_bids or not self._prev_asks:
            return 1.0
        
        # Count levels that persisted
        current_bid_prices = set(b[0] for b in bids[:10])
        current_ask_prices = set(a[0] for a in asks[:10])
        prev_bid_prices = set(b[0] for b in self._prev_bids[:10])
        prev_ask_prices = set(a[0] for a in self._prev_asks[:10])
        
        # Persistence = levels that existed before and still exist
        bid_persist = len(current_bid_prices.intersection(prev_bid_prices))
        ask_persist = len(current_ask_prices.intersection(prev_ask_prices))
        
        total_prev = len(prev_bid_prices) + len(prev_ask_prices)
        if total_prev == 0:
            return 1.0
        
        return (bid_persist + ask_persist) / total_prev
    
    def _detect_pull_wall(self, bids: List, asks: List) -> bool:
        """
        Detect if a large order (wall) was pulled.
        
        Manipulation signal: wall disappears before price reaches it.
        """
        if not self._prev_bids or not self._prev_asks:
            return False
        
        # Look for large orders in previous state that disappeared
        threshold_multiplier = 3.0  # Wall = 3x average
        
        prev_bid_qty_avg = np.mean([b[1] for b in self._prev_bids[:10]]) if self._prev_bids else 0
        prev_ask_qty_avg = np.mean([a[1] for a in self._prev_asks[:10]]) if self._prev_asks else 0
        
        # Check if any wall was pulled
        for prev_bid in self._prev_bids[:5]:
            if prev_bid[1] > prev_bid_qty_avg * threshold_multiplier:
                # This was a wall - is it still there?
                current_at_price = [b for b in bids if abs(b[0] - prev_bid[0]) < 0.01]
                if not current_at_price or current_at_price[0][1] < prev_bid[1] * 0.5:
                    return True
        
        for prev_ask in self._prev_asks[:5]:
            if prev_ask[1] > prev_ask_qty_avg * threshold_multiplier:
                current_at_price = [a for a in asks if abs(a[0] - prev_ask[0]) < 0.01]
                if not current_at_price or current_at_price[0][1] < prev_ask[1] * 0.5:
                    return True
        
        return False
    
    def _detect_push_wall(self, bids: List, asks: List) -> bool:
        """
        Detect if a new large wall appeared.
        
        Can be intimidation or real support/resistance.
        """
        if not self._prev_bids or not self._prev_asks:
            return False
        
        threshold_multiplier = 3.0
        
        current_bid_qty_avg = np.mean([b[1] for b in bids[:10]]) if bids else 0
        current_ask_qty_avg = np.mean([a[1] for a in asks[:10]]) if asks else 0
        
        # Check for new walls
        for bid in bids[:5]:
            if bid[1] > current_bid_qty_avg * threshold_multiplier:
                # Was this here before?
                prev_at_price = [b for b in self._prev_bids if abs(b[0] - bid[0]) < 0.01]
                if not prev_at_price or prev_at_price[0][1] < bid[1] * 0.3:
                    return True
        
        for ask in asks[:5]:
            if ask[1] > current_ask_qty_avg * threshold_multiplier:
                prev_at_price = [a for a in self._prev_asks if abs(a[0] - ask[0]) < 0.01]
                if not prev_at_price or prev_at_price[0][1] < ask[1] * 0.3:
                    return True
        
        return False
    
    def _calculate_persistence_score(self) -> float:
        """Calculate liquidity persistence over time."""
        if len(self.total_depth_buffer) < 10:
            return 0.5
        
        depths = self.total_depth_buffer.to_array()[-10:]
        
        # Low variance = high persistence
        cv = np.std(depths) / np.mean(depths) if np.mean(depths) > 0 else 1.0
        
        return max(0.0, 1.0 - cv)
    
    def _calculate_migration_velocity(self) -> float:
        """Calculate how fast liquidity is migrating (changing levels)."""
        if len(self.imbalance_5_buffer) < 10:
            return 0.0
        
        return self.imbalance_5_buffer.velocity(periods=5)
    
    def _calculate_support_strength(self, bids: List) -> float:
        """Calculate bid-side support strength."""
        if len(bids) < 5:
            return 0.0
        
        # Depth concentration at top levels
        top_3_depth = sum(b[1] for b in bids[:3])
        total_depth = sum(b[1] for b in bids[:10])
        
        if total_depth == 0:
            return 0.0
        
        return top_3_depth / total_depth
    
    def _calculate_resistance_strength(self, asks: List) -> float:
        """Calculate ask-side resistance strength."""
        if len(asks) < 5:
            return 0.0
        
        top_3_depth = sum(a[1] for a in asks[:3])
        total_depth = sum(a[1] for a in asks[:10])
        
        if total_depth == 0:
            return 0.0
        
        return top_3_depth / total_depth
