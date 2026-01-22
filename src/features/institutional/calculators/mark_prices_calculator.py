"""
📊 Mark Price Feature Calculator
================================

Calculates mark price features including basis, index divergence, dislocation.

Features:
- Mark-spot basis (contango/backwardation)
- Index price divergence
- Funding pressure inference
- Dislocation detection
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from ..base import InstitutionalFeatureCalculator, FeatureBuffer

logger = logging.getLogger(__name__)


class MarkPricesFeatureCalculator(InstitutionalFeatureCalculator):
    """
    Real-time calculator for mark price features.
    
    Input: Mark price data (mark_price, index_price, last_price)
    Output: 14 institutional features for basis/dislocation analysis
    
    Features:
    - Mark-spot basis (perpetual premium/discount)
    - Index divergence (mark vs index)
    - Basis z-score and mean reversion
    - Funding pressure inference
    - Dislocation detection (unusual basis movements)
    """
    
    feature_type = "mark_prices"
    feature_count = 14
    requires_history = 30
    
    def _init_buffers(self):
        """Initialize mark price-specific buffers."""
        self.mark_buffer = self._create_buffer('mark')
        self.index_buffer = self._create_buffer('index')
        self.basis_buffer = self._create_buffer('basis')
        self.basis_pct_buffer = self._create_buffer('basis_pct')
        self.mark_velocity_buffer = self._create_buffer('mark_velocity')
        
        # Previous values for velocity
        self._prev_mark: float = 0.0
        self._prev_index: float = 0.0
        self._prev_basis: float = 0.0
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate mark price features.
        
        Args:
            data: Dict with mark_price, index_price, optionally last_price/spot_price
        
        Returns:
            Dict of mark price features
        """
        # Extract prices
        mark_price = float(data.get('mark_price', data.get('markPrice', 0)))
        index_price = float(data.get('index_price', data.get('indexPrice', 0)))
        last_price = float(data.get('last_price', data.get('lastPrice', data.get('spot_price', mark_price))))
        
        if mark_price <= 0:
            return {}
        
        # Use index price if available, otherwise estimate from mark
        if index_price <= 0:
            index_price = mark_price  # Fallback
        
        # Calculate basis (mark - spot/index)
        basis = mark_price - index_price
        basis_pct = (basis / index_price * 100) if index_price > 0 else 0.0
        
        # Mark-mid divergence (if last_price is mid price)
        mark_vs_mid = ((mark_price - last_price) / last_price * 100) if last_price > 0 else 0.0
        
        # Update buffers
        self.mark_buffer.append(mark_price)
        self.index_buffer.append(index_price)
        self.basis_buffer.append(basis)
        self.basis_pct_buffer.append(basis_pct)
        
        # Calculate basis z-score
        basis_zscore = self.basis_pct_buffer.zscore()
        
        # Basis percentile
        basis_percentile = self.basis_pct_buffer.percentile(basis_pct)
        
        # Mark price velocity
        mark_velocity = 0.0
        if self._prev_mark > 0:
            mark_velocity = (mark_price - self._prev_mark) / self._prev_mark * 10000  # In bps
        self.mark_velocity_buffer.append(mark_velocity)
        
        # Index divergence (absolute difference from basis norm)
        index_divergence = self._calculate_index_divergence()
        index_divergence_risk = self._calculate_divergence_risk(index_divergence)
        
        # Mean reversion speed
        mean_reversion_speed = self._calculate_mean_reversion_speed()
        
        # Funding pressure inference (basis indicates funding direction)
        funding_pressure = self._infer_funding_pressure(basis_pct)
        
        # Dislocation detection
        dislocation = self._detect_dislocation(basis_zscore)
        
        # Contango/backwardation regime
        regime = 'contango' if basis_pct > 0.01 else ('backwardation' if basis_pct < -0.01 else 'neutral')
        
        # Annualized basis (funding proxy)
        # Assuming 8-hour funding, annualized = basis_pct * 3 * 365
        annualized_basis = basis_pct * 3 * 365
        
        # Update previous values
        self._prev_mark = mark_price
        self._prev_index = index_price
        self._prev_basis = basis
        
        return {
            # Core Basis
            'mark_spot_basis': basis,
            'mark_spot_basis_pct': basis_pct,
            'basis_zscore': basis_zscore,
            'basis_percentile': basis_percentile,
            'annualized_basis': annualized_basis,
            
            # Regime
            'basis_regime': regime,
            
            # Index Divergence
            'index_divergence': index_divergence,
            'index_divergence_risk': index_divergence_risk,
            
            # Dynamics
            'mark_price_velocity': mark_velocity,
            'mark_price_vs_mid': mark_vs_mid,
            'basis_mean_reversion_speed': mean_reversion_speed,
            
            # Funding Inference
            'funding_pressure': funding_pressure,
            
            # Dislocation
            'dislocation_detected': dislocation,
            
            # Raw Values
            'mark_price': mark_price,
            'index_price': index_price,
        }
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            'mark_spot_basis', 'mark_spot_basis_pct', 'basis_zscore', 'basis_percentile',
            'annualized_basis', 'basis_regime',
            'index_divergence', 'index_divergence_risk',
            'mark_price_velocity', 'mark_price_vs_mid', 'basis_mean_reversion_speed',
            'funding_pressure', 'dislocation_detected',
            'mark_price', 'index_price',
        ]
    
    def _calculate_index_divergence(self) -> float:
        """
        Calculate index divergence.
        
        How far current basis is from recent average.
        """
        if len(self.basis_pct_buffer) < 10:
            return 0.0
        
        current = self.basis_pct_buffer.values[-1]
        avg = self.basis_pct_buffer.mean()
        
        return current - avg
    
    def _calculate_divergence_risk(self, divergence: float) -> float:
        """
        Calculate risk level from divergence.
        
        High divergence = potential for snap-back.
        """
        if len(self.basis_pct_buffer) < 10:
            return 0.0
        
        std = self.basis_pct_buffer.std()
        if std == 0:
            return 0.0
        
        # Risk = how many std devs from mean
        risk = abs(divergence) / std
        
        return min(1.0, risk / 3)  # Normalize to 0-1
    
    def _calculate_mean_reversion_speed(self) -> float:
        """
        Calculate mean reversion speed of basis.
        
        Fast reversion = basis quickly returns to mean.
        Slow reversion = basis trends.
        """
        if len(self.basis_pct_buffer) < 20:
            return 0.5
        
        basis_values = self.basis_pct_buffer.to_array()[-20:]
        mean = np.mean(basis_values)
        
        # Calculate autocorrelation at lag 1
        # High autocorr = slow reversion, low = fast reversion
        if len(basis_values) < 3:
            return 0.5
        
        deviations = basis_values - mean
        autocorr = np.corrcoef(deviations[:-1], deviations[1:])[0, 1]
        
        if np.isnan(autocorr):
            return 0.5
        
        # Convert to speed: high autocorr = slow, low = fast
        speed = 1.0 - abs(autocorr)
        
        return max(0.0, min(1.0, speed))
    
    def _infer_funding_pressure(self, basis_pct: float) -> float:
        """
        Infer funding pressure from basis.
        
        Positive basis = longs pay shorts (positive funding)
        Negative basis = shorts pay longs (negative funding)
        
        Returns -1 to 1 scale.
        """
        # Clip and normalize
        # Typical basis range: -0.1% to 0.1%
        return np.clip(basis_pct * 10, -1.0, 1.0)
    
    def _detect_dislocation(self, basis_zscore: float) -> bool:
        """
        Detect basis dislocation.
        
        Dislocation = unusually large basis, potential liquidation or manipulation.
        """
        return abs(basis_zscore) > 2.5


class BasisArbitrageCalculator(InstitutionalFeatureCalculator):
    """
    Calculator for cross-exchange basis arbitrage signals.
    
    Monitors basis across exchanges for arbitrage opportunities.
    """
    
    feature_type = "basis_arbitrage"
    feature_count = 6
    requires_history = 10
    
    def _init_buffers(self):
        """Initialize arbitrage-specific buffers."""
        self.spread_buffer = self._create_buffer('spread')
        self._exchange_basis: Dict[str, float] = {}
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basis arbitrage features."""
        exchange_basis = data.get('exchange_basis', {})
        
        if len(exchange_basis) < 2:
            return {
                'basis_spread': 0.0,
                'arbitrage_opportunity': False,
                'long_basis_exchange': '',
                'short_basis_exchange': '',
                'expected_profit_bps': 0.0,
                'convergence_probability': 0.0,
            }
        
        # Update exchange basis
        self._exchange_basis.update(exchange_basis)
        
        # Find best arbitrage pair
        basis_list = [(ex, b) for ex, b in self._exchange_basis.items()]
        basis_sorted = sorted(basis_list, key=lambda x: x[1])
        
        lowest = basis_sorted[0]  # Short basis here (buy perp)
        highest = basis_sorted[-1]  # Long basis here (sell perp)
        
        spread = highest[1] - lowest[1]
        self.spread_buffer.append(spread)
        
        # Arbitrage opportunity threshold
        min_spread = 0.0005  # 0.05% minimum spread
        opportunity = spread > min_spread
        
        # Expected profit (annualized)
        expected_profit_bps = spread * 3 * 365 * 10000  # 3 funding periods per day
        
        # Convergence probability
        convergence = self._calculate_convergence_probability()
        
        return {
            'basis_spread': spread,
            'arbitrage_opportunity': opportunity,
            'long_basis_exchange': highest[0],
            'short_basis_exchange': lowest[0],
            'expected_profit_bps': expected_profit_bps,
            'convergence_probability': convergence,
        }
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            'basis_spread', 'arbitrage_opportunity',
            'long_basis_exchange', 'short_basis_exchange',
            'expected_profit_bps', 'convergence_probability',
        ]
    
    def _calculate_convergence_probability(self) -> float:
        """Calculate probability that basis spread will converge."""
        if len(self.spread_buffer) < 5:
            return 0.5
        
        spreads = self.spread_buffer.to_array()[-10:]
        
        # If spread is decreasing, higher convergence probability
        if len(spreads) >= 3:
            trend = spreads[-1] - spreads[-3]
            if trend < 0:
                return min(0.9, 0.5 + abs(trend) * 100)
            else:
                return max(0.2, 0.5 - trend * 100)
        
        return 0.5
