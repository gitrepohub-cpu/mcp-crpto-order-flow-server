"""
💥 Liquidations Feature Calculator
==================================

Calculates liquidation-based features including cascade detection, clusters, exhaustion.

Features:
- Liquidation counts and values
- Cluster detection (grouped liquidations)
- Cascade detection (accelerating liquidations)
- Exhaustion signals (market bottom/top indicators)
- Long/short imbalance
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from collections import deque

from ..base import InstitutionalFeatureCalculator, FeatureBuffer

logger = logging.getLogger(__name__)


class LiquidationsFeatureCalculator(InstitutionalFeatureCalculator):
    """
    Real-time calculator for liquidation features.
    
    Input: Liquidation data (price, quantity, side, timestamp)
    Output: 18 institutional features for liquidation analysis
    
    Features:
    - Liquidation counts and values (long/short/total)
    - Liquidation imbalance (directional bias)
    - Cluster detection (grouped liquidations in time/price)
    - Cascade detection (accelerating liquidation rate)
    - Exhaustion signals (potential reversal after flush)
    """
    
    feature_type = "liquidations"
    feature_count = 18
    requires_history = 50
    
    def _init_buffers(self):
        """Initialize liquidation-specific buffers."""
        self.long_liq_buffer = self._create_buffer('long_liq')
        self.short_liq_buffer = self._create_buffer('short_liq')
        self.total_value_buffer = self._create_buffer('total_value')
        self.imbalance_buffer = self._create_buffer('imbalance')
        self.rate_buffer = self._create_buffer('rate')  # Liquidations per second
        
        # Recent liquidations for cluster detection
        self._recent_liqs: deque = deque(maxlen=100)
        
        # Cascade detection state
        self._last_liq_time: Optional[datetime] = None
        self._cascade_start_time: Optional[datetime] = None
        self._cascade_count: int = 0
        
        # Cluster detection state
        self._current_cluster: List[Dict] = []
        self._cluster_timeout_ms: int = 500  # New cluster if gap > 500ms
        
        # Rolling window for exhaustion
        self._exhaustion_window: deque = deque(maxlen=20)
    
    def calculate(self, data: Any) -> Dict[str, Any]:
        """
        Calculate liquidation features.
        
        Args:
            data: Single liquidation dict or list of liquidations
                  Each with: price, quantity, side (long/short), timestamp
        
        Returns:
            Dict of liquidation features
        """
        # Handle batch or single liquidation
        if isinstance(data, list):
            liqs = data
        elif 'liquidations' in data:
            liqs = data['liquidations']
        else:
            liqs = [data]
        
        if not liqs:
            return self._get_empty_features()
        
        now = datetime.now(timezone.utc)
        
        # Process liquidations
        long_count = 0
        short_count = 0
        long_value = 0.0
        short_value = 0.0
        
        for liq in liqs:
            price = float(liq.get('price', 0))
            qty = float(liq.get('quantity', liq.get('qty', liq.get('amount', 0))))
            side = liq.get('side', '').lower()
            timestamp = liq.get('timestamp', now)
            
            if price <= 0 or qty <= 0:
                continue
            
            value = price * qty
            
            # Classify liquidation side
            is_long = side in ('long', 'buy', 'b')
            is_short = side in ('short', 'sell', 's')
            
            if is_long:
                long_count += 1
                long_value += value
            elif is_short:
                short_count += 1
                short_value += value
            
            # Store for analysis
            self._recent_liqs.append({
                'price': price,
                'qty': qty,
                'value': value,
                'side': 'long' if is_long else 'short',
                'timestamp': timestamp,
            })
            
            # Update cluster
            self._update_cluster(liq, timestamp)
        
        total_count = long_count + short_count
        total_value = long_value + short_value
        
        # Calculate imbalance
        if total_value > 0:
            imbalance = (long_value - short_value) / total_value
        else:
            imbalance = 0.0
        
        # Update buffers
        self.long_liq_buffer.append(long_value)
        self.short_liq_buffer.append(short_value)
        self.total_value_buffer.append(total_value)
        self.imbalance_buffer.append(imbalance)
        
        # Calculate rate (liquidations per second)
        liq_rate = self._calculate_liq_rate(total_count, now)
        self.rate_buffer.append(liq_rate)
        
        # Cluster detection
        cluster_detected, cluster_info = self._detect_cluster()
        
        # Cascade detection
        cascade_info = self._detect_cascade(liq_rate, now)
        
        # Exhaustion detection
        exhaustion = self._detect_exhaustion(total_value, imbalance)
        
        # Absorption after liquidation
        absorption = self._calculate_absorption()
        
        # Update last liquidation time
        self._last_liq_time = now
        
        return {
            # Counts
            'long_liquidation_count': long_count,
            'short_liquidation_count': short_count,
            'total_liquidation_count': total_count,
            
            # Values
            'long_liquidation_value': long_value,
            'short_liquidation_value': short_value,
            'total_liquidation_value': total_value,
            'liquidation_imbalance': imbalance,
            
            # Rate
            'liquidation_rate': liq_rate,
            'liquidation_rate_zscore': self.rate_buffer.zscore(),
            
            # Clustering
            'liquidation_cluster_detected': cluster_detected,
            'cluster_size': cluster_info.get('size', 0),
            'cluster_value': cluster_info.get('value', 0.0),
            'cluster_duration_seconds': cluster_info.get('duration', 0.0),
            
            # Cascade
            'cascade_acceleration': cascade_info.get('acceleration', 0.0),
            'cascade_probability': cascade_info.get('probability', 0.0),
            'cascade_severity': cascade_info.get('severity', 'none'),
            
            # Exhaustion
            'exhaustion_signal': exhaustion.get('signal', False),
            'absorption_after_liq': absorption,
        }
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return [
            'long_liquidation_count', 'short_liquidation_count', 'total_liquidation_count',
            'long_liquidation_value', 'short_liquidation_value', 'total_liquidation_value',
            'liquidation_imbalance',
            'liquidation_rate', 'liquidation_rate_zscore',
            'liquidation_cluster_detected', 'cluster_size', 'cluster_value', 'cluster_duration_seconds',
            'cascade_acceleration', 'cascade_probability', 'cascade_severity',
            'exhaustion_signal', 'absorption_after_liq',
        ]
    
    def _get_empty_features(self) -> Dict[str, Any]:
        """Return empty features dict when no liquidations."""
        return {name: 0 if 'count' in name or 'size' in name else 
                      0.0 if not name.endswith('_detected') and not name.endswith('_signal') else
                      False if name.endswith('_detected') or name.endswith('_signal') else
                      'none' for name in self.get_feature_names()}
    
    def _calculate_liq_rate(self, count: int, now: datetime) -> float:
        """Calculate liquidation rate (per second)."""
        if self._last_liq_time is None:
            return float(count)
        
        elapsed = (now - self._last_liq_time).total_seconds()
        if elapsed <= 0:
            elapsed = 0.001  # Avoid division by zero
        
        return count / elapsed
    
    def _update_cluster(self, liq: Dict, timestamp: datetime):
        """Update cluster detection state."""
        if not self._current_cluster:
            self._current_cluster = [{'liq': liq, 'timestamp': timestamp}]
            return
        
        last_time = self._current_cluster[-1]['timestamp']
        
        # Check if this liquidation belongs to current cluster
        if isinstance(timestamp, datetime) and isinstance(last_time, datetime):
            gap_ms = (timestamp - last_time).total_seconds() * 1000
        else:
            gap_ms = 0
        
        if gap_ms <= self._cluster_timeout_ms:
            self._current_cluster.append({'liq': liq, 'timestamp': timestamp})
        else:
            # Start new cluster
            self._current_cluster = [{'liq': liq, 'timestamp': timestamp}]
    
    def _detect_cluster(self) -> tuple:
        """
        Detect liquidation cluster.
        
        Cluster = multiple liquidations in short time period.
        """
        if len(self._current_cluster) < 3:
            return False, {}
        
        # Calculate cluster metrics
        cluster_size = len(self._current_cluster)
        cluster_value = sum(
            float(item['liq'].get('price', 0)) * float(item['liq'].get('quantity', item['liq'].get('qty', 0)))
            for item in self._current_cluster
        )
        
        if len(self._current_cluster) >= 2:
            first_time = self._current_cluster[0]['timestamp']
            last_time = self._current_cluster[-1]['timestamp']
            if isinstance(first_time, datetime) and isinstance(last_time, datetime):
                duration = (last_time - first_time).total_seconds()
            else:
                duration = 0.0
        else:
            duration = 0.0
        
        detected = cluster_size >= 3
        
        return detected, {
            'size': cluster_size,
            'value': cluster_value,
            'duration': duration,
        }
    
    def _detect_cascade(self, current_rate: float, now: datetime) -> Dict[str, Any]:
        """
        Detect liquidation cascade.
        
        Cascade = accelerating liquidation rate over time.
        """
        if len(self.rate_buffer) < 5:
            return {'acceleration': 0.0, 'probability': 0.0, 'severity': 'none'}
        
        rates = self.rate_buffer.to_array()[-10:]
        
        # Calculate acceleration (second derivative)
        if len(rates) >= 3:
            velocities = np.diff(rates)
            acceleration = velocities[-1] - velocities[-2] if len(velocities) >= 2 else 0.0
        else:
            acceleration = 0.0
        
        # Calculate cascade probability based on:
        # 1. Positive acceleration
        # 2. Rate above average
        # 3. Recent rate trend
        
        avg_rate = np.mean(rates)
        current_vs_avg = current_rate / avg_rate if avg_rate > 0 else 1.0
        
        # Trend: are rates increasing?
        if len(rates) >= 3:
            trend = (rates[-1] - rates[-3]) / 3
        else:
            trend = 0.0
        
        # Combine into probability
        prob = 0.0
        if acceleration > 0:
            prob += 0.3
        if current_vs_avg > 1.5:
            prob += 0.3
        if trend > 0:
            prob += 0.2
        if current_vs_avg > 2.0:
            prob += 0.2
        
        # Determine severity
        if prob >= 0.7:
            severity = 'severe'
        elif prob >= 0.4:
            severity = 'moderate'
        elif prob > 0:
            severity = 'mild'
        else:
            severity = 'none'
        
        return {
            'acceleration': acceleration,
            'probability': min(1.0, prob),
            'severity': severity,
        }
    
    def _detect_exhaustion(self, total_value: float, imbalance: float) -> Dict[str, Any]:
        """
        Detect exhaustion signal.
        
        Exhaustion = large liquidation volume followed by imbalance shift.
        Often indicates potential reversal point.
        """
        self._exhaustion_window.append({
            'value': total_value,
            'imbalance': imbalance,
        })
        
        if len(self._exhaustion_window) < 5:
            return {'signal': False}
        
        recent = list(self._exhaustion_window)[-5:]
        
        # Check for spike in liquidation value
        values = [r['value'] for r in recent]
        avg_value = np.mean(values[:-1]) if len(values) > 1 else 0
        current_value = values[-1]
        
        value_spike = current_value > avg_value * 2 if avg_value > 0 else False
        
        # Check for imbalance extreme
        imbalances = [r['imbalance'] for r in recent]
        imbalance_extreme = abs(imbalances[-1]) > 0.7
        
        # Exhaustion signal when both conditions met
        signal = value_spike and imbalance_extreme
        
        return {'signal': signal}
    
    def _calculate_absorption(self) -> float:
        """
        Calculate absorption after liquidation.
        
        High absorption = market absorbed liquidation well.
        """
        if len(self.total_value_buffer) < 5:
            return 0.5
        
        values = self.total_value_buffer.to_array()[-5:]
        
        # If recent values are decreasing, good absorption
        if len(values) >= 3:
            trend = values[-1] - values[-3]
            if values[-3] > 0:
                absorption = 1.0 - (values[-1] / values[-3])
            else:
                absorption = 0.5
        else:
            absorption = 0.5
        
        return max(0.0, min(1.0, absorption))
