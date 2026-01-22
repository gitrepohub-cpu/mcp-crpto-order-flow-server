"""
🧮 Institutional Feature Calculator Base Class
==============================================

Base class for real-time streaming feature calculators.

Designed for:
- Tick-by-tick feature calculation
- Rolling window computations
- Efficient incremental updates
- Integration with IsolatedDataCollector callbacks
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, Deque
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class FeatureBuffer:
    """
    Rolling buffer for streaming feature calculation.
    
    Maintains a fixed-size window of recent values for
    efficient rolling computations (mean, std, zscore, etc.)
    """
    name: str = "buffer"
    window_size: int = 100
    
    def __post_init__(self):
        # Initialize deques with correct maxlen
        self.maxlen = self.window_size
        self.values: Deque = deque(maxlen=self.window_size)
        self.timestamps: Deque = deque(maxlen=self.window_size)
    
    def append(self, value: float, timestamp: Optional[datetime] = None):
        """Add a value to the buffer."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        self.values.append(value)
        self.timestamps.append(timestamp)
    
    def __len__(self) -> int:
        return len(self.values)
    
    def is_full(self) -> bool:
        """Check if buffer has reached max capacity."""
        return len(self.values) >= self.maxlen
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.values)
    
    def mean(self) -> float:
        """Rolling mean."""
        if not self.values:
            return 0.0
        return np.mean(self.values)
    
    def std(self) -> float:
        """Rolling standard deviation."""
        if len(self.values) < 2:
            return 0.0
        return np.std(self.values)
    
    def zscore(self, value: float = None) -> float:
        """Calculate z-score for a value. Uses last value if none provided."""
        if value is None:
            if not self.values:
                return 0.0
            value = self.values[-1]
        std = self.std()
        if std == 0:
            return 0.0
        return (value - self.mean()) / std
    
    def velocity(self, periods: int = 5) -> float:
        """Calculate rate of change over last N periods."""
        if len(self.values) < periods:
            return 0.0
        arr = self.to_array()
        return (arr[-1] - arr[-periods]) / periods
    
    def ewma(self, span: int = 20) -> float:
        """Exponential weighted moving average."""
        if not self.values:
            return 0.0
        alpha = 2 / (span + 1)
        result = self.values[0]
        for val in list(self.values)[1:]:
            result = alpha * val + (1 - alpha) * result
        return result
    
    def last(self, n: int = 1) -> List[float]:
        """Get last N values."""
        return list(self.values)[-n:]
    
    def percentile(self, value: float) -> float:
        """Calculate percentile of value within buffer."""
        if not self.values:
            return 50.0
        arr = self.to_array()
        return (np.sum(arr < value) / len(arr)) * 100


class InstitutionalFeatureCalculator(ABC):
    """
    Abstract base class for real-time streaming feature calculators.
    
    Designed for:
    - Processing tick-by-tick data from IsolatedDataCollector
    - Maintaining rolling windows for statistical features
    - Efficient incremental computation
    
    Subclass this for each stream type (prices, orderbook, trades, etc.)
    """
    
    # Override in subclasses
    feature_type: str = "base"
    feature_count: int = 0
    requires_history: int = 20  # Minimum data points before features are valid
    
    def __init__(
        self,
        symbol: str,
        exchange: str,
        market_type: str = "futures",
        window_size: int = 100,
        storage = None
    ):
        """
        Initialize calculator.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            exchange: Exchange name (e.g., "binance")
            market_type: "futures" or "spot"
            window_size: Rolling window size for buffers
            storage: InstitutionalFeatureStorage instance (optional)
        """
        self.symbol = symbol.lower()
        self.exchange = exchange.lower()
        self.market_type = market_type.lower()
        self.window_size = window_size
        self.storage = storage
        
        # Feature buffers (override in subclass to add specific buffers)
        self._buffers: Dict[str, FeatureBuffer] = {}
        
        # Latest calculated features cache
        self._latest_features: Dict[str, Any] = {}
        
        # Processing stats
        self.stats = {
            'updates_processed': 0,
            'features_calculated': 0,
            'errors': 0,
            'last_update': None,
        }
        
        # Initialize buffers
        self._init_buffers()
    
    def _init_buffers(self):
        """Initialize feature buffers. Override in subclass."""
        pass
    
    def _create_buffer(self, name: str, window_size: Optional[int] = None) -> FeatureBuffer:
        """Create and register a new buffer."""
        window_size = window_size or self.window_size
        buffer = FeatureBuffer(name=name, window_size=window_size)
        self._buffers[name] = buffer
        return buffer
    
    @abstractmethod
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate features from incoming data.
        
        Args:
            data: Raw data dict from stream
        
        Returns:
            Dict of feature_name: value
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature names this calculator produces."""
        pass
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing pipeline.
        
        1. Calculate features
        2. Cache latest features
        3. Store to database (if storage configured)
        4. Return features
        
        Args:
            data: Raw data from IsolatedDataCollector callback
        
        Returns:
            Calculated features dict
        """
        try:
            self.stats['updates_processed'] += 1
            self.stats['last_update'] = datetime.now(timezone.utc)
            
            # Calculate features
            features = self.calculate(data)
            
            if features:
                self._latest_features = features
                self.stats['features_calculated'] += 1
                
                # Store to database if configured
                if self.storage:
                    await self.storage.store_features(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        market_type=self.market_type,
                        feature_type=self.feature_type,
                        features=features
                    )
            
            return features
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Feature calculation error [{self.feature_type}]: {e}")
            return {}
    
    def get_latest_features(self) -> Dict[str, Any]:
        """Get most recently calculated features."""
        return self._latest_features.copy()
    
    def has_enough_data(self) -> bool:
        """Check if we have enough historical data for valid features."""
        if not self._buffers:
            return True
        return all(len(b) >= self.requires_history for b in self._buffers.values())
    
    def get_buffer_stats(self) -> Dict[str, int]:
        """Get buffer fill levels."""
        return {name: len(buf) for name, buf in self._buffers.items()}
    
    # =========================================================================
    # UTILITY METHODS FOR FEATURE CALCULATION
    # =========================================================================
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division that handles zero denominator."""
        if denominator == 0 or denominator is None:
            return default
        return numerator / denominator
    
    @staticmethod
    def clip(value: float, min_val: float, max_val: float) -> float:
        """Clip value to range."""
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def normalize(value: float, min_val: float, max_val: float) -> float:
        """Normalize value to 0-1 range."""
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
    
    def calculate_hurst_exponent(self, values: np.ndarray, max_lag: int = 20) -> float:
        """
        Calculate Hurst exponent for trend persistence.
        
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        if len(values) < max_lag * 2:
            return 0.5
        
        try:
            lags = range(2, min(max_lag, len(values) // 2))
            tau = []
            
            for lag in lags:
                # R/S analysis
                diffs = values[lag:] - values[:-lag]
                std = np.std(diffs)
                if std > 0:
                    tau.append(np.sqrt(np.abs(np.sum(diffs ** 2)) / len(diffs)))
            
            if len(tau) < 2:
                return 0.5
            
            # Linear regression in log-log space
            log_lags = np.log(list(lags)[:len(tau)])
            log_tau = np.log(tau)
            
            # Simple linear regression
            n = len(log_lags)
            sum_x = np.sum(log_lags)
            sum_y = np.sum(log_tau)
            sum_xy = np.sum(log_lags * log_tau)
            sum_xx = np.sum(log_lags ** 2)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
            
            return self.clip(slope, 0.0, 1.0)
            
        except Exception:
            return 0.5
    
    def calculate_entropy(self, values: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Shannon entropy of value distribution.
        
        Higher entropy = more random/unpredictable
        Lower entropy = more structured/predictable
        """
        if len(values) < bins:
            return 0.0
        
        try:
            hist, _ = np.histogram(values, bins=bins, density=True)
            hist = hist[hist > 0]  # Remove zeros
            return -np.sum(hist * np.log2(hist + 1e-10))
        except Exception:
            return 0.0


class FeatureCalculatorRegistry:
    """
    Registry for institutional feature calculators.
    
    Manages calculator instances per symbol/exchange pair.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._calculators: Dict[str, InstitutionalFeatureCalculator] = {}
        self._calculator_classes: Dict[str, type] = {}
        self._initialized = True
    
    def register_class(self, feature_type: str, calculator_class: type):
        """Register a calculator class for a feature type."""
        self._calculator_classes[feature_type] = calculator_class
        logger.info(f"✓ Registered calculator: {feature_type} -> {calculator_class.__name__}")
    
    def get_class(self, feature_type: str) -> Optional[type]:
        """Get registered calculator class for a feature type."""
        return self._calculator_classes.get(feature_type)
    
    def get_or_create(
        self,
        feature_type: str,
        symbol: str,
        exchange: str,
        market_type: str,
        **kwargs
    ) -> Optional[InstitutionalFeatureCalculator]:
        """
        Get existing calculator instance or create new one.
        
        Args:
            feature_type: "prices", "orderbook", "trades", etc.
            symbol: Trading pair
            exchange: Exchange name
            market_type: "futures" or "spot"
        
        Returns:
            Calculator instance or None if class not registered
        """
        key = f"{symbol.lower()}_{exchange.lower()}_{market_type.lower()}_{feature_type}"
        
        if key in self._calculators:
            return self._calculators[key]
        
        if feature_type not in self._calculator_classes:
            logger.warning(f"No calculator class registered for: {feature_type}")
            return None
        
        calc_class = self._calculator_classes[feature_type]
        instance = calc_class(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            **kwargs
        )
        
        self._calculators[key] = instance
        logger.debug(f"Created calculator instance: {key}")
        
        return instance
    
    def get_all_calculators_for_symbol(
        self,
        symbol: str,
        exchange: str,
        market_type: str
    ) -> Dict[str, InstitutionalFeatureCalculator]:
        """Get all calculator instances for a symbol."""
        prefix = f"{symbol.lower()}_{exchange.lower()}_{market_type.lower()}_"
        return {
            k.replace(prefix, ''): v
            for k, v in self._calculators.items()
            if k.startswith(prefix)
        }
    
    def get_all_latest_features(
        self,
        symbol: str,
        exchange: str,
        market_type: str
    ) -> Dict[str, Dict[str, Any]]:
        """Get latest features from all calculators for a symbol."""
        calculators = self.get_all_calculators_for_symbol(symbol, exchange, market_type)
        return {
            feature_type: calc.get_latest_features()
            for feature_type, calc in calculators.items()
        }


# Global registry instance
institutional_feature_registry = FeatureCalculatorRegistry()


def get_institutional_registry() -> FeatureCalculatorRegistry:
    """Get the global institutional feature calculator registry."""
    return institutional_feature_registry
