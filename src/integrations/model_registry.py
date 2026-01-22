"""
Model Registry Enhancement

Comprehensive model management system for Darts forecasting models.
Provides centralized registry, versioning, performance tracking, and intelligent recommendations.

Features:
=========
1. Centralized Model Registry with rich metadata
2. Model versioning and configuration management
3. Performance tracking per symbol/exchange
4. Intelligent model recommendation based on historical performance
5. Model persistence (save/load trained models)
6. Model lifecycle management (train, validate, deploy, retire)

Usage:
======
    from src.integrations.model_registry import ModelRegistry
    
    # Get singleton registry
    registry = ModelRegistry.get_instance()
    
    # Register a trained model
    registry.register_trained_model(
        model_name='XGBoost',
        symbol='BTCUSDT',
        exchange='binance',
        wrapper=trained_wrapper,
        metrics={'mape': 1.5, 'direction_accuracy': 75.0}
    )
    
    # Get best model for a symbol
    best_model = registry.recommend_model('BTCUSDT', 'binance')
    
    # Load a persisted model
    wrapper = registry.load_model('XGBoost', 'BTCUSDT', 'binance')
"""

import os
import json
import pickle
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from pathlib import Path
from threading import Lock
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class ModelStatus(Enum):
    """Model lifecycle status."""
    REGISTERED = "registered"       # Just registered, not trained
    TRAINING = "training"           # Currently training
    TRAINED = "trained"             # Training complete
    VALIDATED = "validated"         # Passed validation
    DEPLOYED = "deployed"           # Active for production use
    DEPRECATED = "deprecated"       # Older version, still usable
    RETIRED = "retired"             # No longer usable


class ModelTier(Enum):
    """Model performance tier for quick selection."""
    S_TIER = "S"    # Top 5% - Best performers
    A_TIER = "A"    # Top 15% - Excellent
    B_TIER = "B"    # Top 40% - Good
    C_TIER = "C"    # Top 70% - Average
    D_TIER = "D"    # Bottom 30% - Below average
    UNRANKED = "U"  # Not enough data


class DataCharacteristic(Enum):
    """Data characteristics for model matching."""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE = "volatile"
    STABLE = "stable"
    SEASONAL = "seasonal"
    NOISY = "noisy"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ModelMetadata:
    """Rich metadata for a model type."""
    name: str
    category: str  # statistical, ml, dl, naive
    description: str
    
    # Capabilities
    supports_covariates: bool = False
    supports_multivariate: bool = False
    supports_probabilistic: bool = False
    supports_gpu: bool = False
    
    # Performance characteristics
    typical_training_time: str = "fast"  # fast, medium, slow
    memory_usage: str = "low"  # low, medium, high
    min_data_points: int = 50
    
    # Best use cases
    best_for: List[str] = field(default_factory=list)
    not_recommended_for: List[str] = field(default_factory=list)
    
    # Hyperparameters
    key_hyperparameters: Dict[str, Any] = field(default_factory=dict)
    default_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelPerformanceRecord:
    """Single performance record for a model run."""
    model_name: str
    symbol: str
    exchange: str
    timestamp: datetime
    
    # Core metrics
    mape: float  # Mean Absolute Percentage Error
    mae: float   # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    direction_accuracy: float  # % of correct direction predictions
    
    # Trading metrics
    simulated_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    
    # Context
    horizon: int = 24
    data_points_used: int = 0
    training_time_seconds: float = 0.0
    
    # Configuration used
    config_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelPerformanceRecord':
        d['timestamp'] = datetime.fromisoformat(d['timestamp'])
        return cls(**d)


@dataclass
class TrainedModelEntry:
    """Entry for a trained model in the registry."""
    model_name: str
    symbol: str
    exchange: str
    version: str
    
    # Status
    status: ModelStatus = ModelStatus.TRAINED
    tier: ModelTier = ModelTier.UNRANKED
    
    # Timestamps
    trained_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Performance
    latest_metrics: Dict[str, float] = field(default_factory=dict)
    historical_mape: List[float] = field(default_factory=list)
    
    # Storage
    model_path: Optional[str] = None
    config_hash: Optional[str] = None
    
    # Usage stats
    total_predictions: int = 0
    successful_predictions: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['status'] = self.status.value
        d['tier'] = self.tier.value
        d['trained_at'] = self.trained_at.isoformat()
        d['last_used_at'] = self.last_used_at.isoformat() if self.last_used_at else None
        d['expires_at'] = self.expires_at.isoformat() if self.expires_at else None
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainedModelEntry':
        d['status'] = ModelStatus(d['status'])
        d['tier'] = ModelTier(d['tier'])
        d['trained_at'] = datetime.fromisoformat(d['trained_at'])
        d['last_used_at'] = datetime.fromisoformat(d['last_used_at']) if d['last_used_at'] else None
        d['expires_at'] = datetime.fromisoformat(d['expires_at']) if d['expires_at'] else None
        return cls(**d)


# ============================================================================
# MODEL METADATA CATALOG
# ============================================================================

MODEL_CATALOG: Dict[str, ModelMetadata] = {
    # ===== STATISTICAL MODELS =====
    'Theta': ModelMetadata(
        name='Theta',
        category='statistical',
        description='Decomposition-based model that won M4 competition for certain data types. Fast and reliable.',
        supports_covariates=False,
        typical_training_time='fast',
        memory_usage='low',
        min_data_points=24,
        best_for=['trending data', 'medium-term forecasts', 'quick predictions'],
        not_recommended_for=['highly volatile data', 'regime changes'],
        key_hyperparameters={'theta': 'decomposition parameter'},
        default_config={'theta': 2}
    ),
    'ExponentialSmoothing': ModelMetadata(
        name='ExponentialSmoothing',
        category='statistical',
        description='Classic Holt-Winters model with trend and seasonality components.',
        supports_covariates=False,
        typical_training_time='fast',
        memory_usage='low',
        min_data_points=48,
        best_for=['seasonal data', 'stable trends', 'interpretable results'],
        not_recommended_for=['sudden changes', 'complex patterns'],
        key_hyperparameters={'seasonal_periods': 'seasonality length', 'trend': 'trend type'},
        default_config={'seasonal_periods': 24}
    ),
    'AutoARIMA': ModelMetadata(
        name='AutoARIMA',
        category='statistical',
        description='Automatic ARIMA model selection. Finds optimal (p,d,q) parameters.',
        supports_covariates=True,
        typical_training_time='medium',
        memory_usage='low',
        min_data_points=50,
        best_for=['stationary data', 'linear dependencies', 'automated selection'],
        not_recommended_for=['non-linear patterns', 'very long horizons'],
        key_hyperparameters={'max_p': 'max AR order', 'max_q': 'max MA order'},
        default_config={'max_p': 5, 'max_q': 5}
    ),
    'Prophet': ModelMetadata(
        name='Prophet',
        category='statistical',
        description='Facebook Prophet - handles holidays, seasonality, and trend changes.',
        supports_covariates=True,
        typical_training_time='medium',
        memory_usage='medium',
        min_data_points=100,
        best_for=['multiple seasonality', 'holidays', 'missing data'],
        not_recommended_for=['high-frequency data', 'sub-hourly forecasts'],
        key_hyperparameters={'changepoint_prior_scale': 'flexibility', 'seasonality_mode': 'additive/multiplicative'},
        default_config={'changepoint_prior_scale': 0.05}
    ),
    'ARIMA': ModelMetadata(
        name='ARIMA',
        category='statistical',
        description='Classic Auto-Regressive Integrated Moving Average model.',
        supports_covariates=True,
        typical_training_time='fast',
        memory_usage='low',
        min_data_points=30,
        best_for=['stationary data', 'short-term forecasts', 'interpretability'],
        not_recommended_for=['non-stationary data', 'complex seasonality'],
        key_hyperparameters={'p': 'AR order', 'd': 'differencing', 'q': 'MA order'},
        default_config={'p': 1, 'd': 1, 'q': 1}
    ),
    'FourTheta': ModelMetadata(
        name='FourTheta',
        category='statistical',
        description='Optimized Theta variant with 4 different theta values.',
        supports_covariates=False,
        typical_training_time='fast',
        memory_usage='low',
        min_data_points=24,
        best_for=['trending data', 'M4-style forecasting'],
        not_recommended_for=['volatile crypto', 'regime changes'],
        key_hyperparameters={'theta': 'multiple theta values'},
        default_config={}
    ),
    
    # ===== MACHINE LEARNING MODELS =====
    'XGBoost': ModelMetadata(
        name='XGBoost',
        category='ml',
        description='Gradient boosted trees - excellent for capturing non-linear patterns.',
        supports_covariates=True,
        supports_multivariate=True,
        typical_training_time='medium',
        memory_usage='medium',
        min_data_points=100,
        best_for=['complex patterns', 'feature importance', 'non-linear relationships'],
        not_recommended_for=['very small datasets', 'pure trend extrapolation'],
        key_hyperparameters={'lags': 'number of lags', 'n_estimators': 'number of trees', 'max_depth': 'tree depth'},
        default_config={'lags': 24, 'n_estimators': 100, 'max_depth': 6}
    ),
    'LightGBM': ModelMetadata(
        name='LightGBM',
        category='ml',
        description='Fast gradient boosting - best speed/accuracy trade-off.',
        supports_covariates=True,
        supports_multivariate=True,
        typical_training_time='fast',
        memory_usage='low',
        min_data_points=100,
        best_for=['large datasets', 'speed-critical applications', 'many features'],
        not_recommended_for=['very small datasets'],
        key_hyperparameters={'lags': 'number of lags', 'num_leaves': 'tree complexity'},
        default_config={'lags': 24, 'num_leaves': 31}
    ),
    'RandomForest': ModelMetadata(
        name='RandomForest',
        category='ml',
        description='Ensemble of decision trees - robust and interpretable.',
        supports_covariates=True,
        supports_multivariate=True,
        typical_training_time='medium',
        memory_usage='high',
        min_data_points=100,
        best_for=['outlier robustness', 'feature importance', 'stable predictions'],
        not_recommended_for=['capturing trends', 'extrapolation'],
        key_hyperparameters={'lags': 'number of lags', 'n_estimators': 'number of trees'},
        default_config={'lags': 24, 'n_estimators': 100}
    ),
    'CatBoost': ModelMetadata(
        name='CatBoost',
        category='ml',
        description='Gradient boosting with native categorical support.',
        supports_covariates=True,
        supports_multivariate=True,
        typical_training_time='medium',
        memory_usage='medium',
        min_data_points=100,
        best_for=['categorical features', 'ordered boosting', 'reduced overfitting'],
        not_recommended_for=['very small datasets'],
        key_hyperparameters={'lags': 'number of lags', 'iterations': 'boosting rounds'},
        default_config={'lags': 24, 'iterations': 100}
    ),
    
    # ===== DEEP LEARNING MODELS =====
    'NBEATS': ModelMetadata(
        name='NBEATS',
        category='dl',
        description='Neural Basis Expansion Analysis - state-of-the-art accuracy.',
        supports_covariates=False,
        supports_multivariate=True,
        supports_gpu=True,
        supports_probabilistic=True,
        typical_training_time='slow',
        memory_usage='high',
        min_data_points=200,
        best_for=['highest accuracy needed', 'enough data available', 'interpretable components'],
        not_recommended_for=['small datasets', 'real-time requirements'],
        key_hyperparameters={'input_chunk_length': 'lookback', 'output_chunk_length': 'horizon', 'num_stacks': 'model depth'},
        default_config={'input_chunk_length': 24, 'output_chunk_length': 12, 'num_stacks': 2}
    ),
    'NHiTS': ModelMetadata(
        name='NHiTS',
        category='dl',
        description='Neural Hierarchical Interpolation - fast DL with multi-scale learning.',
        supports_covariates=False,
        supports_multivariate=True,
        supports_gpu=True,
        typical_training_time='medium',
        memory_usage='medium',
        min_data_points=150,
        best_for=['long horizons', 'multi-scale patterns', 'faster than NBEATS'],
        not_recommended_for=['very short series'],
        key_hyperparameters={'input_chunk_length': 'lookback', 'output_chunk_length': 'horizon'},
        default_config={'input_chunk_length': 24, 'output_chunk_length': 12}
    ),
    'TFT': ModelMetadata(
        name='TFT',
        category='dl',
        description='Temporal Fusion Transformer - attention-based with interpretability.',
        supports_covariates=True,
        supports_multivariate=True,
        supports_gpu=True,
        supports_probabilistic=True,
        typical_training_time='slow',
        memory_usage='high',
        min_data_points=300,
        best_for=['rich covariates', 'interpretable attention', 'probabilistic forecasts'],
        not_recommended_for=['limited data', 'no covariates'],
        key_hyperparameters={'input_chunk_length': 'lookback', 'output_chunk_length': 'horizon', 'hidden_size': 'model size'},
        default_config={'input_chunk_length': 48, 'output_chunk_length': 24, 'hidden_size': 64}
    ),
    'Transformer': ModelMetadata(
        name='Transformer',
        category='dl',
        description='Vanilla transformer for time series - attention mechanism.',
        supports_covariates=True,
        supports_multivariate=True,
        supports_gpu=True,
        typical_training_time='slow',
        memory_usage='high',
        min_data_points=200,
        best_for=['long-range dependencies', 'parallel computation'],
        not_recommended_for=['small datasets', 'simple patterns'],
        key_hyperparameters={'input_chunk_length': 'lookback', 'd_model': 'embedding dimension'},
        default_config={'input_chunk_length': 24, 'output_chunk_length': 12}
    ),
    'TCN': ModelMetadata(
        name='TCN',
        category='dl',
        description='Temporal Convolutional Network - dilated convolutions.',
        supports_covariates=True,
        supports_multivariate=True,
        supports_gpu=True,
        typical_training_time='medium',
        memory_usage='medium',
        min_data_points=150,
        best_for=['local patterns', 'faster than transformers', 'causal convolutions'],
        not_recommended_for=['very long dependencies'],
        key_hyperparameters={'input_chunk_length': 'lookback', 'kernel_size': 'convolution size'},
        default_config={'input_chunk_length': 24, 'output_chunk_length': 12}
    ),
    'DLinear': ModelMetadata(
        name='DLinear',
        category='dl',
        description='Simple linear model that often beats complex DL - decomposition-based.',
        supports_covariates=False,
        supports_multivariate=True,
        supports_gpu=True,
        typical_training_time='fast',
        memory_usage='low',
        min_data_points=50,
        best_for=['simple baselines', 'trend-seasonal data', 'fast DL'],
        not_recommended_for=['complex non-linear patterns'],
        key_hyperparameters={'input_chunk_length': 'lookback'},
        default_config={'input_chunk_length': 24, 'output_chunk_length': 12}
    ),
    
    # ===== NAIVE MODELS =====
    'NaiveDrift': ModelMetadata(
        name='NaiveDrift',
        category='naive',
        description='Simple drift model - uses average change rate.',
        supports_covariates=False,
        typical_training_time='instant',
        memory_usage='low',
        min_data_points=2,
        best_for=['baselines', 'trending data', 'no training needed'],
        not_recommended_for=['volatile data', 'mean-reverting'],
        key_hyperparameters={},
        default_config={}
    ),
    'NaiveSeasonal': ModelMetadata(
        name='NaiveSeasonal',
        category='naive',
        description='Repeats last seasonal pattern.',
        supports_covariates=False,
        typical_training_time='instant',
        memory_usage='low',
        min_data_points=24,
        best_for=['strong seasonality', 'simple baseline'],
        not_recommended_for=['trending data'],
        key_hyperparameters={'K': 'seasonal period'},
        default_config={'K': 24}
    ),
}


# ============================================================================
# MODEL REGISTRY SINGLETON
# ============================================================================

class ModelRegistry:
    """
    Centralized model registry with performance tracking and recommendations.
    
    Singleton pattern ensures one registry instance across the application.
    """
    
    _instance: Optional['ModelRegistry'] = None
    _lock: Lock = Lock()
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the model registry.
        
        Args:
            storage_path: Directory for persisting models and registry state
        """
        if storage_path is None:
            storage_path = os.path.join(
                os.path.dirname(__file__), 
                '..', '..', 'model_registry'
            )
        
        self.storage_path = Path(storage_path)
        self.models_path = self.storage_path / 'models'
        self.registry_file = self.storage_path / 'registry.json'
        self.performance_file = self.storage_path / 'performance_history.json'
        
        # In-memory state
        self._trained_models: Dict[str, TrainedModelEntry] = {}
        self._performance_history: Dict[str, List[ModelPerformanceRecord]] = {}
        self._model_cache: Dict[str, Any] = {}  # Cache for loaded models
        
        # Initialize storage
        self._init_storage()
        self._load_state()
    
    @classmethod
    def get_instance(cls, storage_path: Optional[str] = None) -> 'ModelRegistry':
        """Get singleton instance of ModelRegistry."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(storage_path)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton (mainly for testing)."""
        with cls._lock:
            cls._instance = None
    
    def _init_storage(self):
        """Initialize storage directories."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
    
    def _load_state(self):
        """Load persisted registry state."""
        # Load registry
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    for key, entry_dict in data.items():
                        self._trained_models[key] = TrainedModelEntry.from_dict(entry_dict)
                logger.info(f"Loaded {len(self._trained_models)} model entries from registry")
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
        
        # Load performance history
        if self.performance_file.exists():
            try:
                with open(self.performance_file, 'r') as f:
                    data = json.load(f)
                    for key, records in data.items():
                        self._performance_history[key] = [
                            ModelPerformanceRecord.from_dict(r) for r in records
                        ]
                logger.info(f"Loaded performance history for {len(self._performance_history)} symbol-model pairs")
            except Exception as e:
                logger.warning(f"Failed to load performance history: {e}")
    
    def _save_state(self):
        """Persist registry state to disk."""
        # Save registry
        try:
            registry_data = {
                key: entry.to_dict() 
                for key, entry in self._trained_models.items()
            }
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
        
        # Save performance history
        try:
            history_data = {
                key: [r.to_dict() for r in records]
                for key, records in self._performance_history.items()
            }
            with open(self.performance_file, 'w') as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save performance history: {e}")
    
    # ========================================================================
    # MODEL REGISTRATION
    # ========================================================================
    
    def _get_key(self, model_name: str, symbol: str, exchange: str) -> str:
        """Generate unique key for a model-symbol-exchange combination."""
        return f"{model_name}:{symbol.upper()}:{exchange.lower()}"
    
    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate hash for model configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def register_trained_model(
        self,
        model_name: str,
        symbol: str,
        exchange: str,
        wrapper: Any,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
        save_model: bool = True,
        ttl_hours: int = 168  # 7 days default
    ) -> TrainedModelEntry:
        """
        Register a trained model in the registry.
        
        Args:
            model_name: Name of the model (e.g., 'XGBoost')
            symbol: Trading pair (e.g., 'BTCUSDT')
            exchange: Exchange name (e.g., 'binance')
            wrapper: DartsModelWrapper instance
            metrics: Performance metrics dict
            config: Model configuration used
            save_model: Whether to persist the model to disk
            ttl_hours: Time-to-live in hours
        
        Returns:
            TrainedModelEntry for the registered model
        """
        key = self._get_key(model_name, symbol, exchange)
        config_hash = self._get_config_hash(config or {})
        
        # Generate version
        existing = self._trained_models.get(key)
        if existing:
            version_num = int(existing.version.split('.')[0]) + 1
        else:
            version_num = 1
        version = f"{version_num}.0"
        
        # Create entry
        entry = TrainedModelEntry(
            model_name=model_name,
            symbol=symbol.upper(),
            exchange=exchange.lower(),
            version=version,
            status=ModelStatus.TRAINED,
            trained_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=ttl_hours),
            latest_metrics=metrics,
            config_hash=config_hash
        )
        
        # Save model to disk if requested
        if save_model and wrapper is not None:
            model_path = self._save_model_to_disk(key, version, wrapper)
            entry.model_path = str(model_path)
        
        # Store in registry
        self._trained_models[key] = entry
        
        # Record performance
        self._record_performance(model_name, symbol, exchange, metrics)
        
        # Update tier
        entry.tier = self._calculate_tier(model_name, symbol, exchange)
        
        # Persist
        self._save_state()
        
        logger.info(f"Registered model {key} version {version} with tier {entry.tier.value}")
        return entry
    
    def _save_model_to_disk(self, key: str, version: str, wrapper: Any) -> Path:
        """Save model wrapper to disk."""
        safe_key = key.replace(':', '_')
        model_dir = self.models_path / safe_key
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = model_dir / f"v{version}.pkl"
        
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(wrapper, f)
            logger.debug(f"Saved model to {model_file}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
        
        return model_file
    
    def _record_performance(
        self, 
        model_name: str, 
        symbol: str, 
        exchange: str, 
        metrics: Dict[str, float]
    ):
        """Record performance metrics."""
        key = self._get_key(model_name, symbol, exchange)
        
        record = ModelPerformanceRecord(
            model_name=model_name,
            symbol=symbol.upper(),
            exchange=exchange.lower(),
            timestamp=datetime.utcnow(),
            mape=metrics.get('mape', 0),
            mae=metrics.get('mae', 0),
            rmse=metrics.get('rmse', 0),
            direction_accuracy=metrics.get('direction_accuracy', 0),
            simulated_return=metrics.get('simulated_return'),
            sharpe_ratio=metrics.get('sharpe_ratio'),
            horizon=metrics.get('horizon', 24)
        )
        
        if key not in self._performance_history:
            self._performance_history[key] = []
        
        self._performance_history[key].append(record)
        
        # Keep only last 100 records per key
        if len(self._performance_history[key]) > 100:
            self._performance_history[key] = self._performance_history[key][-100:]
    
    def _calculate_tier(self, model_name: str, symbol: str, exchange: str) -> ModelTier:
        """Calculate performance tier for a model."""
        key = self._get_key(model_name, symbol, exchange)
        history = self._performance_history.get(key, [])
        
        if len(history) < 3:
            return ModelTier.UNRANKED
        
        # Calculate average direction accuracy (most important for trading)
        avg_direction = np.mean([r.direction_accuracy for r in history[-10:]])
        avg_mape = np.mean([r.mape for r in history[-10:]])
        
        # Composite score (direction is more important)
        score = avg_direction * 0.7 - avg_mape * 0.3
        
        if score >= 65:
            return ModelTier.S_TIER
        elif score >= 55:
            return ModelTier.A_TIER
        elif score >= 45:
            return ModelTier.B_TIER
        elif score >= 35:
            return ModelTier.C_TIER
        else:
            return ModelTier.D_TIER
    
    # ========================================================================
    # MODEL RETRIEVAL
    # ========================================================================
    
    def get_model_entry(
        self, 
        model_name: str, 
        symbol: str, 
        exchange: str
    ) -> Optional[TrainedModelEntry]:
        """Get trained model entry if exists."""
        key = self._get_key(model_name, symbol, exchange)
        return self._trained_models.get(key)
    
    def load_model(
        self, 
        model_name: str, 
        symbol: str, 
        exchange: str
    ) -> Optional[Any]:
        """
        Load a trained model from disk.
        
        Returns DartsModelWrapper if found, None otherwise.
        """
        key = self._get_key(model_name, symbol, exchange)
        
        # Check cache first
        if key in self._model_cache:
            logger.debug(f"Loading {key} from cache")
            return self._model_cache[key]
        
        # Get entry
        entry = self._trained_models.get(key)
        if entry is None or entry.model_path is None:
            return None
        
        # Check expiration
        if entry.expires_at and datetime.utcnow() > entry.expires_at:
            logger.warning(f"Model {key} has expired")
            return None
        
        # Load from disk
        try:
            model_path = Path(entry.model_path)
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return None
            
            with open(model_path, 'rb') as f:
                wrapper = pickle.load(f)
            
            # Update usage
            entry.last_used_at = datetime.utcnow()
            entry.total_predictions += 1
            self._save_state()
            
            # Cache
            self._model_cache[key] = wrapper
            
            logger.info(f"Loaded model {key} from disk")
            return wrapper
            
        except Exception as e:
            logger.error(f"Failed to load model {key}: {e}")
            return None
    
    def list_models(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        tier: Optional[ModelTier] = None
    ) -> List[TrainedModelEntry]:
        """
        List registered models with optional filters.
        """
        results = []
        
        for entry in self._trained_models.values():
            if symbol and entry.symbol != symbol.upper():
                continue
            if exchange and entry.exchange != exchange.lower():
                continue
            if status and entry.status != status:
                continue
            if tier and entry.tier != tier:
                continue
            results.append(entry)
        
        return sorted(results, key=lambda x: (x.tier.value, -x.latest_metrics.get('direction_accuracy', 0)))
    
    # ========================================================================
    # MODEL RECOMMENDATIONS
    # ========================================================================
    
    def recommend_model(
        self,
        symbol: str,
        exchange: str,
        data_characteristics: Optional[List[DataCharacteristic]] = None,
        prefer_fast: bool = False,
        require_trained: bool = True
    ) -> Tuple[str, float]:
        """
        Recommend the best model for a symbol/exchange.
        
        Args:
            symbol: Trading pair
            exchange: Exchange name
            data_characteristics: Optional hints about data
            prefer_fast: Prefer faster models over accurate ones
            require_trained: Only recommend models that are already trained
        
        Returns:
            Tuple of (model_name, confidence_score)
        """
        symbol = symbol.upper()
        exchange = exchange.lower()
        
        candidates = []
        
        # First, check trained models with good performance
        for entry in self._trained_models.values():
            if entry.symbol != symbol or entry.exchange != exchange:
                continue
            if entry.status in [ModelStatus.RETIRED, ModelStatus.DEPRECATED]:
                continue
            
            # Calculate score based on tier and recent performance
            tier_scores = {'S': 100, 'A': 85, 'B': 70, 'C': 55, 'D': 40, 'U': 50}
            base_score = tier_scores.get(entry.tier.value, 50)
            
            # Adjust for recent performance
            if entry.latest_metrics.get('direction_accuracy', 0) > 70:
                base_score += 10
            
            # Adjust for speed preference
            if prefer_fast:
                meta = MODEL_CATALOG.get(entry.model_name)
                if meta and meta.typical_training_time == 'fast':
                    base_score += 15
            
            candidates.append((entry.model_name, base_score, True))
        
        # If no trained models or not requiring trained, add default recommendations
        if not require_trained or not candidates:
            # Add untrained model recommendations based on characteristics
            if data_characteristics:
                if DataCharacteristic.TRENDING in data_characteristics:
                    candidates.append(('Theta', 75, False))
                    candidates.append(('NaiveDrift', 65, False))
                if DataCharacteristic.VOLATILE in data_characteristics:
                    candidates.append(('XGBoost', 80, False))
                    candidates.append(('LightGBM', 78, False))
                if DataCharacteristic.SEASONAL in data_characteristics:
                    candidates.append(('Prophet', 80, False))
                    candidates.append(('ExponentialSmoothing', 75, False))
            else:
                # Default recommendations
                if prefer_fast:
                    candidates.append(('Theta', 70, False))
                    candidates.append(('LightGBM', 75, False))
                else:
                    candidates.append(('XGBoost', 75, False))
                    candidates.append(('NHiTS', 70, False))
        
        if not candidates:
            return ('Theta', 50.0)  # Safe default
        
        # Sort by score
        candidates.sort(key=lambda x: -x[1])
        best = candidates[0]
        
        return (best[0], best[1])
    
    def get_model_rankings(
        self,
        symbol: str,
        exchange: str,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get ranked list of models for a symbol.
        
        Returns list of dicts with model info and scores.
        """
        symbol = symbol.upper()
        exchange = exchange.lower()
        
        rankings = []
        
        for entry in self._trained_models.values():
            if entry.symbol != symbol or entry.exchange != exchange:
                continue
            
            key = self._get_key(entry.model_name, symbol, exchange)
            history = self._performance_history.get(key, [])
            
            # Calculate stats
            if history:
                avg_mape = np.mean([r.mape for r in history[-10:]])
                avg_direction = np.mean([r.direction_accuracy for r in history[-10:]])
                best_direction = max(r.direction_accuracy for r in history)
                runs = len(history)
            else:
                avg_mape = avg_direction = best_direction = 0
                runs = 0
            
            rankings.append({
                'model': entry.model_name,
                'tier': entry.tier.value,
                'version': entry.version,
                'avg_mape': round(avg_mape, 2),
                'avg_direction_accuracy': round(avg_direction, 1),
                'best_direction_accuracy': round(best_direction, 1),
                'total_runs': runs,
                'status': entry.status.value,
                'trained_at': entry.trained_at.isoformat()
            })
        
        # Sort by direction accuracy
        rankings.sort(key=lambda x: -x['avg_direction_accuracy'])
        
        return rankings[:top_n]
    
    # ========================================================================
    # METADATA ACCESS
    # ========================================================================
    
    def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """Get metadata for a model type."""
        return MODEL_CATALOG.get(model_name)
    
    def list_all_models(self) -> Dict[str, List[str]]:
        """List all available models by category."""
        categories: Dict[str, List[str]] = {
            'statistical': [],
            'ml': [],
            'dl': [],
            'naive': []
        }
        
        for name, meta in MODEL_CATALOG.items():
            categories[meta.category].append(name)
        
        return categories
    
    def get_model_comparison(self, model_names: List[str]) -> List[Dict[str, Any]]:
        """Compare multiple models by their characteristics."""
        comparisons = []
        
        for name in model_names:
            meta = MODEL_CATALOG.get(name)
            if meta:
                comparisons.append({
                    'name': name,
                    'category': meta.category,
                    'training_time': meta.typical_training_time,
                    'memory_usage': meta.memory_usage,
                    'min_data_points': meta.min_data_points,
                    'supports_covariates': meta.supports_covariates,
                    'supports_gpu': meta.supports_gpu,
                    'best_for': meta.best_for,
                    'not_recommended_for': meta.not_recommended_for
                })
        
        return comparisons
    
    # ========================================================================
    # LIFECYCLE MANAGEMENT
    # ========================================================================
    
    def update_status(
        self,
        model_name: str,
        symbol: str,
        exchange: str,
        new_status: ModelStatus
    ) -> bool:
        """Update model status."""
        key = self._get_key(model_name, symbol, exchange)
        entry = self._trained_models.get(key)
        
        if entry is None:
            return False
        
        entry.status = new_status
        self._save_state()
        
        logger.info(f"Updated {key} status to {new_status.value}")
        return True
    
    def deploy_model(
        self,
        model_name: str,
        symbol: str,
        exchange: str
    ) -> bool:
        """Mark a model as deployed for production use."""
        return self.update_status(model_name, symbol, exchange, ModelStatus.DEPLOYED)
    
    def retire_model(
        self,
        model_name: str,
        symbol: str,
        exchange: str
    ) -> bool:
        """Retire a model (no longer usable)."""
        key = self._get_key(model_name, symbol, exchange)
        entry = self._trained_models.get(key)
        
        if entry is None:
            return False
        
        entry.status = ModelStatus.RETIRED
        
        # Clear from cache
        if key in self._model_cache:
            del self._model_cache[key]
        
        self._save_state()
        
        logger.info(f"Retired model {key}")
        return True
    
    def cleanup_expired(self) -> int:
        """Clean up expired models. Returns count of cleaned models."""
        now = datetime.utcnow()
        cleaned = 0
        
        for key, entry in list(self._trained_models.items()):
            if entry.expires_at and now > entry.expires_at:
                entry.status = ModelStatus.DEPRECATED
                cleaned += 1
                
                # Remove from cache
                if key in self._model_cache:
                    del self._model_cache[key]
        
        if cleaned > 0:
            self._save_state()
            logger.info(f"Deprecated {cleaned} expired models")
        
        return cleaned
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get overall registry statistics."""
        total = len(self._trained_models)
        by_status = {}
        by_tier = {}
        by_symbol = {}
        by_model = {}
        
        for entry in self._trained_models.values():
            by_status[entry.status.value] = by_status.get(entry.status.value, 0) + 1
            by_tier[entry.tier.value] = by_tier.get(entry.tier.value, 0) + 1
            by_symbol[entry.symbol] = by_symbol.get(entry.symbol, 0) + 1
            by_model[entry.model_name] = by_model.get(entry.model_name, 0) + 1
        
        total_performance_records = sum(
            len(records) for records in self._performance_history.values()
        )
        
        return {
            'total_models': total,
            'by_status': by_status,
            'by_tier': by_tier,
            'by_symbol': by_symbol,
            'by_model_type': by_model,
            'total_performance_records': total_performance_records,
            'cache_size': len(self._model_cache),
            'storage_path': str(self.storage_path)
        }
