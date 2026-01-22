"""
Integrations Module

Third-party library integrations for enhanced analytics capabilities.
"""

from .darts_bridge import (
    DartsBridge,
    DartsModelWrapper,
    DartsEnsemble,
    DartsAutoML,
    DartsForecastResult,
    ModelCategory,
    STATISTICAL_MODELS,
    DEEP_LEARNING_MODELS,
    ML_MODELS,
    ALL_MODELS
)

from .model_registry import (
    ModelRegistry,
    ModelMetadata,
    ModelPerformanceRecord,
    TrainedModelEntry,
    ModelStatus,
    ModelTier,
    DataCharacteristic,
    MODEL_CATALOG
)

__all__ = [
    # Core Bridge
    'DartsBridge',
    'DartsModelWrapper',
    'DartsEnsemble',
    'DartsAutoML',
    'DartsForecastResult',
    
    # Enums & Constants
    'ModelCategory',
    'STATISTICAL_MODELS',
    'DEEP_LEARNING_MODELS',
    'ML_MODELS',
    'ALL_MODELS',
    
    # Model Registry
    'ModelRegistry',
    'ModelMetadata',
    'ModelPerformanceRecord',
    'TrainedModelEntry',
    'ModelStatus',
    'ModelTier',
    'DataCharacteristic',
    'MODEL_CATALOG',
]
