"""
🧠 Institutional Feature Engineering Framework
==============================================

Advanced feature calculation system for institutional-grade market intelligence.

Architecture:
- schemas.py: Feature table definitions (DuckDB)
- storage.py: Feature storage manager
- base.py: Base class for real-time feature calculators
- calculators/: Per-stream feature calculators
- composite.py: Cross-stream composite signals
- integration.py: Data collector integration

Features:
- Real-time feature calculation on streaming data
- Batch feature aggregation for historical analysis
- Composite signals (smart money, squeeze, stop hunt)
- Integration with IsolatedDataCollector callbacks
"""

from .schemas import (
    FEATURE_TABLE_SCHEMAS,
    COMPOSITE_TABLE_SCHEMAS,
    AGGREGATED_TABLE_SCHEMAS,
    create_feature_tables_for_symbol,
)

from .storage import InstitutionalFeatureStorage

from .base import (
    InstitutionalFeatureCalculator,
    FeatureBuffer,
)

from .composite import CompositeSignalCalculator, CompositeSignal, EnhancedSignal

from .signal_aggregator import (
    SignalAggregator,
    SignalDirection,
    SignalCategory,
    RecommendationStrength,
    ConflictSeverity,
    RankedSignal,
    SignalConflict,
    TradeRecommendation,
    AggregatedIntelligence,
)

from .integration import (
    FeatureEngine,
    create_feature_engine,
    integrate_with_collector,
)

__all__ = [
    # Schemas
    'FEATURE_TABLE_SCHEMAS',
    'COMPOSITE_TABLE_SCHEMAS',
    'AGGREGATED_TABLE_SCHEMAS',
    'create_feature_tables_for_symbol',
    
    # Storage
    'InstitutionalFeatureStorage',
    
    # Base
    'InstitutionalFeatureCalculator',
    'FeatureBuffer',
    
    # Composite
    'CompositeSignalCalculator',
    'CompositeSignal',
    'EnhancedSignal',
    
    # Signal Aggregator (Phase 3 Week 3-4)
    'SignalAggregator',
    'SignalDirection',
    'SignalCategory',
    'RecommendationStrength',
    'ConflictSeverity',
    'RankedSignal',
    'SignalConflict',
    'TradeRecommendation',
    'AggregatedIntelligence',
    
    # Integration
    'FeatureEngine',
    'create_feature_engine',
    'integrate_with_collector',
]
