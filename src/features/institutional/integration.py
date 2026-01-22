"""
🔗 Feature Engine Integration - Phase 3 Real-Time
=================================================

Integrates institutional feature calculation with the data collection pipeline.

This module:
1. Hooks into IsolatedDataCollector callbacks
2. Routes data to appropriate feature calculators
3. Calculates composite signals
4. Aggregates signals and generates recommendations (Phase 3)
5. Stores features, signals, and recommendations in DuckDB

Architecture (Phase 3 Enhanced):
    IsolatedDataCollector
          ↓ callbacks
    FeatureEngine.on_data_update()
          ↓ routes by stream type
    PricesCalculator / OrderbookCalculator / etc
          ↓ features
    CompositeSignalCalculator
          ↓ composite signals
    SignalAggregator (NEW in Phase 3)
          ↓ ranked signals + recommendations
    InstitutionalFeatureStorage → DuckDB

Usage:
    from src.features.institutional.integration import FeatureEngine
    
    # Initialize with real-time mode
    engine = FeatureEngine(db_manager, enable_realtime=True)
    
    # Connect to data collector
    collector.register_data_callback(engine.on_data_update)
    
    # Get aggregated intelligence
    intelligence = engine.get_aggregated_intelligence("BTCUSDT", "binance")
    recommendation = intelligence.recommendation
"""

import logging
import asyncio
import time
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timezone
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
import queue

from .base import InstitutionalFeatureCalculator, institutional_feature_registry
from .storage import InstitutionalFeatureStorage
from .composite import CompositeSignalCalculator, CompositeSignal, EnhancedSignal
from .signal_aggregator import (
    SignalAggregator,
    AggregatedIntelligence,
    TradeRecommendation,
    RankedSignal,
    SignalConflict,
    SignalDirection,
    RecommendationStrength,
)
from .calculators import (
    # Phase 1
    PricesFeatureCalculator,
    OrderbookFeatureCalculator,
    TradesFeatureCalculator,
    FundingFeatureCalculator,
    OIFeatureCalculator,
    # Phase 2
    LiquidationsFeatureCalculator,
    MarkPricesFeatureCalculator,
    TickerFeatureCalculator,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PERFORMANCE OPTIMIZATION: Batching & Caching
# =============================================================================

@dataclass
class FeatureCache:
    """Cache for recently calculated features to avoid redundant calculations."""
    features: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: float = 1.0  # Cache valid for 1 second
    
    def is_valid(self) -> bool:
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age < self.ttl_seconds
    
    def update(self, features: Dict[str, Any]):
        self.features = features
        self.timestamp = datetime.now(timezone.utc)


@dataclass
class SignalBatch:
    """Batch of signals waiting to be stored."""
    signals: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    max_size: int = 100
    max_age_seconds: float = 5.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def should_flush(self) -> bool:
        if len(self.signals) >= self.max_size:
            return True
        if len(self.recommendations) >= self.max_size:
            return True
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age >= self.max_age_seconds
    
    def add_signal(self, signal_data: Dict[str, Any]):
        self.signals.append(signal_data)
    
    def add_recommendation(self, rec_data: Dict[str, Any]):
        self.recommendations.append(rec_data)
    
    def clear(self):
        self.signals.clear()
        self.recommendations.clear()
        self.created_at = datetime.now(timezone.utc)


class FeatureEngine:
    """
    Central feature calculation engine with real-time signal aggregation.
    
    Manages per-symbol feature calculators and coordinates feature storage.
    
    Phase 3 Enhanced Architecture:
        IsolatedDataCollector
              ↓ callbacks
        FeatureEngine.on_data_update()
              ↓ routes by stream type
        PricesCalculator / OrderbookCalculator / etc
              ↓ features (cached)
        CompositeSignalCalculator
              ↓ composite signals
        SignalAggregator (Phase 3)
              ↓ ranked signals + recommendations
        InstitutionalFeatureStorage → DuckDB
              ↓ (batched writes)
        Real-time Callbacks → External Systems
    """
    
    def __init__(
        self,
        db_manager: Any,
        storage: Optional[InstitutionalFeatureStorage] = None,
        enable_composites: bool = True,
        enable_realtime: bool = True,
        enable_aggregation: bool = True,
        batch_interval: float = 5.0,
        aggregation_interval: float = 1.0,
        cache_ttl: float = 0.5,
    ):
        """
        Initialize the feature engine.
        
        Args:
            db_manager: DuckDB manager instance
            storage: Optional pre-configured storage (will create if None)
            enable_composites: Whether to calculate composite signals
            enable_realtime: Enable real-time mode with caching/batching
            enable_aggregation: Enable SignalAggregator for recommendations
            batch_interval: How often to flush features to storage (seconds)
            aggregation_interval: Min interval between aggregations (seconds)
            cache_ttl: Feature cache time-to-live (seconds)
        """
        self.db_manager = db_manager
        self.storage = storage or InstitutionalFeatureStorage(db_manager)
        self.enable_composites = enable_composites
        self.enable_realtime = enable_realtime
        self.enable_aggregation = enable_aggregation
        self.batch_interval = batch_interval
        self.aggregation_interval = aggregation_interval
        self.cache_ttl = cache_ttl
        
        # Per-symbol calculators
        self._calculators: Dict[str, Dict[str, InstitutionalFeatureCalculator]] = defaultdict(dict)
        
        # Per-symbol composite calculators
        self._composite_calculators: Dict[str, CompositeSignalCalculator] = {}
        
        # Per-symbol signal aggregators (Phase 3)
        self._signal_aggregators: Dict[str, SignalAggregator] = {}
        
        # Per-symbol aggregated intelligence cache (Phase 3)
        self._intelligence_cache: Dict[str, AggregatedIntelligence] = {}
        self._last_aggregation: Dict[str, datetime] = {}
        
        # Feature caches for performance
        self._feature_caches: Dict[str, Dict[str, FeatureCache]] = defaultdict(dict)
        
        # Signal batches for efficient storage
        self._signal_batches: Dict[str, SignalBatch] = defaultdict(SignalBatch)
        
        # Thread pool for async feature calculation
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Thread-safe lock for shared state
        self._lock = Lock()
        
        # Real-time callbacks for external systems
        self._signal_callbacks: List[Callable] = []
        self._recommendation_callbacks: List[Callable] = []
        
        # Metrics
        self._metrics = {
            'features_calculated': 0,
            'composites_calculated': 0,
            'aggregations_calculated': 0,
            'recommendations_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batches_flushed': 0,
            'errors': 0,
            'avg_processing_time_ms': 0.0,
            'processing_times': [],
        }
        
        # Running flag
        self._running = False
        
        logger.info(f"FeatureEngine initialized (realtime={enable_realtime}, aggregation={enable_aggregation})")
    
    def _get_symbol_key(self, symbol: str, exchange: str) -> str:
        """Create unique key for symbol+exchange."""
        return f"{exchange}:{symbol}"
    
    def _get_or_create_calculator(
        self, 
        symbol: str, 
        exchange: str, 
        stream_type: str
    ) -> Optional[InstitutionalFeatureCalculator]:
        """Get or create a feature calculator for a symbol/stream combination."""
        key = self._get_symbol_key(symbol, exchange)
        
        if stream_type not in self._calculators[key]:
            # Create new calculator
            calc_class = {
                # Phase 1 calculators
                'prices': PricesFeatureCalculator,
                'orderbook': OrderbookFeatureCalculator,
                'trades': TradesFeatureCalculator,
                'funding': FundingFeatureCalculator,
                'oi': OIFeatureCalculator,
                # Phase 2 calculators
                'liquidations': LiquidationsFeatureCalculator,
                'mark_prices': MarkPricesFeatureCalculator,
                'ticker': TickerFeatureCalculator,
            }.get(stream_type)
            
            if calc_class:
                self._calculators[key][stream_type] = calc_class(symbol, exchange)
                logger.debug(f"Created {stream_type} calculator for {key}")
        
        return self._calculators[key].get(stream_type)
    
    def _get_or_create_composite(self, symbol: str, exchange: str) -> CompositeSignalCalculator:
        """Get or create composite calculator for a symbol."""
        key = self._get_symbol_key(symbol, exchange)
        
        if key not in self._composite_calculators:
            self._composite_calculators[key] = CompositeSignalCalculator()
        
        return self._composite_calculators[key]
    
    def _get_or_create_aggregator(self, symbol: str, exchange: str) -> SignalAggregator:
        """Get or create signal aggregator for a symbol (Phase 3)."""
        key = self._get_symbol_key(symbol, exchange)
        
        if key not in self._signal_aggregators:
            self._signal_aggregators[key] = SignalAggregator(
                recency_decay=0.95,
                confidence_weight=0.3,
                correlation_penalty=0.1,
                conflict_threshold=0.25,
            )
        
        return self._signal_aggregators[key]
    
    def _get_feature_cache(self, symbol: str, exchange: str, stream_type: str) -> FeatureCache:
        """Get or create feature cache for a stream."""
        key = self._get_symbol_key(symbol, exchange)
        
        if stream_type not in self._feature_caches[key]:
            self._feature_caches[key][stream_type] = FeatureCache(ttl_seconds=self.cache_ttl)
        
        return self._feature_caches[key][stream_type]
    
    # === Callback Registration (Phase 3) ===
    
    def register_signal_callback(self, callback: Callable):
        """
        Register callback for real-time signal updates.
        
        Callback signature: (symbol: str, exchange: str, signals: Dict[str, CompositeSignal]) -> None
        """
        self._signal_callbacks.append(callback)
    
    def register_recommendation_callback(self, callback: Callable):
        """
        Register callback for real-time recommendations.
        
        Callback signature: (symbol: str, exchange: str, recommendation: TradeRecommendation) -> None
        """
        self._recommendation_callbacks.append(callback)
    
    def _notify_signal_callbacks(self, symbol: str, exchange: str, signals: Dict[str, Any]):
        """Notify all registered signal callbacks."""
        for callback in self._signal_callbacks:
            try:
                callback(symbol, exchange, signals)
            except Exception as e:
                logger.error(f"Error in signal callback: {e}")
    
    def _notify_recommendation_callbacks(
        self, 
        symbol: str, 
        exchange: str, 
        recommendation: TradeRecommendation
    ):
        """Notify all registered recommendation callbacks."""
        for callback in self._recommendation_callbacks:
            try:
                callback(symbol, exchange, recommendation)
            except Exception as e:
                logger.error(f"Error in recommendation callback: {e}")
    
    async def start(self):
        """Start the feature engine (background processing)."""
        self._running = True
        
        # Start background batch flusher if in real-time mode
        if self.enable_realtime:
            asyncio.create_task(self._batch_flush_loop())
        
        logger.info("FeatureEngine started")
    
    async def stop(self):
        """Stop the feature engine and flush remaining data."""
        self._running = False
        
        # Flush all pending batches
        await self._flush_all_batches()
        
        await self.storage.flush_all()
        self._executor.shutdown(wait=True)
        logger.info("FeatureEngine stopped")
    
    async def _batch_flush_loop(self):
        """Background loop to flush signal batches periodically."""
        while self._running:
            await asyncio.sleep(self.batch_interval)
            await self._flush_all_batches()
    
    async def _flush_all_batches(self):
        """Flush all pending signal batches to storage."""
        with self._lock:
            for key, batch in self._signal_batches.items():
                if batch.signals or batch.recommendations:
                    await self._flush_batch(key, batch)
                    batch.clear()
                    self._metrics['batches_flushed'] += 1
    
    async def _flush_batch(self, key: str, batch: SignalBatch):
        """Flush a single batch to storage."""
        try:
            # Store signals
            for signal_data in batch.signals:
                self.storage.store_composite_signals(
                    symbol=signal_data['symbol'],
                    exchange=signal_data['exchange'],
                    signals=signal_data['signals'],
                    timestamp=signal_data['timestamp'],
                )
            
            # Store recommendations
            for rec_data in batch.recommendations:
                self._store_recommendation(rec_data)
                
        except Exception as e:
            logger.error(f"Error flushing batch {key}: {e}")
            self._metrics['errors'] += 1
    
    def _store_recommendation(self, rec_data: Dict[str, Any]):
        """Store a recommendation to the database."""
        # For now, store as part of composite signals
        # In production, this would go to a dedicated recommendations table
        pass
    
    # === Data Processing Methods ===
    
    def on_data_update(
        self,
        symbol: str,
        exchange: str,
        stream_type: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        """
        Universal callback for data updates from IsolatedDataCollector.
        
        This is the main entry point for processing streaming data.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            stream_type: Type of data ('prices', 'orderbook', 'trades', 'funding', 'oi')
            data: Raw data from the stream
            timestamp: Optional timestamp (uses current time if None)
        """
        try:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)
            
            # Route to appropriate processor
            if stream_type == 'prices':
                self.process_prices(symbol, exchange, data, timestamp)
            elif stream_type == 'orderbook':
                self.process_orderbook(symbol, exchange, data, timestamp)
            elif stream_type == 'trades':
                self.process_trades(symbol, exchange, data, timestamp)
            elif stream_type == 'funding':
                self.process_funding(symbol, exchange, data, timestamp)
            elif stream_type in ('oi', 'open_interest'):
                self.process_oi(symbol, exchange, data, timestamp)
            # Phase 2 stream types
            elif stream_type == 'liquidations':
                self.process_liquidations(symbol, exchange, data, timestamp)
            elif stream_type == 'mark_prices':
                self.process_mark_prices(symbol, exchange, data, timestamp)
            elif stream_type == 'ticker':
                self.process_ticker(symbol, exchange, data, timestamp)
            else:
                logger.debug(f"Unknown stream type: {stream_type}")
                
        except Exception as e:
            logger.error(f"Error processing {stream_type} data for {symbol}@{exchange}: {e}")
            self._metrics['errors'] += 1
    
    def process_prices(
        self,
        symbol: str,
        exchange: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process prices data and calculate features.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data: Price data (bid, ask, last, etc.)
            timestamp: Optional timestamp
        
        Returns:
            Calculated features dict or None
        """
        calc = self._get_or_create_calculator(symbol, exchange, 'prices')
        if not calc:
            return None
        
        try:
            features = calc.calculate(data)
            if features:
                self._metrics['features_calculated'] += 1
                
                # Store features
                self.storage.store_features(
                    symbol=symbol,
                    exchange=exchange,
                    feature_type='prices',
                    features=features,
                    timestamp=timestamp or datetime.now(timezone.utc)
                )
                
                # Update composite
                if self.enable_composites:
                    composite = self._get_or_create_composite(symbol, exchange)
                    composite.update_features('prices', features)
                    self._update_composite_signals(symbol, exchange, timestamp)
                
                return features
                
        except Exception as e:
            logger.error(f"Error calculating prices features for {symbol}@{exchange}: {e}")
            self._metrics['errors'] += 1
        
        return None
    
    def process_orderbook(
        self,
        symbol: str,
        exchange: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """Process orderbook data and calculate features."""
        calc = self._get_or_create_calculator(symbol, exchange, 'orderbook')
        if not calc:
            return None
        
        try:
            features = calc.calculate(data)
            if features:
                self._metrics['features_calculated'] += 1
                
                self.storage.store_features(
                    symbol=symbol,
                    exchange=exchange,
                    feature_type='orderbook',
                    features=features,
                    timestamp=timestamp or datetime.now(timezone.utc)
                )
                
                if self.enable_composites:
                    composite = self._get_or_create_composite(symbol, exchange)
                    composite.update_features('orderbook', features)
                    self._update_composite_signals(symbol, exchange, timestamp)
                
                return features
                
        except Exception as e:
            logger.error(f"Error calculating orderbook features for {symbol}@{exchange}: {e}")
            self._metrics['errors'] += 1
        
        return None
    
    def process_trades(
        self,
        symbol: str,
        exchange: str,
        data: Any,  # Can be single trade dict or list of trades
        timestamp: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """Process trades data and calculate features."""
        calc = self._get_or_create_calculator(symbol, exchange, 'trades')
        if not calc:
            return None
        
        try:
            features = calc.calculate(data)
            if features:
                self._metrics['features_calculated'] += 1
                
                self.storage.store_features(
                    symbol=symbol,
                    exchange=exchange,
                    feature_type='trades',
                    features=features,
                    timestamp=timestamp or datetime.now(timezone.utc)
                )
                
                if self.enable_composites:
                    composite = self._get_or_create_composite(symbol, exchange)
                    composite.update_features('trades', features)
                    self._update_composite_signals(symbol, exchange, timestamp)
                
                return features
                
        except Exception as e:
            logger.error(f"Error calculating trades features for {symbol}@{exchange}: {e}")
            self._metrics['errors'] += 1
        
        return None
    
    def process_funding(
        self,
        symbol: str,
        exchange: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """Process funding data and calculate features."""
        calc = self._get_or_create_calculator(symbol, exchange, 'funding')
        if not calc:
            return None
        
        try:
            features = calc.calculate(data)
            if features:
                self._metrics['features_calculated'] += 1
                
                self.storage.store_features(
                    symbol=symbol,
                    exchange=exchange,
                    feature_type='funding',
                    features=features,
                    timestamp=timestamp or datetime.now(timezone.utc)
                )
                
                if self.enable_composites:
                    composite = self._get_or_create_composite(symbol, exchange)
                    composite.update_features('funding', features)
                    self._update_composite_signals(symbol, exchange, timestamp)
                
                return features
                
        except Exception as e:
            logger.error(f"Error calculating funding features for {symbol}@{exchange}: {e}")
            self._metrics['errors'] += 1
        
        return None
    
    def process_oi(
        self,
        symbol: str,
        exchange: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """Process open interest data and calculate features."""
        calc = self._get_or_create_calculator(symbol, exchange, 'oi')
        if not calc:
            return None
        
        try:
            features = calc.calculate(data)
            if features:
                self._metrics['features_calculated'] += 1
                
                self.storage.store_features(
                    symbol=symbol,
                    exchange=exchange,
                    feature_type='oi',
                    features=features,
                    timestamp=timestamp or datetime.now(timezone.utc)
                )
                
                if self.enable_composites:
                    composite = self._get_or_create_composite(symbol, exchange)
                    composite.update_features('oi', features)
                    self._update_composite_signals(symbol, exchange, timestamp)
                
                return features
                
        except Exception as e:
            logger.error(f"Error calculating OI features for {symbol}@{exchange}: {e}")
            self._metrics['errors'] += 1
        
        return None
    
    # === Phase 2 Processing Methods ===
    
    def process_liquidations(
        self,
        symbol: str,
        exchange: str,
        data: Any,  # Can be single liquidation or list
        timestamp: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process liquidation data and calculate features.
        
        Detects liquidation cascades, clusters, and exhaustion patterns.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data: Liquidation data (side, quantity, price, etc.)
            timestamp: Optional timestamp
        
        Returns:
            Calculated features dict or None
        """
        calc = self._get_or_create_calculator(symbol, exchange, 'liquidations')
        if not calc:
            return None
        
        try:
            features = calc.calculate(data)
            if features:
                self._metrics['features_calculated'] += 1
                
                self.storage.store_features(
                    symbol=symbol,
                    exchange=exchange,
                    feature_type='liquidations',
                    features=features,
                    timestamp=timestamp or datetime.now(timezone.utc)
                )
                
                if self.enable_composites:
                    composite = self._get_or_create_composite(symbol, exchange)
                    composite.update_features('liquidations', features)
                    self._update_composite_signals(symbol, exchange, timestamp)
                
                return features
                
        except Exception as e:
            logger.error(f"Error calculating liquidations features for {symbol}@{exchange}: {e}")
            self._metrics['errors'] += 1
        
        return None
    
    def process_mark_prices(
        self,
        symbol: str,
        exchange: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process mark price data and calculate features.
        
        Calculates basis, funding pressure, and dislocation signals.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data: Mark price data (mark_price, index_price, spot_price, etc.)
            timestamp: Optional timestamp
        
        Returns:
            Calculated features dict or None
        """
        calc = self._get_or_create_calculator(symbol, exchange, 'mark_prices')
        if not calc:
            return None
        
        try:
            features = calc.calculate(data)
            if features:
                self._metrics['features_calculated'] += 1
                
                self.storage.store_features(
                    symbol=symbol,
                    exchange=exchange,
                    feature_type='mark_prices',
                    features=features,
                    timestamp=timestamp or datetime.now(timezone.utc)
                )
                
                if self.enable_composites:
                    composite = self._get_or_create_composite(symbol, exchange)
                    composite.update_features('mark_prices', features)
                    self._update_composite_signals(symbol, exchange, timestamp)
                
                return features
                
        except Exception as e:
            logger.error(f"Error calculating mark_prices features for {symbol}@{exchange}: {e}")
            self._metrics['errors'] += 1
        
        return None
    
    def process_ticker(
        self,
        symbol: str,
        exchange: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process ticker data and calculate features.
        
        Analyzes volume profile, range characteristics, and market strength.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data: Ticker data (volume_24h, high_24h, low_24h, price_change_24h, etc.)
            timestamp: Optional timestamp
        
        Returns:
            Calculated features dict or None
        """
        calc = self._get_or_create_calculator(symbol, exchange, 'ticker')
        if not calc:
            return None
        
        try:
            features = calc.calculate(data)
            if features:
                self._metrics['features_calculated'] += 1
                
                self.storage.store_features(
                    symbol=symbol,
                    exchange=exchange,
                    feature_type='ticker',
                    features=features,
                    timestamp=timestamp or datetime.now(timezone.utc)
                )
                
                if self.enable_composites:
                    composite = self._get_or_create_composite(symbol, exchange)
                    composite.update_features('ticker', features)
                    self._update_composite_signals(symbol, exchange, timestamp)
                
                return features
                
        except Exception as e:
            logger.error(f"Error calculating ticker features for {symbol}@{exchange}: {e}")
            self._metrics['errors'] += 1
        
        return None
    
    def _update_composite_signals(
        self,
        symbol: str,
        exchange: str,
        timestamp: Optional[datetime] = None
    ):
        """Calculate and store composite signals for a symbol (Phase 3 Enhanced)."""
        start_time = time.time()
        key = self._get_symbol_key(symbol, exchange)
        
        try:
            composite = self._get_or_create_composite(symbol, exchange)
            signals = composite.calculate_all(timestamp)
            
            if signals:
                self._metrics['composites_calculated'] += 1
                
                # Convert to storage format
                signals_dict = {}
                for name, signal in signals.items():
                    if isinstance(signal, EnhancedSignal):
                        signals_dict[name] = signal.value
                        signals_dict[f'{name}_confidence'] = signal.confidence
                        # Store metadata keys
                        for meta_key, meta_val in signal.metadata.items():
                            if isinstance(meta_val, (int, float, str, bool)):
                                signals_dict[f'{name}_{meta_key}'] = meta_val
                    elif isinstance(signal, CompositeSignal):
                        signals_dict[name] = signal.value
                        signals_dict[f'{name}_confidence'] = signal.confidence
                
                # Batch signals for storage
                if self.enable_realtime:
                    batch = self._signal_batches[key]
                    batch.add_signal({
                        'symbol': symbol,
                        'exchange': exchange,
                        'signals': signals_dict,
                        'timestamp': timestamp or datetime.now(timezone.utc),
                    })
                    
                    # Flush if batch is full
                    if batch.should_flush():
                        asyncio.create_task(self._flush_batch(key, batch))
                        batch.clear()
                else:
                    # Direct storage
                    self.storage.store_composite_signals(
                        symbol=symbol,
                        exchange=exchange,
                        signals=signals_dict,
                        timestamp=timestamp or datetime.now(timezone.utc)
                    )
                
                # Notify signal callbacks
                self._notify_signal_callbacks(symbol, exchange, signals)
                
                # Phase 3: Signal aggregation
                if self.enable_aggregation:
                    self._update_aggregation(symbol, exchange, signals, timestamp)
                
        except Exception as e:
            logger.error(f"Error calculating composite signals for {symbol}@{exchange}: {e}")
            self._metrics['errors'] += 1
        
        # Update processing time metrics
        processing_time = (time.time() - start_time) * 1000
        self._update_processing_metrics(processing_time)
    
    def _update_aggregation(
        self,
        symbol: str,
        exchange: str,
        signals: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        """
        Update signal aggregation and generate recommendations (Phase 3).
        
        Throttled to avoid excessive computation.
        """
        key = self._get_symbol_key(symbol, exchange)
        now = datetime.now(timezone.utc)
        
        # Throttle aggregation
        last_agg = self._last_aggregation.get(key)
        if last_agg:
            elapsed = (now - last_agg).total_seconds()
            if elapsed < self.aggregation_interval:
                return  # Skip, too soon
        
        try:
            composite = self._get_or_create_composite(symbol, exchange)
            aggregator = self._get_or_create_aggregator(symbol, exchange)
            
            # Get Phase 3 signals
            phase3_signals = composite.get_phase3_signals(timestamp)
            
            # Filter out EnhancedSignals from main signals dict for aggregator
            phase1_signals = {
                name: sig for name, sig in signals.items()
                if isinstance(sig, CompositeSignal) and not isinstance(sig, EnhancedSignal)
            }
            
            # Aggregate signals
            intelligence = aggregator.aggregate(
                symbol=symbol,
                signals=phase1_signals,
                phase3_signals=phase3_signals,
            )
            
            # Cache intelligence
            self._intelligence_cache[key] = intelligence
            self._last_aggregation[key] = now
            self._metrics['aggregations_calculated'] += 1
            
            # Generate recommendation
            if intelligence.recommendation:
                self._metrics['recommendations_generated'] += 1
                
                # Batch recommendation for storage
                if self.enable_realtime:
                    batch = self._signal_batches[key]
                    batch.add_recommendation({
                        'symbol': symbol,
                        'exchange': exchange,
                        'recommendation': intelligence.recommendation.to_dict(),
                        'timestamp': timestamp or now,
                    })
                
                # Notify recommendation callbacks
                self._notify_recommendation_callbacks(symbol, exchange, intelligence.recommendation)
                
        except Exception as e:
            logger.error(f"Error in signal aggregation for {symbol}@{exchange}: {e}")
            self._metrics['errors'] += 1
    
    def _update_processing_metrics(self, processing_time_ms: float):
        """Update processing time metrics."""
        times = self._metrics['processing_times']
        times.append(processing_time_ms)
        
        # Keep last 100 measurements
        if len(times) > 100:
            times.pop(0)
        
        self._metrics['avg_processing_time_ms'] = sum(times) / len(times) if times else 0.0
    
    # === Query Methods ===
    
    def get_latest_features(
        self,
        symbol: str,
        exchange: str,
        feature_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get most recent features for a symbol."""
        return self.storage.get_latest_features(symbol, exchange, feature_type)
    
    def get_latest_composites(
        self,
        symbol: str,
        exchange: str
    ) -> Optional[Dict[str, Any]]:
        """Get most recent composite signals for a symbol."""
        key = self._get_symbol_key(symbol, exchange)
        composite = self._composite_calculators.get(key)
        
        if composite:
            signals = composite.calculate_all()
            return {name: signal.to_dict() for name, signal in signals.items()}
        
        return None
    
    def get_aggregated_intelligence(
        self,
        symbol: str,
        exchange: str,
        force_recalculate: bool = False
    ) -> Optional[AggregatedIntelligence]:
        """
        Get aggregated intelligence for a symbol (Phase 3).
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            force_recalculate: Force recalculation even if cached
            
        Returns:
            AggregatedIntelligence with ranked signals and recommendation
        """
        key = self._get_symbol_key(symbol, exchange)
        
        # Return cached if available and not forcing recalculate
        if not force_recalculate and key in self._intelligence_cache:
            cached = self._intelligence_cache[key]
            # Check if cache is still valid (within aggregation interval)
            last_agg = self._last_aggregation.get(key)
            if last_agg:
                age = (datetime.now(timezone.utc) - last_agg).total_seconds()
                if age < self.aggregation_interval * 2:
                    return cached
        
        # Force calculation
        composite = self._composite_calculators.get(key)
        if not composite:
            return None
        
        try:
            signals = composite.calculate_all()
            phase3_signals = composite.get_phase3_signals()
            
            # Filter Phase 1 signals
            phase1_signals = {
                name: sig for name, sig in signals.items()
                if isinstance(sig, CompositeSignal) and not isinstance(sig, EnhancedSignal)
            }
            
            aggregator = self._get_or_create_aggregator(symbol, exchange)
            intelligence = aggregator.aggregate(
                symbol=symbol,
                signals=phase1_signals,
                phase3_signals=phase3_signals,
            )
            
            # Update cache
            self._intelligence_cache[key] = intelligence
            self._last_aggregation[key] = datetime.now(timezone.utc)
            
            return intelligence
            
        except Exception as e:
            logger.error(f"Error getting aggregated intelligence for {symbol}@{exchange}: {e}")
            return None
    
    def get_recommendation(
        self,
        symbol: str,
        exchange: str
    ) -> Optional[TradeRecommendation]:
        """
        Get current trade recommendation for a symbol (Phase 3).
        
        Convenience method that returns just the recommendation.
        """
        intelligence = self.get_aggregated_intelligence(symbol, exchange)
        if intelligence:
            return intelligence.recommendation
        return None
    
    def get_top_signals(
        self,
        symbol: str,
        exchange: str,
        limit: int = 5
    ) -> List[RankedSignal]:
        """
        Get top ranked signals for a symbol (Phase 3).
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            limit: Maximum number of signals to return
            
        Returns:
            List of top ranked signals
        """
        intelligence = self.get_aggregated_intelligence(symbol, exchange)
        if intelligence:
            return intelligence.top_signals[:limit]
        return []
    
    def get_signal_conflicts(
        self,
        symbol: str,
        exchange: str
    ) -> Optional[SignalConflict]:
        """
        Get any detected signal conflicts for a symbol (Phase 3).
        
        Returns:
            SignalConflict if conflicts detected, None otherwise
        """
        intelligence = self.get_aggregated_intelligence(symbol, exchange)
        if intelligence:
            return intelligence.conflict
        return None
    
    def get_feature_history(
        self,
        symbol: str,
        exchange: str,
        feature_type: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get historical features for a symbol."""
        return self.storage.query_features(
            symbol=symbol,
            exchange=exchange,
            feature_type=feature_type,
            limit=limit
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get feature engine metrics (Phase 3 enhanced)."""
        return {
            **{k: v for k, v in self._metrics.items() if k != 'processing_times'},
            'active_symbols': len(self._calculators),
            'composite_symbols': len(self._composite_calculators),
            'aggregator_symbols': len(self._signal_aggregators),
            'cached_intelligence': len(self._intelligence_cache),
            'pending_signal_batches': sum(len(b.signals) for b in self._signal_batches.values()),
            'pending_recommendation_batches': sum(len(b.recommendations) for b in self._signal_batches.values()),
        }
    
    def reset_symbol(self, symbol: str, exchange: str):
        """Reset calculators for a symbol (clear state)."""
        key = self._get_symbol_key(symbol, exchange)
        
        # Reset per-stream calculators (recreate them)
        if key in self._calculators:
            # Simply clear to force recreation on next use
            del self._calculators[key]
        
        if key in self._composite_calculators:
            self._composite_calculators[key].reset()
        
        if key in self._signal_aggregators:
            self._signal_aggregators[key].reset()
        
        # Clear caches
        if key in self._feature_caches:
            del self._feature_caches[key]
        if key in self._intelligence_cache:
            del self._intelligence_cache[key]
        if key in self._last_aggregation:
            del self._last_aggregation[key]
    
    def reset_all(self):
        """Reset all calculators."""
        # Clear all calculators (will be recreated on demand)
        self._calculators.clear()
        
        for composite in self._composite_calculators.values():
            composite.reset()
        
        for aggregator in self._signal_aggregators.values():
            aggregator.reset()
        
        # Clear all caches
        self._feature_caches.clear()
        self._intelligence_cache.clear()
        self._last_aggregation.clear()
        self._signal_batches.clear()


def create_feature_engine(db_manager: Any, **kwargs) -> FeatureEngine:
    """
    Factory function to create a configured FeatureEngine.
    
    Args:
        db_manager: DuckDB manager instance
        **kwargs: Additional configuration options
    
    Returns:
        Configured FeatureEngine instance
    """
    storage = InstitutionalFeatureStorage(db_manager)
    return FeatureEngine(db_manager, storage=storage, **kwargs)


async def integrate_with_collector(
    collector: Any,
    engine: FeatureEngine
):
    """
    Integrate FeatureEngine with IsolatedDataCollector.
    
    This sets up the callback connections.
    
    Args:
        collector: IsolatedDataCollector instance
        engine: FeatureEngine instance
    """
    # Register callbacks based on available methods
    if hasattr(collector, 'register_price_callback'):
        def price_callback(symbol: str, exchange: str, data: dict):
            engine.process_prices(symbol, exchange, data)
        collector.register_price_callback(price_callback)
    
    if hasattr(collector, 'register_trade_callback'):
        def trade_callback(symbol: str, exchange: str, data: dict):
            engine.process_trades(symbol, exchange, data)
        collector.register_trade_callback(trade_callback)
    
    if hasattr(collector, 'register_data_callback'):
        # Generic callback for all stream types
        def data_callback(symbol: str, exchange: str, stream_type: str, data: dict):
            engine.on_data_update(symbol, exchange, stream_type, data)
        collector.register_data_callback(data_callback)
    
    logger.info("FeatureEngine integrated with data collector")
