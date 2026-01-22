"""
💾 Institutional Feature Storage Manager
========================================

Handles all feature storage operations for institutional features.

Features:
- Create feature tables on-demand
- Batch insert features for performance
- Query features by time range
- Aggregate features to candle timeframes
- Export features for backtesting
"""

import duckdb
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from pathlib import Path
import asyncio

from .schemas import (
    FEATURE_TABLE_SCHEMAS,
    COMPOSITE_TABLE_SCHEMAS,
    AGGREGATED_TABLE_SCHEMAS,
    get_feature_table_name,
    get_composite_table_name,
    get_aggregated_table_name,
    create_feature_tables_for_symbol,
)

logger = logging.getLogger(__name__)


class InstitutionalFeatureStorage:
    """
    Manages storage and retrieval of institutional features.
    
    Features:
    - Automatic table creation
    - Buffered batch insertion (5-second windows)
    - Time-range queries
    - Feature aggregation to candle timeframes
    """
    
    def __init__(
        self,
        db_path: str = "data/institutional_features.duckdb",
        buffer_flush_interval: float = 5.0,
        max_buffer_size: int = 1000
    ):
        self.db_path = db_path
        self.buffer_flush_interval = buffer_flush_interval
        self.max_buffer_size = max_buffer_size
        
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self._lock = asyncio.Lock()
        
        # Feature buffers: table_name -> List[records]
        self.buffers: Dict[str, List[Tuple]] = defaultdict(list)
        
        # ID counters for each table
        self.id_counters: Dict[str, int] = defaultdict(int)
        
        # Track existing tables
        self.existing_tables: set = set()
        
        # Stats
        self.stats = {
            'total_features_stored': 0,
            'features_by_type': defaultdict(int),
            'last_flush': datetime.now(timezone.utc),
            'flush_count': 0,
            'errors': 0,
        }
    
    def connect(self) -> None:
        """Connect to DuckDB and initialize tables."""
        # Ensure data directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = duckdb.connect(self.db_path)
        
        # Load existing tables
        try:
            tables = self.conn.execute("SHOW TABLES").fetchall()
            self.existing_tables = {t[0] for t in tables}
            logger.info(f"✅ Connected to feature storage: {self.db_path}")
            logger.info(f"   Found {len(self.existing_tables)} existing feature tables")
        except Exception as e:
            logger.warning(f"Could not list tables: {e}")
            self.existing_tables = set()
        
        # Load ID counters for existing tables
        for table_name in self.existing_tables:
            try:
                result = self.conn.execute(f"SELECT MAX(id) FROM {table_name}").fetchone()
                max_id = result[0] if result and result[0] is not None else 0
                self.id_counters[table_name] = max_id
            except Exception:
                self.id_counters[table_name] = 0
    
    def ensure_connected(self) -> None:
        """Ensure database connection is active."""
        if self.conn is None:
            self.connect()
    
    def close(self) -> None:
        """Close database connection after flushing buffers."""
        if self.conn:
            # Flush any remaining buffers
            self._flush_all_buffers()
            self.conn.close()
            self.conn = None
            logger.info("🔌 Disconnected from feature storage")
    
    # =========================================================================
    # TABLE MANAGEMENT
    # =========================================================================
    
    def ensure_feature_tables(
        self,
        symbol: str,
        exchange: str,
        market_type: str
    ) -> List[str]:
        """
        Ensure all feature tables exist for a symbol/exchange pair.
        
        Creates tables if they don't exist.
        
        Returns:
            List of table names (existing or newly created)
        """
        self.ensure_connected()
        
        created = create_feature_tables_for_symbol(
            self.conn,
            symbol,
            exchange,
            market_type
        )
        
        # Update existing tables set
        self.existing_tables.update(created)
        
        return created
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        return table_name in self.existing_tables
    
    # =========================================================================
    # FEATURE INSERTION
    # =========================================================================
    
    def _get_next_id(self, table_name: str) -> int:
        """Get next ID for a table."""
        self.id_counters[table_name] += 1
        return self.id_counters[table_name]
    
    async def store_features(
        self,
        symbol: str,
        exchange: str,
        market_type: str,
        feature_type: str,
        features: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Store features to buffer (will be flushed periodically).
        
        Args:
            symbol: Trading pair
            exchange: Exchange name
            market_type: "futures" or "spot"
            feature_type: "prices", "orderbook", "trades", etc.
            features: Dict of feature_name: value
            timestamp: Feature timestamp (default: now)
        
        Returns:
            True if buffered successfully
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        table_name = get_feature_table_name(symbol, exchange, market_type, feature_type)
        
        # Ensure table exists
        if not self.table_exists(table_name):
            self.ensure_feature_tables(symbol, exchange, market_type)
        
        async with self._lock:
            record_id = self._get_next_id(table_name)
            
            # Build record tuple
            # Order must match schema columns
            record = self._build_feature_record(
                record_id,
                timestamp,
                feature_type,
                features,
                symbol,
                exchange,
                market_type
            )
            
            self.buffers[table_name].append(record)
            self.stats['total_features_stored'] += 1
            self.stats['features_by_type'][feature_type] += 1
            
            # Flush if buffer is full
            if len(self.buffers[table_name]) >= self.max_buffer_size:
                self._flush_buffer(table_name)
        
        return True
    
    def _build_feature_record(
        self,
        record_id: int,
        timestamp: datetime,
        feature_type: str,
        features: Dict[str, Any],
        symbol: str,
        exchange: str,
        market_type: str
    ) -> Tuple:
        """
        Build a record tuple from features dict.
        
        Returns:
            Tuple matching the table schema column order
        """
        # This is a simplified approach - in production, you'd want to
        # dynamically match schema columns
        
        base = [record_id, timestamp]
        
        # Add features in schema order (feature-type specific)
        if feature_type == 'prices':
            base.extend([
                features.get('microprice'),
                features.get('microprice_deviation'),
                features.get('microprice_zscore'),
                features.get('spread'),
                features.get('spread_bps'),
                features.get('spread_zscore'),
                features.get('spread_compression_velocity'),
                features.get('spread_expansion_spike'),
                features.get('pressure_ratio'),
                features.get('bid_pressure'),
                features.get('ask_pressure'),
                features.get('price_efficiency'),
                features.get('tick_reversal_rate'),
                features.get('price_vs_vwap'),
                features.get('mid_price_entropy'),
                features.get('hurst_exponent'),
                features.get('mid_price'),
                features.get('bid_price'),
                features.get('ask_price'),
            ])
        elif feature_type == 'orderbook':
            base.extend([
                features.get('depth_imbalance_5'),
                features.get('depth_imbalance_10'),
                features.get('cumulative_depth_imbalance'),
                features.get('liquidity_gradient'),
                features.get('liquidity_concentration_idx'),
                features.get('vwap_depth'),
                features.get('bid_depth_5'),
                features.get('ask_depth_5'),
                features.get('bid_depth_10'),
                features.get('ask_depth_10'),
                features.get('queue_position_drift'),
                features.get('add_cancel_ratio'),
                features.get('absorption_ratio'),
                features.get('replenishment_speed'),
                features.get('liquidity_persistence_score'),
                features.get('liquidity_migration_velocity'),
                features.get('pull_wall_detected'),
                features.get('push_wall_detected'),
                features.get('support_strength'),
                features.get('resistance_strength'),
            ])
        elif feature_type == 'trades':
            base.extend([
                features.get('cvd'),
                features.get('cvd_slope'),
                features.get('cvd_acceleration'),
                features.get('aggressive_delta'),
                features.get('aggressive_delta_ratio'),
                features.get('buy_volume'),
                features.get('sell_volume'),
                features.get('net_flow'),
                features.get('bid_hit_volume'),
                features.get('ask_hit_volume'),
                features.get('bid_ask_hit_ratio'),
                features.get('whale_trade_detected'),
                features.get('whale_trade_count'),
                features.get('whale_trade_size_avg'),
                features.get('whale_trade_direction'),
                features.get('iceberg_probability'),
                features.get('trade_clustering_index'),
                features.get('trade_concentration'),
                features.get('sweep_detected'),
                features.get('market_impact_per_volume'),
                features.get('slippage_estimator'),
            ])
        elif feature_type == 'funding':
            base.extend([
                features.get('funding_rate'),
                features.get('funding_rate_zscore'),
                features.get('funding_momentum'),
                features.get('funding_velocity'),
                features.get('funding_skew_index'),
                features.get('funding_stress_index'),
                features.get('funding_anomaly_score'),
                features.get('next_funding_prediction'),
                features.get('funding_reversal_probability'),
                features.get('funding_carry_yield'),
                features.get('annualized_funding'),
                features.get('funding_hour_of_day_bias'),
                features.get('funding_day_of_week_bias'),
            ])
        elif feature_type == 'oi':
            base.extend([
                features.get('oi'),
                features.get('oi_delta'),
                features.get('oi_delta_pct'),
                features.get('oi_velocity'),
                features.get('oi_acceleration'),
                features.get('oi_price_delta_product'),
                features.get('oi_price_correlation'),
                features.get('oi_volume_correlation'),
                features.get('oi_divergence_score'),
                features.get('leverage_index'),
                features.get('leverage_expansion_rate'),
                features.get('leverage_stress_index'),
                features.get('liquidation_cascade_risk'),
                features.get('oi_velocity_spike'),
                features.get('position_intent'),
                features.get('position_intent_score'),
                features.get('oi_absorption_ratio'),
            ])
        elif feature_type == 'liquidations':
            base.extend([
                features.get('long_liquidation_count'),
                features.get('short_liquidation_count'),
                features.get('total_liquidation_count'),
                features.get('long_liquidation_value'),
                features.get('short_liquidation_value'),
                features.get('total_liquidation_value'),
                features.get('liquidation_imbalance'),
                features.get('liquidation_cluster_detected'),
                features.get('cluster_size'),
                features.get('cluster_value'),
                features.get('cluster_duration_seconds'),
                features.get('cascade_acceleration'),
                features.get('cascade_probability'),
                features.get('cascade_severity'),
                features.get('absorption_after_liq'),
                features.get('exhaustion_signal'),
            ])
        elif feature_type == 'mark_prices':
            base.extend([
                features.get('mark_spot_basis'),
                features.get('mark_spot_basis_pct'),
                features.get('basis_zscore'),
                features.get('basis_mean_reversion_speed'),
                features.get('index_divergence'),
                features.get('index_divergence_risk'),
                features.get('mark_price_velocity'),
                features.get('mark_price_vs_mid'),
                features.get('mark_price'),
                features.get('index_price'),
            ])
        elif feature_type == 'ticker':
            base.extend([
                features.get('volume_acceleration'),
                features.get('relative_volume_percentile'),
                features.get('volume_profile_skew'),
                features.get('range_expansion_ratio'),
                features.get('high_low_range'),
                features.get('range_vs_atr'),
                features.get('volatility_compression_idx'),
                features.get('volatility_expansion_idx'),
                features.get('price_change_pct'),
                features.get('price_change_zscore'),
                features.get('trade_count_percentile'),
                features.get('institutional_interest_idx'),
            ])
        
        # Add metadata
        base.extend([symbol, exchange, market_type])
        
        return tuple(base)
    
    def _flush_buffer(self, table_name: str) -> None:
        """Flush a single buffer to database."""
        if not self.buffers[table_name]:
            return
        
        records = self.buffers[table_name]
        self.buffers[table_name] = []
        
        try:
            # Build INSERT statement
            placeholders = ', '.join(['?' for _ in records[0]])
            query = f"INSERT INTO {table_name} VALUES ({placeholders})"
            
            # Batch insert
            self.conn.executemany(query, records)
            
            self.stats['flush_count'] += 1
            self.stats['last_flush'] = datetime.now(timezone.utc)
            
            logger.debug(f"Flushed {len(records)} records to {table_name}")
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error flushing {table_name}: {e}")
            # Put records back in buffer for retry
            self.buffers[table_name].extend(records)
    
    def _flush_all_buffers(self) -> None:
        """Flush all buffers to database."""
        for table_name in list(self.buffers.keys()):
            self._flush_buffer(table_name)
    
    async def flush_buffers(self) -> None:
        """Async flush all buffers."""
        async with self._lock:
            self._flush_all_buffers()
    
    # =========================================================================
    # COMPOSITE SIGNAL STORAGE
    # =========================================================================
    
    async def store_composite_signals(
        self,
        symbol: str,
        exchange: str,
        market_type: str,
        signals: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Store composite signals.
        
        Args:
            signals: Dict containing all composite signal values
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        table_name = get_composite_table_name(symbol, exchange, market_type)
        
        # Ensure table exists
        if not self.table_exists(table_name):
            self.ensure_feature_tables(symbol, exchange, market_type)
        
        async with self._lock:
            record_id = self._get_next_id(table_name)
            
            record = (
                record_id,
                timestamp,
                # Smart Money
                signals.get('smart_accumulation_score'),
                signals.get('smart_distribution_score'),
                signals.get('institutional_activity_idx'),
                signals.get('smart_money_direction'),
                # Squeeze
                signals.get('short_squeeze_probability'),
                signals.get('long_squeeze_probability'),
                signals.get('squeeze_severity'),
                signals.get('squeeze_trigger_price'),
                signals.get('squeeze_target_price'),
                # Stop Hunt
                signals.get('stop_hunt_probability'),
                signals.get('stop_hunt_type'),
                signals.get('stop_hunt_direction'),
                signals.get('manipulation_confidence'),
                # Regime
                signals.get('market_regime'),
                signals.get('regime_confidence'),
                signals.get('regime_change_probability'),
                # Stress
                signals.get('liquidity_stress_index'),
                signals.get('leverage_stress_index'),
                signals.get('cascade_risk_score'),
                # Divergence
                signals.get('orderbook_trade_divergence'),
                signals.get('oi_price_divergence'),
                signals.get('funding_oi_divergence'),
                # Metadata
                symbol,
                exchange,
                market_type,
            )
            
            self.buffers[table_name].append(record)
            
            if len(self.buffers[table_name]) >= self.max_buffer_size:
                self._flush_buffer(table_name)
        
        return True
    
    # =========================================================================
    # FEATURE QUERIES
    # =========================================================================
    
    def query_features(
        self,
        symbol: str,
        exchange: str,
        market_type: str,
        feature_type: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        columns: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Query features by time range.
        
        Returns:
            List of feature dicts
        """
        self.ensure_connected()
        
        table_name = get_feature_table_name(symbol, exchange, market_type, feature_type)
        
        if not self.table_exists(table_name):
            return []
        
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        
        if columns:
            cols = ', '.join(columns)
        else:
            cols = '*'
        
        query = f"""
            SELECT {cols}
            FROM {table_name}
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        try:
            result = self.conn.execute(query, [start_time, end_time, limit]).fetchall()
            
            # Get column names
            col_names = [desc[0] for desc in self.conn.description]
            
            return [dict(zip(col_names, row)) for row in result]
            
        except Exception as e:
            logger.error(f"Query error for {table_name}: {e}")
            return []
    
    def get_latest_features(
        self,
        symbol: str,
        exchange: str,
        market_type: str,
        feature_type: str
    ) -> Optional[Dict]:
        """
        Get most recent features.
        
        Returns:
            Feature dict or None
        """
        self.ensure_connected()
        
        table_name = get_feature_table_name(symbol, exchange, market_type, feature_type)
        
        if not self.table_exists(table_name):
            return None
        
        query = f"""
            SELECT *
            FROM {table_name}
            ORDER BY timestamp DESC
            LIMIT 1
        """
        
        try:
            result = self.conn.execute(query).fetchone()
            
            if result:
                col_names = [desc[0] for desc in self.conn.description]
                return dict(zip(col_names, result))
            
            return None
            
        except Exception as e:
            logger.error(f"Query error for {table_name}: {e}")
            return None
    
    def get_latest_composite_signals(
        self,
        symbol: str,
        exchange: str,
        market_type: str
    ) -> Optional[Dict]:
        """
        Get most recent composite signals.
        """
        self.ensure_connected()
        
        table_name = get_composite_table_name(symbol, exchange, market_type)
        
        if not self.table_exists(table_name):
            return None
        
        query = f"""
            SELECT *
            FROM {table_name}
            ORDER BY timestamp DESC
            LIMIT 1
        """
        
        try:
            result = self.conn.execute(query).fetchone()
            
            if result:
                col_names = [desc[0] for desc in self.conn.description]
                return dict(zip(col_names, result))
            
            return None
            
        except Exception as e:
            logger.error(f"Query error for {table_name}: {e}")
            return None
    
    # =========================================================================
    # AGGREGATION
    # =========================================================================
    
    def aggregate_features_to_timeframe(
        self,
        symbol: str,
        exchange: str,
        market_type: str,
        timeframe: str = "1m",
        lookback_minutes: int = 60
    ) -> int:
        """
        Aggregate real-time features to candle timeframes.
        
        Args:
            timeframe: "1m", "5m", "15m"
            lookback_minutes: How far back to aggregate
        
        Returns:
            Number of aggregated records created
        """
        self.ensure_connected()
        
        interval_map = {
            "1m": "1 minute",
            "5m": "5 minutes",
            "15m": "15 minutes",
            "1h": "1 hour",
        }
        
        if timeframe not in interval_map:
            logger.error(f"Unknown timeframe: {timeframe}")
            return 0
        
        interval = interval_map[timeframe]
        target_table = get_aggregated_table_name(symbol, exchange, market_type, timeframe)
        
        # Ensure target table exists
        if not self.table_exists(target_table):
            self.ensure_feature_tables(symbol, exchange, market_type)
        
        # This would be a complex aggregation query joining multiple feature tables
        # For now, return 0 - full implementation would be added later
        logger.info(f"Aggregation to {timeframe} not yet implemented")
        return 0
    
    # =========================================================================
    # STATS & MONITORING
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Get storage statistics."""
        return {
            **self.stats,
            'features_by_type': dict(self.stats['features_by_type']),
            'buffer_sizes': {k: len(v) for k, v in self.buffers.items()},
            'table_count': len(self.existing_tables),
        }
    
    def get_table_row_counts(self) -> Dict[str, int]:
        """Get row counts for all feature tables."""
        self.ensure_connected()
        
        counts = {}
        for table_name in self.existing_tables:
            try:
                result = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                counts[table_name] = result[0] if result else 0
            except Exception:
                counts[table_name] = -1
        
        return counts
