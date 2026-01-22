"""
🗃️ Institutional Feature Table Schemas
=======================================

Defines all DuckDB table schemas for institutional features.

Table Types:
1. Per-Stream Features: prices, orderbook, trades, funding, oi
2. Composite Signals: smart_money, squeeze, stop_hunt, regime
3. Aggregated Features: 1m, 5m, 15m timeframe aggregations

Naming Convention:
    {symbol}_{exchange}_{market_type}_features_{stream}
    Example: btcusdt_binance_futures_features_prices
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# PER-STREAM FEATURE TABLE SCHEMAS
# =============================================================================

FEATURE_TABLE_SCHEMAS = {
    
    # =========================================================================
    # PRICES FEATURES - Microprice, Spread Dynamics, Pressure
    # =========================================================================
    'prices': """
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                          INTEGER PRIMARY KEY,
            timestamp                   TIMESTAMP NOT NULL,
            
            -- Microprice Features
            microprice                  DOUBLE,
            microprice_deviation        DOUBLE,
            microprice_zscore           DOUBLE,
            
            -- Spread Features
            spread                      DOUBLE,
            spread_bps                  DOUBLE,
            spread_zscore               DOUBLE,
            spread_compression_velocity DOUBLE,
            spread_expansion_spike      DOUBLE,
            
            -- Pressure Features
            pressure_ratio              DOUBLE,
            bid_pressure                DOUBLE,
            ask_pressure                DOUBLE,
            
            -- Efficiency Features
            price_efficiency            DOUBLE,
            tick_reversal_rate          DOUBLE,
            price_vs_vwap               DOUBLE,
            mid_price_entropy           DOUBLE,
            
            -- Hurst Exponent (trend persistence)
            hurst_exponent              DOUBLE,
            
            -- Raw reference values
            mid_price                   DOUBLE,
            bid_price                   DOUBLE,
            ask_price                   DOUBLE,
            
            -- Metadata
            symbol                      VARCHAR(20),
            exchange                    VARCHAR(30),
            market_type                 VARCHAR(10)
        );
        
        CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp);
    """,
    
    # =========================================================================
    # ORDERBOOK FEATURES - Liquidity Structure, Dynamics, Heatmap
    # =========================================================================
    'orderbook': """
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                          INTEGER PRIMARY KEY,
            timestamp                   TIMESTAMP NOT NULL,
            
            -- Depth Imbalance (multi-level)
            depth_imbalance_5           DOUBLE,
            depth_imbalance_10          DOUBLE,
            cumulative_depth_imbalance  DOUBLE,
            
            -- Liquidity Structure
            liquidity_gradient          DOUBLE,
            liquidity_concentration_idx DOUBLE,
            vwap_depth                  DOUBLE,
            bid_depth_5                 DOUBLE,
            ask_depth_5                 DOUBLE,
            bid_depth_10                DOUBLE,
            ask_depth_10                DOUBLE,
            
            -- Orderbook Dynamics
            queue_position_drift        DOUBLE,
            add_cancel_ratio            DOUBLE,
            absorption_ratio            DOUBLE,
            replenishment_speed         DOUBLE,
            
            -- Liquidity Heatmap Metrics
            liquidity_persistence_score DOUBLE,
            liquidity_migration_velocity DOUBLE,
            pull_wall_detected          BOOLEAN,
            push_wall_detected          BOOLEAN,
            
            -- Support/Resistance Detection
            support_strength            DOUBLE,
            resistance_strength         DOUBLE,
            
            -- Metadata
            symbol                      VARCHAR(20),
            exchange                    VARCHAR(30),
            market_type                 VARCHAR(10)
        );
        
        CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp);
    """,
    
    # =========================================================================
    # TRADES FEATURES - Delta, CVD, Whale Detection, Market Impact
    # =========================================================================
    'trades': """
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                          INTEGER PRIMARY KEY,
            timestamp                   TIMESTAMP NOT NULL,
            
            -- Delta Features
            cvd                         DOUBLE,
            cvd_slope                   DOUBLE,
            cvd_acceleration            DOUBLE,
            aggressive_delta            DOUBLE,
            aggressive_delta_ratio      DOUBLE,
            
            -- Flow Features
            buy_volume                  DOUBLE,
            sell_volume                 DOUBLE,
            net_flow                    DOUBLE,
            bid_hit_volume              DOUBLE,
            ask_hit_volume              DOUBLE,
            bid_ask_hit_ratio           DOUBLE,
            
            -- Whale Detection
            whale_trade_detected        BOOLEAN,
            whale_trade_count           INTEGER,
            whale_trade_size_avg        DOUBLE,
            whale_trade_direction       VARCHAR(10),
            iceberg_probability         DOUBLE,
            
            -- Clustering & Impact
            trade_clustering_index      DOUBLE,
            trade_concentration         DOUBLE,
            sweep_detected              BOOLEAN,
            market_impact_per_volume    DOUBLE,
            slippage_estimator          DOUBLE,
            
            -- Metadata
            symbol                      VARCHAR(20),
            exchange                    VARCHAR(30),
            market_type                 VARCHAR(10)
        );
        
        CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp);
    """,
    
    # =========================================================================
    # FUNDING FEATURES - Rate Analysis, Skew, Momentum
    # =========================================================================
    'funding': """
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                          INTEGER PRIMARY KEY,
            timestamp                   TIMESTAMP NOT NULL,
            
            -- Rate Features
            funding_rate                DOUBLE,
            funding_rate_zscore         DOUBLE,
            funding_momentum            DOUBLE,
            funding_velocity            DOUBLE,
            
            -- Skew Features
            funding_skew_index          DOUBLE,
            funding_stress_index        DOUBLE,
            funding_anomaly_score       DOUBLE,
            
            -- Predictive Features
            next_funding_prediction     DOUBLE,
            funding_reversal_probability DOUBLE,
            
            -- Carry & Arbitrage
            funding_carry_yield         DOUBLE,
            annualized_funding          DOUBLE,
            
            -- Seasonality
            funding_hour_of_day_bias    DOUBLE,
            funding_day_of_week_bias    DOUBLE,
            
            -- Metadata
            symbol                      VARCHAR(20),
            exchange                    VARCHAR(30),
            market_type                 VARCHAR(10)
        );
        
        CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp);
    """,
    
    # =========================================================================
    # OPEN INTEREST FEATURES - OI Delta, Leverage, Liquidation Risk
    # =========================================================================
    'oi': """
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                          INTEGER PRIMARY KEY,
            timestamp                   TIMESTAMP NOT NULL,
            
            -- OI Delta Features
            oi                          DOUBLE,
            oi_delta                    DOUBLE,
            oi_delta_pct                DOUBLE,
            oi_velocity                 DOUBLE,
            oi_acceleration             DOUBLE,
            
            -- OI × Price Analysis
            oi_price_delta_product      DOUBLE,
            oi_price_correlation        DOUBLE,
            oi_volume_correlation       DOUBLE,
            oi_divergence_score         DOUBLE,
            
            -- Leverage Features
            leverage_index              DOUBLE,
            leverage_expansion_rate     DOUBLE,
            leverage_stress_index       DOUBLE,
            
            -- Liquidation Risk
            liquidation_cascade_risk    DOUBLE,
            oi_velocity_spike           BOOLEAN,
            position_intent             VARCHAR(20),
            position_intent_score       DOUBLE,
            
            -- Absorption
            oi_absorption_ratio         DOUBLE,
            
            -- Metadata
            symbol                      VARCHAR(20),
            exchange                    VARCHAR(30),
            market_type                 VARCHAR(10)
        );
        
        CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp);
    """,
    
    # =========================================================================
    # LIQUIDATIONS FEATURES - Clusters, Cascades, Exhaustion
    # =========================================================================
    'liquidations': """
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                          INTEGER PRIMARY KEY,
            timestamp                   TIMESTAMP NOT NULL,
            
            -- Liquidation Counts
            long_liquidation_count      INTEGER,
            short_liquidation_count     INTEGER,
            total_liquidation_count     INTEGER,
            
            -- Liquidation Values
            long_liquidation_value      DOUBLE,
            short_liquidation_value     DOUBLE,
            total_liquidation_value     DOUBLE,
            liquidation_imbalance       DOUBLE,
            
            -- Clustering
            liquidation_cluster_detected BOOLEAN,
            cluster_size                INTEGER,
            cluster_value               DOUBLE,
            cluster_duration_seconds    DOUBLE,
            
            -- Cascade Detection
            cascade_acceleration        DOUBLE,
            cascade_probability         DOUBLE,
            cascade_severity            VARCHAR(10),
            
            -- Absorption & Exhaustion
            absorption_after_liq        DOUBLE,
            exhaustion_signal           BOOLEAN,
            
            -- Metadata
            symbol                      VARCHAR(20),
            exchange                    VARCHAR(30),
            market_type                 VARCHAR(10)
        );
        
        CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp);
    """,
    
    # =========================================================================
    # MARK PRICES FEATURES - Basis, Dislocation, Funding Pressure
    # =========================================================================
    'mark_prices': """
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                          INTEGER PRIMARY KEY,
            timestamp                   TIMESTAMP NOT NULL,
            
            -- Basis Features
            mark_spot_basis             DOUBLE,
            mark_spot_basis_pct         DOUBLE,
            basis_zscore                DOUBLE,
            basis_mean_reversion_speed  DOUBLE,
            
            -- Index Divergence
            index_divergence            DOUBLE,
            index_divergence_risk       DOUBLE,
            
            -- Mark Price Movement
            mark_price_velocity         DOUBLE,
            mark_price_vs_mid           DOUBLE,
            
            -- Raw values
            mark_price                  DOUBLE,
            index_price                 DOUBLE,
            
            -- Metadata
            symbol                      VARCHAR(20),
            exchange                    VARCHAR(30),
            market_type                 VARCHAR(10)
        );
        
        CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp);
    """,
    
    # =========================================================================
    # TICKER 24H FEATURES - Volume, Range, Volatility
    # =========================================================================
    'ticker': """
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                          INTEGER PRIMARY KEY,
            timestamp                   TIMESTAMP NOT NULL,
            
            -- Volume Features
            volume_acceleration         DOUBLE,
            relative_volume_percentile  DOUBLE,
            volume_profile_skew         DOUBLE,
            
            -- Range Features
            range_expansion_ratio       DOUBLE,
            high_low_range              DOUBLE,
            range_vs_atr                DOUBLE,
            
            -- Volatility Features
            volatility_compression_idx  DOUBLE,
            volatility_expansion_idx    DOUBLE,
            
            -- Price Change
            price_change_pct            DOUBLE,
            price_change_zscore         DOUBLE,
            
            -- Participation
            trade_count_percentile      DOUBLE,
            institutional_interest_idx  DOUBLE,
            
            -- Metadata
            symbol                      VARCHAR(20),
            exchange                    VARCHAR(30),
            market_type                 VARCHAR(10)
        );
        
        CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp);
    """,
}


# =============================================================================
# COMPOSITE SIGNAL TABLE SCHEMAS
# =============================================================================

COMPOSITE_TABLE_SCHEMAS = {
    
    # Cross-stream composite signals
    'composite_signals': """
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                          INTEGER PRIMARY KEY,
            timestamp                   TIMESTAMP NOT NULL,
            
            -- Smart Money Signals
            smart_accumulation_score    DOUBLE,
            smart_distribution_score    DOUBLE,
            institutional_activity_idx  DOUBLE,
            smart_money_direction       VARCHAR(10),
            
            -- Squeeze Signals
            short_squeeze_probability   DOUBLE,
            long_squeeze_probability    DOUBLE,
            squeeze_severity            VARCHAR(10),
            squeeze_trigger_price       DOUBLE,
            squeeze_target_price        DOUBLE,
            
            -- Stop Hunt Signals
            stop_hunt_probability       DOUBLE,
            stop_hunt_type              VARCHAR(20),
            stop_hunt_direction         VARCHAR(10),
            manipulation_confidence     DOUBLE,
            
            -- Regime Signals
            market_regime               VARCHAR(20),
            regime_confidence           DOUBLE,
            regime_change_probability   DOUBLE,
            
            -- Stress Indices
            liquidity_stress_index      DOUBLE,
            leverage_stress_index       DOUBLE,
            cascade_risk_score          DOUBLE,
            
            -- Cross-Stream Divergence
            orderbook_trade_divergence  DOUBLE,
            oi_price_divergence         DOUBLE,
            funding_oi_divergence       DOUBLE,
            
            -- Metadata
            symbol                      VARCHAR(20),
            exchange                    VARCHAR(30),
            market_type                 VARCHAR(10)
        );
        
        CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp);
    """,
}


# =============================================================================
# AGGREGATED FEATURE TABLE SCHEMAS (for candle timeframes)
# =============================================================================

AGGREGATED_TABLE_SCHEMAS = {
    
    # 1-minute aggregated features
    'features_1m': """
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                          INTEGER PRIMARY KEY,
            timestamp                   TIMESTAMP NOT NULL,
            
            -- Aggregated from prices_features
            microprice_avg              DOUBLE,
            microprice_deviation_avg    DOUBLE,
            spread_avg                  DOUBLE,
            spread_zscore_avg           DOUBLE,
            pressure_ratio_avg          DOUBLE,
            
            -- Aggregated from orderbook_features
            depth_imbalance_avg         DOUBLE,
            liquidity_gradient_avg      DOUBLE,
            absorption_ratio_avg        DOUBLE,
            
            -- Aggregated from trades_features
            cvd_sum                     DOUBLE,
            aggressive_delta_sum        DOUBLE,
            whale_trades_count          INTEGER,
            net_flow_sum                DOUBLE,
            
            -- Aggregated from oi_features
            oi_delta_sum                DOUBLE,
            leverage_index_avg          DOUBLE,
            
            -- Aggregated from funding_features
            funding_rate_avg            DOUBLE,
            funding_stress_avg          DOUBLE,
            
            -- Aggregated from liquidations_features
            liquidation_value_sum       DOUBLE,
            cascade_probability_max     DOUBLE,
            
            -- Composite signals (latest in window)
            smart_accumulation_score    DOUBLE,
            short_squeeze_probability   DOUBLE,
            stop_hunt_probability       DOUBLE,
            market_regime               VARCHAR(20),
            
            -- Metadata
            symbol                      VARCHAR(20),
            exchange                    VARCHAR(30),
            market_type                 VARCHAR(10)
        );
        
        CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp);
    """,
    
    # 5-minute aggregated features
    'features_5m': """
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                          INTEGER PRIMARY KEY,
            timestamp                   TIMESTAMP NOT NULL,
            
            -- Same structure as 1m but aggregated over 5 minutes
            microprice_avg              DOUBLE,
            microprice_deviation_avg    DOUBLE,
            spread_avg                  DOUBLE,
            spread_zscore_avg           DOUBLE,
            pressure_ratio_avg          DOUBLE,
            
            depth_imbalance_avg         DOUBLE,
            liquidity_gradient_avg      DOUBLE,
            absorption_ratio_avg        DOUBLE,
            
            cvd_sum                     DOUBLE,
            aggressive_delta_sum        DOUBLE,
            whale_trades_count          INTEGER,
            net_flow_sum                DOUBLE,
            
            oi_delta_sum                DOUBLE,
            leverage_index_avg          DOUBLE,
            
            funding_rate_avg            DOUBLE,
            funding_stress_avg          DOUBLE,
            
            liquidation_value_sum       DOUBLE,
            cascade_probability_max     DOUBLE,
            
            smart_accumulation_score    DOUBLE,
            short_squeeze_probability   DOUBLE,
            stop_hunt_probability       DOUBLE,
            market_regime               VARCHAR(20),
            
            symbol                      VARCHAR(20),
            exchange                    VARCHAR(30),
            market_type                 VARCHAR(10)
        );
        
        CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp);
    """,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_feature_table_name(
    symbol: str,
    exchange: str,
    market_type: str,
    feature_type: str
) -> str:
    """
    Generate feature table name.
    
    Args:
        symbol: e.g., "btcusdt"
        exchange: e.g., "binance"
        market_type: "futures" or "spot"
        feature_type: "prices", "orderbook", "trades", etc.
    
    Returns:
        Table name, e.g., "btcusdt_binance_futures_features_prices"
    """
    return f"{symbol.lower()}_{exchange.lower()}_{market_type.lower()}_features_{feature_type}"


def get_composite_table_name(
    symbol: str,
    exchange: str,
    market_type: str
) -> str:
    """
    Generate composite signal table name.
    
    Returns:
        Table name, e.g., "btcusdt_binance_futures_composite_signals"
    """
    return f"{symbol.lower()}_{exchange.lower()}_{market_type.lower()}_composite_signals"


def get_aggregated_table_name(
    symbol: str,
    exchange: str,
    market_type: str,
    timeframe: str
) -> str:
    """
    Generate aggregated feature table name.
    
    Args:
        timeframe: "1m", "5m", "15m", etc.
    
    Returns:
        Table name, e.g., "btcusdt_binance_futures_features_1m"
    """
    return f"{symbol.lower()}_{exchange.lower()}_{market_type.lower()}_features_{timeframe}"


def create_feature_tables_for_symbol(
    conn,
    symbol: str,
    exchange: str,
    market_type: str,
    include_composite: bool = True,
    include_aggregated: bool = True
) -> List[str]:
    """
    Create all feature tables for a symbol/exchange pair.
    
    Args:
        conn: DuckDB connection
        symbol: Trading pair
        exchange: Exchange name
        market_type: "futures" or "spot"
        include_composite: Create composite signal tables
        include_aggregated: Create aggregated feature tables
    
    Returns:
        List of created table names
    """
    created_tables = []
    
    # Create per-stream feature tables
    for feature_type, schema_template in FEATURE_TABLE_SCHEMAS.items():
        table_name = get_feature_table_name(symbol, exchange, market_type, feature_type)
        schema = schema_template.format(table_name=table_name)
        
        try:
            conn.execute(schema)
            created_tables.append(table_name)
            logger.debug(f"✓ Created feature table: {table_name}")
        except Exception as e:
            logger.error(f"✗ Failed to create {table_name}: {e}")
    
    # Create composite signal tables
    if include_composite:
        for signal_type, schema_template in COMPOSITE_TABLE_SCHEMAS.items():
            table_name = get_composite_table_name(symbol, exchange, market_type)
            schema = schema_template.format(table_name=table_name)
            
            try:
                conn.execute(schema)
                created_tables.append(table_name)
                logger.debug(f"✓ Created composite table: {table_name}")
            except Exception as e:
                logger.error(f"✗ Failed to create {table_name}: {e}")
    
    # Create aggregated feature tables
    if include_aggregated:
        for agg_type, schema_template in AGGREGATED_TABLE_SCHEMAS.items():
            timeframe = agg_type.replace('features_', '')
            table_name = get_aggregated_table_name(symbol, exchange, market_type, timeframe)
            schema = schema_template.format(table_name=table_name)
            
            try:
                conn.execute(schema)
                created_tables.append(table_name)
                logger.debug(f"✓ Created aggregated table: {table_name}")
            except Exception as e:
                logger.error(f"✗ Failed to create {table_name}: {e}")
    
    return created_tables


def get_all_feature_types() -> List[str]:
    """Get list of all feature types."""
    return list(FEATURE_TABLE_SCHEMAS.keys())


def get_feature_columns(feature_type: str) -> List[str]:
    """
    Get list of feature columns for a feature type.
    
    Extracts column names from the schema (excludes id, timestamp, metadata).
    """
    if feature_type not in FEATURE_TABLE_SCHEMAS:
        return []
    
    schema = FEATURE_TABLE_SCHEMAS[feature_type]
    
    # Parse columns from schema (simple extraction)
    columns = []
    for line in schema.split('\n'):
        line = line.strip()
        if line and not line.startswith('--') and not line.startswith('CREATE') and not line.startswith(');'):
            if 'PRIMARY KEY' in line or 'INDEX' in line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                col_name = parts[0].rstrip(',')
                if col_name not in ['id', 'timestamp', 'symbol', 'exchange', 'market_type']:
                    columns.append(col_name)
    
    return columns
