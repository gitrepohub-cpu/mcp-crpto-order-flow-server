"""Crypto arbitrage and exchange analysis tools"""

from .crypto_arbitrage_tool import (
    analyze_crypto_arbitrage,
    get_exchange_prices,
    get_spread_matrix,
    get_recent_opportunities,
    arbitrage_scanner_health
)

from .binance_futures_tools import (
    # Market Data
    binance_get_ticker,
    binance_get_prices,
    binance_get_orderbook,
    binance_get_trades,
    binance_get_klines,
    # Derivatives Data
    binance_get_open_interest,
    binance_get_open_interest_history,
    binance_get_funding_rate,
    binance_get_premium_index,
    # Positioning Data
    binance_get_long_short_ratio,
    binance_get_taker_volume,
    # Basis Data
    binance_get_basis,
    # Liquidations
    binance_get_liquidations,
    # Comprehensive
    binance_market_snapshot,
    binance_full_analysis,
)

from .bybit_tools import (
    # Spot Market Tools
    bybit_spot_ticker,
    bybit_spot_orderbook,
    bybit_spot_trades,
    bybit_spot_klines,
    bybit_all_spot_tickers,
    # Futures/Linear Tools
    bybit_futures_ticker,
    bybit_futures_orderbook,
    bybit_open_interest,
    bybit_funding_rate,
    bybit_long_short_ratio,
    bybit_historical_volatility,
    bybit_insurance_fund,
    bybit_all_perpetual_tickers,
    # Analysis Tools
    bybit_derivatives_analysis,
    bybit_market_snapshot,
    bybit_instruments_info,
    bybit_options_overview,
    bybit_risk_limit,
    bybit_announcements,
    bybit_full_market_analysis,
)

from .binance_spot_tools import (
    # Market Data
    binance_spot_ticker,
    binance_spot_price,
    binance_spot_orderbook,
    binance_spot_trades,
    binance_spot_klines,
    binance_spot_avg_price,
    binance_spot_book_ticker,
    binance_spot_agg_trades,
    binance_spot_exchange_info,
    binance_spot_rolling_ticker,
    binance_spot_all_tickers,
    # Analysis
    binance_spot_snapshot,
    binance_spot_full_analysis,
)

from .okx_tools import (
    # Ticker Tools
    okx_ticker,
    okx_all_tickers,
    okx_index_ticker,
    # Orderbook Tools
    okx_orderbook,
    # Trades Tools
    okx_trades,
    # Klines Tools
    okx_klines,
    # Funding Rate Tools
    okx_funding_rate,
    okx_funding_rate_history,
    # Open Interest Tools
    okx_open_interest,
    okx_oi_volume,
    # Long/Short & Taker Volume
    okx_long_short_ratio,
    okx_taker_volume,
    # Instrument Info
    okx_instruments,
    okx_mark_price,
    # Insurance & Platform
    okx_insurance_fund,
    okx_platform_volume,
    # Options
    okx_options_summary,
    # Analysis
    okx_market_snapshot,
    okx_full_analysis,
    okx_top_movers,
)

from .kraken_tools import (
    # Spot Tools
    kraken_spot_ticker,
    kraken_all_spot_tickers,
    kraken_spot_orderbook,
    kraken_spot_trades,
    kraken_spot_klines,
    kraken_spread,
    kraken_assets,
    kraken_spot_pairs,
    # Futures Tools
    kraken_futures_ticker,
    kraken_all_futures_tickers,
    kraken_futures_orderbook,
    kraken_futures_trades,
    kraken_futures_klines,
    kraken_futures_instruments,
    # Funding & OI
    kraken_funding_rates,
    kraken_open_interest,
    # System
    kraken_system_status,
    # Analysis
    kraken_top_movers,
    kraken_market_snapshot,
    kraken_full_analysis,
)

from .gateio_tools import (
    # Perpetual Futures Tools
    gateio_futures_contracts_tool,
    gateio_futures_contract_tool,
    gateio_futures_ticker_tool,
    gateio_all_futures_tickers_tool,
    gateio_futures_orderbook_tool,
    gateio_futures_trades_tool,
    gateio_futures_klines_tool,
    gateio_funding_rate_tool,
    gateio_all_funding_rates_tool,
    gateio_contract_stats_tool,
    gateio_open_interest_tool,
    gateio_liquidations_tool,
    gateio_insurance_fund_tool,
    gateio_risk_limit_tiers_tool,
    # Delivery Futures Tools
    gateio_delivery_contracts_tool,
    gateio_delivery_ticker_tool,
    # Options Tools
    gateio_options_underlyings_tool,
    gateio_options_expirations_tool,
    gateio_options_contracts_tool,
    gateio_options_tickers_tool,
    gateio_options_underlying_ticker_tool,
    gateio_options_orderbook_tool,
    # Analysis Tools
    gateio_market_snapshot_tool,
    gateio_top_movers_tool,
    gateio_full_analysis_tool,
    gateio_perpetuals_tool,
)

from .hyperliquid_tools import (
    # Market Data Tools
    hyperliquid_meta_tool,
    hyperliquid_all_mids_tool,
    hyperliquid_ticker_tool,
    hyperliquid_all_tickers_tool,
    hyperliquid_orderbook_tool,
    hyperliquid_klines_tool,
    # Funding & OI Tools
    hyperliquid_funding_rate_tool,
    hyperliquid_all_funding_rates_tool,
    hyperliquid_open_interest_tool,
    hyperliquid_top_movers_tool,
    hyperliquid_exchange_stats_tool,
    # Spot Tools
    hyperliquid_spot_meta_tool,
    hyperliquid_spot_meta_and_ctxs_tool,
    # Analysis Tools
    hyperliquid_market_snapshot_tool,
    hyperliquid_full_analysis_tool,
    hyperliquid_perpetuals_tool,
    hyperliquid_recent_trades_tool,
)

from .deribit_tools import (
    # Instrument Tools
    deribit_instruments_tool,
    deribit_currencies_tool,
    # Ticker Tools
    deribit_ticker_tool,
    deribit_perpetual_ticker_tool,
    deribit_all_perpetual_tickers_tool,
    deribit_futures_tickers_tool,
    # Orderbook Tools
    deribit_orderbook_tool,
    # Trades Tools
    deribit_trades_tool,
    deribit_trades_by_currency_tool,
    # Index & Price Tools
    deribit_index_price_tool,
    deribit_index_names_tool,
    # Funding Rate Tools
    deribit_funding_rate_tool,
    deribit_funding_history_tool,
    deribit_funding_analysis_tool,
    # Volatility Tools
    deribit_historical_volatility_tool,
    deribit_dvol_tool,
    # Klines Tools
    deribit_klines_tool,
    # Open Interest Tools
    deribit_open_interest_tool,
    # Options Tools
    deribit_options_summary_tool,
    deribit_options_chain_tool,
    deribit_option_ticker_tool,
    deribit_top_options_tool,
    # Analysis Tools
    deribit_market_snapshot_tool,
    deribit_full_analysis_tool,
    deribit_exchange_stats_tool,
    # Book Summary Tools
    deribit_book_summary_tool,
    # Settlement Tools
    deribit_settlements_tool,
)

# ============================================================================
# DARTS FORECASTING TOOLS
# ============================================================================

from .darts_tools import (
    # Core Forecasting
    forecast_with_darts_statistical,
    forecast_with_darts_ml,
    forecast_with_darts_dl,
    forecast_quick,
    forecast_zero_shot,  # Chronos-2 foundation model
    list_darts_models,
    route_forecast_request,  # Intelligent routing
)

from .darts_ensemble_tools import (
    # Ensemble Forecasting
    ensemble_forecast_simple,
    ensemble_forecast_advanced,
    ensemble_auto_select,
)

from .darts_ml_tools import (
    # Model Comparison
    compare_all_models,
    auto_model_select,
)

from .darts_explainability_tools import (
    # Explainability
    explain_forecast_features,
    explain_model_decision,
)

from .darts_production_tools import (
    # Production & Backtesting
    backtest_model,
    compare_models_backtest,
)

from .production_forecast_tools import (
    # Hyperparameter Tuning
    tune_model_hyperparameters,
    get_parameter_space,
    # Cross-Validation
    cross_validate_forecast_model,
    list_cv_strategies,
    # Drift Detection
    check_model_drift,
    get_model_health_report,
    monitor_prediction_quality,
)

# ============================================================================
# MODEL REGISTRY TOOLS
# ============================================================================

from .model_registry_tools import (
    registry_list_models,
    registry_get_model_info,
    registry_recommend_model,
    registry_get_rankings,
    registry_get_stats,
    registry_compare_models,
    registry_register_result,
)

# ============================================================================
# COMPREHENSIVE ANALYTICS TOOLS
# ============================================================================

# ============================================================================
# STREAMING CONTROL TOOLS
# ============================================================================

from .streaming_control_tools import (
    start_streaming,
    stop_streaming,
    get_streaming_status,
    get_streaming_health,
    get_streaming_alerts,
    configure_streaming,
    get_realtime_analytics_status,
    get_stream_forecast,
)

from .analytics_tools import (
    # Alpha Signals (Composite Intelligence)
    compute_alpha_signals,
    get_institutional_pressure,
    compute_squeeze_probability,
    # Leverage Analytics (Positioning & Risk)
    analyze_leverage_positioning,
    compute_oi_flow_decomposition,
    compute_leverage_index,
    compute_funding_stress,
    # Regime Analytics (Market State Intelligence)
    detect_market_regime,
    detect_event_risk,
    # TimeSeries Engine (Forecasting & Anomaly Detection)
    forecast_timeseries,
    detect_anomalies,
    detect_change_points,
    detect_trend,
    # Backtesting
    backtest_forecast_model,
    # Prophet Forecasting
    forecast_with_prophet,
    # Streaming Analysis
    analyze_price_stream,
)

# ============================================================================
# INSTITUTIONAL FEATURE TOOLS (Phase 4 Week 1 - 15 Tools)
# ============================================================================

from .institutional_feature_tools import (
    # Price Features (3 tools)
    get_price_features,
    get_spread_dynamics,
    get_price_efficiency_metrics,
    # Orderbook Features (3 tools)
    get_orderbook_features,
    get_depth_imbalance,
    get_wall_detection,
    # Trade Features (3 tools)
    get_trade_features,
    get_cvd_analysis,
    get_whale_detection,
    # Funding Features (2 tools)
    get_funding_features,
    get_funding_sentiment,
    # Open Interest Features (2 tools)
    get_oi_features,
    get_leverage_risk,
    # Liquidation Features (1 tool)
    get_liquidation_features,
    # Mark Price Features (1 tool)
    get_mark_price_features,
)

# ============================================================================
# COMPOSITE INTELLIGENCE TOOLS (Phase 4 Week 2 - 10 Tools)
# ============================================================================

from .composite_intelligence_tools import (
    # Smart Money Detection (2 tools)
    get_smart_accumulation_signal,
    get_smart_money_flow,
    # Squeeze & Stop Hunt (2 tools)
    get_short_squeeze_probability,
    get_stop_hunt_detector,
    # Momentum Analysis (2 tools)
    get_momentum_quality_signal,
    get_momentum_exhaustion,
    # Risk Assessment (2 tools)
    get_market_maker_activity,
    get_liquidation_cascade_risk,
    # Market Intelligence (2 tools)
    get_institutional_phase,
    get_aggregated_intelligence,
    # Bonus: Execution Quality
    get_execution_quality,
)

# ============================================================================
# VISUALIZATION TOOLS (Phase 4 Week 3 - 5 Tools)
# ============================================================================

from .visualization_tools import (
    # Feature Candles
    get_feature_candles,
    # Liquidity Heatmap
    get_liquidity_heatmap,
    # Signal Dashboard
    get_signal_dashboard,
    # Regime Visualization
    get_regime_visualization,
    # Correlation Matrix
    get_correlation_matrix,
)

# ============================================================================
# FEATURE QUERY TOOLS (Phase 4 Week 4 - 4 Tools)
# ============================================================================

from .feature_query_tools import (
    # Historical Query
    query_historical_features,
    # CSV Export
    export_features_csv,
    # Feature Statistics
    get_feature_statistics,
    # Correlation Analysis
    get_feature_correlation_analysis,
)

__all__ = [
    # Arbitrage tools
    "analyze_crypto_arbitrage",
    "get_exchange_prices",
    "get_spread_matrix",
    "get_recent_opportunities",
    "arbitrage_scanner_health",
    # Binance Futures - Market Data
    "binance_get_ticker",
    "binance_get_prices",
    "binance_get_orderbook",
    "binance_get_trades",
    "binance_get_klines",
    # Binance Futures - Derivatives
    "binance_get_open_interest",
    "binance_get_open_interest_history",
    "binance_get_funding_rate",
    "binance_get_premium_index",
    # Binance Futures - Positioning
    "binance_get_long_short_ratio",
    "binance_get_taker_volume",
    # Binance Futures - Basis
    "binance_get_basis",
    # Binance Futures - Liquidations
    "binance_get_liquidations",
    # Binance Futures - Comprehensive
    "binance_market_snapshot",
    "binance_full_analysis",
    # Bybit Spot
    "bybit_spot_ticker",
    "bybit_spot_orderbook",
    "bybit_spot_trades",
    "bybit_spot_klines",
    "bybit_all_spot_tickers",
    # Bybit Futures
    "bybit_futures_ticker",
    "bybit_futures_orderbook",
    "bybit_open_interest",
    "bybit_funding_rate",
    "bybit_long_short_ratio",
    "bybit_historical_volatility",
    "bybit_insurance_fund",
    "bybit_all_perpetual_tickers",
    # Bybit Analysis
    "bybit_derivatives_analysis",
    "bybit_market_snapshot",
    "bybit_instruments_info",
    "bybit_options_overview",
    "bybit_risk_limit",
    "bybit_announcements",
    "bybit_full_market_analysis",
    # Binance Spot - Market Data
    "binance_spot_ticker",
    "binance_spot_price",
    "binance_spot_orderbook",
    "binance_spot_trades",
    "binance_spot_klines",
    "binance_spot_avg_price",
    "binance_spot_book_ticker",
    "binance_spot_agg_trades",
    "binance_spot_exchange_info",
    "binance_spot_rolling_ticker",
    "binance_spot_all_tickers",
    # Binance Spot - Analysis
    "binance_spot_snapshot",
    "binance_spot_full_analysis",
    # OKX - Ticker Tools
    "okx_ticker",
    "okx_all_tickers",
    "okx_index_ticker",
    # OKX - Orderbook
    "okx_orderbook",
    # OKX - Trades
    "okx_trades",
    # OKX - Klines
    "okx_klines",
    # OKX - Funding Rate
    "okx_funding_rate",
    "okx_funding_rate_history",
    # OKX - Open Interest
    "okx_open_interest",
    "okx_oi_volume",
    # OKX - Long/Short & Taker
    "okx_long_short_ratio",
    "okx_taker_volume",
    # OKX - Instruments
    "okx_instruments",
    "okx_mark_price",
    # OKX - Insurance & Platform
    "okx_insurance_fund",
    "okx_platform_volume",
    # OKX - Options
    "okx_options_summary",
    # OKX - Analysis
    "okx_market_snapshot",
    "okx_full_analysis",
    "okx_top_movers",
    # Kraken - Spot
    "kraken_spot_ticker",
    "kraken_all_spot_tickers",
    "kraken_spot_orderbook",
    "kraken_spot_trades",
    "kraken_spot_klines",
    "kraken_spread",
    "kraken_assets",
    "kraken_spot_pairs",
    # Kraken - Futures
    "kraken_futures_ticker",
    "kraken_all_futures_tickers",
    "kraken_futures_orderbook",
    "kraken_futures_trades",
    "kraken_futures_klines",
    "kraken_futures_instruments",
    # Kraken - Funding & OI
    "kraken_funding_rates",
    "kraken_open_interest",
    # Kraken - System
    "kraken_system_status",
    # Kraken - Analysis
    "kraken_top_movers",
    "kraken_market_snapshot",
    "kraken_full_analysis",
    # Gate.io - Perpetual Futures
    "gateio_futures_contracts_tool",
    "gateio_futures_contract_tool",
    "gateio_futures_ticker_tool",
    "gateio_all_futures_tickers_tool",
    "gateio_futures_orderbook_tool",
    "gateio_futures_trades_tool",
    "gateio_futures_klines_tool",
    "gateio_funding_rate_tool",
    "gateio_all_funding_rates_tool",
    "gateio_contract_stats_tool",
    "gateio_open_interest_tool",
    "gateio_liquidations_tool",
    "gateio_insurance_fund_tool",
    "gateio_risk_limit_tiers_tool",
    # Gate.io - Delivery Futures
    "gateio_delivery_contracts_tool",
    "gateio_delivery_ticker_tool",
    # Gate.io - Options
    "gateio_options_underlyings_tool",
    "gateio_options_expirations_tool",
    "gateio_options_contracts_tool",
    "gateio_options_tickers_tool",
    "gateio_options_underlying_ticker_tool",
    "gateio_options_orderbook_tool",
    # Gate.io - Analysis
    "gateio_market_snapshot_tool",
    "gateio_top_movers_tool",
    "gateio_full_analysis_tool",
    "gateio_perpetuals_tool",
    # Hyperliquid - Market Data
    "hyperliquid_meta_tool",
    "hyperliquid_all_mids_tool",
    "hyperliquid_ticker_tool",
    "hyperliquid_all_tickers_tool",
    "hyperliquid_orderbook_tool",
    "hyperliquid_klines_tool",
    # Hyperliquid - Funding & OI
    "hyperliquid_funding_rate_tool",
    "hyperliquid_all_funding_rates_tool",
    "hyperliquid_open_interest_tool",
    "hyperliquid_top_movers_tool",
    "hyperliquid_exchange_stats_tool",
    # Hyperliquid - Spot
    "hyperliquid_spot_meta_tool",
    "hyperliquid_spot_meta_and_ctxs_tool",
    # Hyperliquid - Analysis
    "hyperliquid_market_snapshot_tool",
    "hyperliquid_full_analysis_tool",
    "hyperliquid_perpetuals_tool",
    "hyperliquid_recent_trades_tool",
    # Deribit - Instruments
    "deribit_instruments_tool",
    "deribit_currencies_tool",
    # Deribit - Tickers
    "deribit_ticker_tool",
    "deribit_perpetual_ticker_tool",
    "deribit_all_perpetual_tickers_tool",
    "deribit_futures_tickers_tool",
    # Deribit - Orderbook
    "deribit_orderbook_tool",
    # Deribit - Trades
    "deribit_trades_tool",
    "deribit_trades_by_currency_tool",
    # Deribit - Index & Price
    "deribit_index_price_tool",
    "deribit_index_names_tool",
    # Deribit - Funding
    "deribit_funding_rate_tool",
    "deribit_funding_history_tool",
    "deribit_funding_analysis_tool",
    # Deribit - Volatility
    "deribit_historical_volatility_tool",
    "deribit_dvol_tool",
    # Deribit - Klines
    "deribit_klines_tool",
    # Deribit - Open Interest
    "deribit_open_interest_tool",
    # Deribit - Options
    "deribit_options_summary_tool",
    "deribit_options_chain_tool",
    "deribit_option_ticker_tool",
    "deribit_top_options_tool",
    # Deribit - Analysis
    "deribit_market_snapshot_tool",
    "deribit_full_analysis_tool",
    "deribit_exchange_stats_tool",
    # Deribit - Book Summary
    "deribit_book_summary_tool",
    # Deribit - Settlements
    "deribit_settlements_tool",
    # ========================================================================
    # DARTS FORECASTING TOOLS
    # ========================================================================
    # Darts - Core Forecasting
    "forecast_with_darts_statistical",
    "forecast_with_darts_ml",
    "forecast_with_darts_dl",
    "forecast_quick",
    "forecast_zero_shot",  # Chronos-2 foundation model
    "list_darts_models",
    "route_forecast_request",  # Intelligent routing
    # Darts - Ensemble
    "ensemble_forecast_simple",
    "ensemble_forecast_advanced",
    "ensemble_auto_select",
    # Darts - Model Comparison
    "compare_all_models",
    "auto_model_select",
    # Darts - Explainability
    "explain_forecast_features",
    "explain_model_decision",
    # Darts - Production & Backtesting
    "backtest_model",
    "compare_models_backtest",
    # ========================================================================
    # PRODUCTION FORECASTING TOOLS
    # ========================================================================
    # Hyperparameter Tuning
    "tune_model_hyperparameters",
    "get_parameter_space",
    # Cross-Validation
    "cross_validate_forecast_model",
    "list_cv_strategies",
    # Drift Detection & Monitoring
    "check_model_drift",
    "get_model_health_report",
    "monitor_prediction_quality",
    # ========================================================================
    # MODEL REGISTRY TOOLS
    # ========================================================================
    "registry_list_models",
    "registry_get_model_info",
    "registry_recommend_model",
    "registry_get_rankings",
    "registry_get_stats",
    "registry_compare_models",
    "registry_register_result",
    # ========================================================================
    # COMPREHENSIVE ANALYTICS TOOLS
    # ========================================================================
    # Alpha Signals (Composite Intelligence)
    "compute_alpha_signals",
    "get_institutional_pressure",
    "compute_squeeze_probability",
    # Leverage Analytics (Positioning & Risk)
    "analyze_leverage_positioning",
    "compute_oi_flow_decomposition",
    "compute_leverage_index",
    "compute_funding_stress",
    # Regime Analytics (Market State Intelligence)
    "detect_market_regime",
    "detect_event_risk",
    # TimeSeries Engine (Forecasting & Anomaly Detection)
    "forecast_timeseries",
    "detect_anomalies",
    "detect_change_points",
    "detect_trend",
    # Backtesting
    "backtest_forecast_model",
    # Prophet Forecasting
    "forecast_with_prophet",
    # Streaming Analysis
    "analyze_price_stream",
    # ========================================================================
    # STREAMING CONTROL TOOLS
    # ========================================================================
    "start_streaming",
    "stop_streaming",
    "get_streaming_status",
    "get_streaming_health",
    "get_streaming_alerts",
    "configure_streaming",
    "get_realtime_analytics_status",
    "get_stream_forecast",
    # ========================================================================
    # INSTITUTIONAL FEATURE TOOLS (Phase 4 Week 1 - 15 Tools)
    # ========================================================================
    # Price Features
    "get_price_features",
    "get_spread_dynamics",
    "get_price_efficiency_metrics",
    # Orderbook Features
    "get_orderbook_features",
    "get_depth_imbalance",
    "get_wall_detection",
    # Trade Features
    "get_trade_features",
    "get_cvd_analysis",
    "get_whale_detection",
    # Funding Features
    "get_funding_features",
    "get_funding_sentiment",
    # Open Interest Features
    "get_oi_features",
    "get_leverage_risk",
    # Liquidation Features
    "get_liquidation_features",
    # Mark Price Features
    "get_mark_price_features",
    # ========================================================================
    # COMPOSITE INTELLIGENCE TOOLS (Phase 4 Week 2 - 10 Tools)
    # ========================================================================
    # Smart Money Detection
    "get_smart_accumulation_signal",
    "get_smart_money_flow",
    # Squeeze & Stop Hunt
    "get_short_squeeze_probability",
    "get_stop_hunt_detector",
    # Momentum Analysis
    "get_momentum_quality_signal",
    "get_momentum_exhaustion",
    # Risk Assessment
    "get_market_maker_activity",
    "get_liquidation_cascade_risk",
    # Market Intelligence
    "get_institutional_phase",
    "get_aggregated_intelligence",
    # Bonus: Execution Quality
    "get_execution_quality",
    # ========================================================================
    # VISUALIZATION TOOLS (Phase 4 Week 3 - 5 Tools)
    # ========================================================================
    # Feature Candles
    "get_feature_candles",
    # Liquidity Heatmap
    "get_liquidity_heatmap",
    # Signal Dashboard
    "get_signal_dashboard",
    # Regime Visualization
    "get_regime_visualization",
    # Correlation Matrix
    "get_correlation_matrix",
    # ========================================================================
    # FEATURE QUERY TOOLS (Phase 4 Week 4 - 4 Tools)
    # ========================================================================
    # Historical Query
    "query_historical_features",
    # CSV Export
    "export_features_csv",
    # Feature Statistics
    "get_feature_statistics",
    # Correlation Analysis
    "get_feature_correlation_analysis",
]
