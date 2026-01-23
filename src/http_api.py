"""
HTTP API Layer for MCP Crypto Order Flow Server
================================================

Exposes all MCP tools via REST endpoints for Sibyl UI integration.
This creates a FastAPI wrapper around the MCP tools.

Usage:
    uvicorn src.http_api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /                      - API info
    GET  /health                - Health check
    GET  /tools                 - List all available tools
    POST /tools/{tool_name}     - Call any MCP tool
    GET  /tools/{tool_name}/schema - Get tool schema
    
    # Convenience endpoints
    GET  /features/{symbol}     - All features for symbol
    GET  /signals/{symbol}      - All composite signals
    GET  /forecast/{symbol}     - Quick forecast
    GET  /streaming/status      - Streaming health
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from functools import lru_cache
import inspect

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# IMPORT ALL MCP TOOLS
# =============================================================================

# Exchange Tools
from src.tools.binance_futures_tools import (
    binance_get_ticker, binance_get_prices, binance_get_orderbook,
    binance_get_trades, binance_get_klines, binance_get_open_interest,
    binance_get_open_interest_history, binance_get_funding_rate,
    binance_get_premium_index, binance_get_long_short_ratio,
    binance_get_taker_volume, binance_get_basis, binance_get_liquidations,
    binance_market_snapshot, binance_full_analysis,
)

from src.tools.binance_spot_tools import (
    binance_spot_ticker, binance_spot_price, binance_spot_orderbook,
    binance_spot_trades, binance_spot_klines, binance_spot_avg_price,
    binance_spot_book_ticker, binance_spot_agg_trades, binance_spot_exchange_info,
    binance_spot_rolling_ticker, binance_spot_all_tickers,
    binance_spot_snapshot, binance_spot_full_analysis,
)

from src.tools.bybit_tools import (
    bybit_spot_ticker, bybit_spot_orderbook, bybit_spot_trades,
    bybit_spot_klines, bybit_all_spot_tickers, bybit_futures_ticker,
    bybit_futures_orderbook, bybit_open_interest, bybit_funding_rate,
    bybit_long_short_ratio, bybit_historical_volatility, bybit_insurance_fund,
    bybit_all_perpetual_tickers, bybit_derivatives_analysis, bybit_market_snapshot,
    bybit_instruments_info, bybit_options_overview, bybit_risk_limit,
    bybit_announcements, bybit_full_market_analysis,
)

from src.tools.okx_tools import (
    okx_ticker, okx_all_tickers, okx_index_ticker, okx_orderbook,
    okx_trades, okx_klines, okx_funding_rate, okx_funding_rate_history,
    okx_open_interest, okx_oi_volume, okx_long_short_ratio,
    okx_taker_volume, okx_instruments, okx_mark_price, okx_insurance_fund,
    okx_platform_volume, okx_options_summary, okx_market_snapshot,
    okx_full_analysis, okx_top_movers,
)

from src.tools.kraken_tools import (
    kraken_spot_ticker, kraken_all_spot_tickers, kraken_spot_orderbook,
    kraken_spot_trades, kraken_spot_klines, kraken_spread, kraken_assets,
    kraken_spot_pairs, kraken_futures_ticker, kraken_all_futures_tickers,
    kraken_futures_orderbook, kraken_futures_trades, kraken_futures_klines,
    kraken_futures_instruments, kraken_funding_rates, kraken_open_interest,
    kraken_system_status, kraken_top_movers, kraken_market_snapshot,
    kraken_full_analysis,
)

from src.tools.gateio_tools import (
    gateio_futures_contracts_tool, gateio_futures_contract_tool,
    gateio_futures_ticker_tool, gateio_all_futures_tickers_tool,
    gateio_futures_orderbook_tool, gateio_futures_trades_tool,
    gateio_futures_klines_tool, gateio_funding_rate_tool,
    gateio_all_funding_rates_tool, gateio_contract_stats_tool,
    gateio_open_interest_tool, gateio_liquidations_tool,
    gateio_insurance_fund_tool, gateio_risk_limit_tiers_tool,
    gateio_delivery_contracts_tool, gateio_delivery_ticker_tool,
    gateio_options_underlyings_tool, gateio_options_expirations_tool,
    gateio_options_contracts_tool, gateio_options_tickers_tool,
    gateio_options_underlying_ticker_tool, gateio_options_orderbook_tool,
    gateio_market_snapshot_tool, gateio_top_movers_tool,
    gateio_full_analysis_tool, gateio_perpetuals_tool,
)

from src.tools.hyperliquid_tools import (
    hyperliquid_meta_tool, hyperliquid_all_mids_tool, hyperliquid_ticker_tool,
    hyperliquid_all_tickers_tool, hyperliquid_orderbook_tool,
    hyperliquid_klines_tool, hyperliquid_funding_rate_tool,
    hyperliquid_all_funding_rates_tool, hyperliquid_open_interest_tool,
    hyperliquid_top_movers_tool, hyperliquid_exchange_stats_tool,
    hyperliquid_spot_meta_tool, hyperliquid_spot_meta_and_ctxs_tool,
    hyperliquid_market_snapshot_tool, hyperliquid_full_analysis_tool,
    hyperliquid_perpetuals_tool, hyperliquid_recent_trades_tool,
)

from src.tools.deribit_tools import (
    deribit_instruments_tool, deribit_currencies_tool, deribit_ticker_tool,
    deribit_perpetual_ticker_tool, deribit_all_perpetual_tickers_tool,
    deribit_futures_tickers_tool, deribit_orderbook_tool, deribit_trades_tool,
    deribit_trades_by_currency_tool, deribit_index_price_tool,
    deribit_index_names_tool, deribit_funding_rate_tool,
    deribit_funding_history_tool, deribit_funding_analysis_tool,
    deribit_historical_volatility_tool, deribit_dvol_tool, deribit_klines_tool,
    deribit_open_interest_tool, deribit_options_summary_tool,
    deribit_options_chain_tool, deribit_option_ticker_tool,
    deribit_top_options_tool, deribit_market_snapshot_tool,
    deribit_full_analysis_tool, deribit_exchange_stats_tool,
    deribit_book_summary_tool, deribit_settlements_tool,
)

# Forecasting Tools
from src.tools.darts_tools import (
    forecast_with_darts_statistical, forecast_with_darts_ml,
    forecast_with_darts_dl, forecast_quick, forecast_zero_shot,
    list_darts_models, route_forecast_request,
)

from src.tools.darts_ensemble_tools import (
    ensemble_forecast_simple, ensemble_forecast_advanced, ensemble_auto_select,
)

from src.tools.darts_ml_tools import (
    compare_all_models, auto_model_select,
)

from src.tools.darts_explainability_tools import (
    explain_forecast_features, explain_model_decision,
)

from src.tools.darts_production_tools import (
    backtest_model, compare_models_backtest,
)

from src.tools.production_forecast_tools import (
    tune_model_hyperparameters, get_parameter_space,
    cross_validate_forecast_model, list_cv_strategies,
    check_model_drift, get_model_health_report, monitor_prediction_quality,
)

# Model Registry Tools
from src.tools.model_registry_tools import (
    registry_list_models, registry_get_model_info, registry_recommend_model,
    registry_get_rankings, registry_get_stats, registry_compare_models,
    registry_register_result,
)

# Streaming Control Tools
from src.tools.streaming_control_tools import (
    start_streaming, stop_streaming, get_streaming_status,
    get_streaming_health, get_streaming_alerts, configure_streaming,
    get_realtime_analytics_status, get_stream_forecast,
)

# Analytics Tools
from src.tools.analytics_tools import (
    compute_alpha_signals, get_institutional_pressure, compute_squeeze_probability,
    analyze_leverage_positioning, compute_oi_flow_decomposition,
    compute_leverage_index, compute_funding_stress, detect_market_regime,
    detect_event_risk, forecast_timeseries, detect_anomalies,
    detect_change_points, detect_trend, backtest_forecast_model,
    forecast_with_prophet, analyze_price_stream,
)

# Institutional Feature Tools (Phase 4 Week 1)
from src.tools.institutional_feature_tools import (
    get_price_features, get_spread_dynamics, get_price_efficiency_metrics,
    get_orderbook_features, get_depth_imbalance, get_wall_detection,
    get_trade_features, get_cvd_analysis, get_whale_detection,
    get_funding_features, get_funding_sentiment,
    get_oi_features, get_leverage_risk,
    get_liquidation_features, get_mark_price_features,
)

# Ticker Feature Tools (24h Market Stats)
from src.tools.institutional_feature_tools_ticker import get_ticker_features

# Composite Intelligence Tools (Phase 4 Week 2)
from src.tools.composite_intelligence_tools import (
    get_smart_accumulation_signal, get_smart_money_flow,
    get_short_squeeze_probability, get_stop_hunt_detector,
    get_momentum_quality_signal, get_momentum_exhaustion,
    get_market_maker_activity, get_liquidation_cascade_risk,
    get_institutional_phase, get_aggregated_intelligence,
    get_execution_quality,
)

# Visualization Tools (Phase 4 Week 3)
from src.tools.visualization_tools import (
    get_feature_candles, get_liquidity_heatmap, get_signal_dashboard,
    get_regime_visualization, get_correlation_matrix,
)

# Feature Query Tools (Phase 4 Week 4)
from src.tools.feature_query_tools import (
    query_historical_features, export_features_csv,
    get_feature_statistics, get_feature_correlation_analysis,
)

# =============================================================================
# TOOL REGISTRY - Maps tool names to functions
# =============================================================================

TOOL_REGISTRY: Dict[str, callable] = {
    # =========================================================================
    # EXCHANGE TOOLS - BINANCE FUTURES
    # =========================================================================
    "binance_get_ticker": binance_get_ticker,
    "binance_get_prices": binance_get_prices,
    "binance_get_orderbook": binance_get_orderbook,
    "binance_get_trades": binance_get_trades,
    "binance_get_klines": binance_get_klines,
    "binance_get_open_interest": binance_get_open_interest,
    "binance_get_open_interest_history": binance_get_open_interest_history,
    "binance_get_funding_rate": binance_get_funding_rate,
    "binance_get_premium_index": binance_get_premium_index,
    "binance_get_long_short_ratio": binance_get_long_short_ratio,
    "binance_get_taker_volume": binance_get_taker_volume,
    "binance_get_basis": binance_get_basis,
    "binance_get_liquidations": binance_get_liquidations,
    "binance_market_snapshot": binance_market_snapshot,
    "binance_full_analysis": binance_full_analysis,
    
    # =========================================================================
    # EXCHANGE TOOLS - BINANCE SPOT
    # =========================================================================
    "binance_spot_ticker": binance_spot_ticker,
    "binance_spot_price": binance_spot_price,
    "binance_spot_orderbook": binance_spot_orderbook,
    "binance_spot_trades": binance_spot_trades,
    "binance_spot_klines": binance_spot_klines,
    "binance_spot_avg_price": binance_spot_avg_price,
    "binance_spot_book_ticker": binance_spot_book_ticker,
    "binance_spot_agg_trades": binance_spot_agg_trades,
    "binance_spot_exchange_info": binance_spot_exchange_info,
    "binance_spot_rolling_ticker": binance_spot_rolling_ticker,
    "binance_spot_all_tickers": binance_spot_all_tickers,
    "binance_spot_snapshot": binance_spot_snapshot,
    "binance_spot_full_analysis": binance_spot_full_analysis,
    
    # =========================================================================
    # EXCHANGE TOOLS - BYBIT
    # =========================================================================
    "bybit_spot_ticker": bybit_spot_ticker,
    "bybit_spot_orderbook": bybit_spot_orderbook,
    "bybit_spot_trades": bybit_spot_trades,
    "bybit_spot_klines": bybit_spot_klines,
    "bybit_all_spot_tickers": bybit_all_spot_tickers,
    "bybit_futures_ticker": bybit_futures_ticker,
    "bybit_futures_orderbook": bybit_futures_orderbook,
    "bybit_open_interest": bybit_open_interest,
    "bybit_funding_rate": bybit_funding_rate,
    "bybit_long_short_ratio": bybit_long_short_ratio,
    "bybit_historical_volatility": bybit_historical_volatility,
    "bybit_insurance_fund": bybit_insurance_fund,
    "bybit_all_perpetual_tickers": bybit_all_perpetual_tickers,
    "bybit_derivatives_analysis": bybit_derivatives_analysis,
    "bybit_market_snapshot": bybit_market_snapshot,
    "bybit_instruments_info": bybit_instruments_info,
    "bybit_options_overview": bybit_options_overview,
    "bybit_risk_limit": bybit_risk_limit,
    "bybit_announcements": bybit_announcements,
    "bybit_full_market_analysis": bybit_full_market_analysis,
    
    # =========================================================================
    # EXCHANGE TOOLS - OKX
    # =========================================================================
    "okx_ticker": okx_ticker,
    "okx_all_tickers": okx_all_tickers,
    "okx_index_ticker": okx_index_ticker,
    "okx_orderbook": okx_orderbook,
    "okx_trades": okx_trades,
    "okx_klines": okx_klines,
    "okx_funding_rate": okx_funding_rate,
    "okx_funding_rate_history": okx_funding_rate_history,
    "okx_open_interest": okx_open_interest,
    "okx_oi_volume": okx_oi_volume,
    "okx_long_short_ratio": okx_long_short_ratio,
    "okx_taker_volume": okx_taker_volume,
    "okx_instruments": okx_instruments,
    "okx_mark_price": okx_mark_price,
    "okx_insurance_fund": okx_insurance_fund,
    "okx_platform_volume": okx_platform_volume,
    "okx_options_summary": okx_options_summary,
    "okx_market_snapshot": okx_market_snapshot,
    "okx_full_analysis": okx_full_analysis,
    "okx_top_movers": okx_top_movers,
    
    # =========================================================================
    # EXCHANGE TOOLS - KRAKEN
    # =========================================================================
    "kraken_spot_ticker": kraken_spot_ticker,
    "kraken_all_spot_tickers": kraken_all_spot_tickers,
    "kraken_spot_orderbook": kraken_spot_orderbook,
    "kraken_spot_trades": kraken_spot_trades,
    "kraken_spot_klines": kraken_spot_klines,
    "kraken_spread": kraken_spread,
    "kraken_assets": kraken_assets,
    "kraken_spot_pairs": kraken_spot_pairs,
    "kraken_futures_ticker": kraken_futures_ticker,
    "kraken_all_futures_tickers": kraken_all_futures_tickers,
    "kraken_futures_orderbook": kraken_futures_orderbook,
    "kraken_futures_trades": kraken_futures_trades,
    "kraken_futures_klines": kraken_futures_klines,
    "kraken_futures_instruments": kraken_futures_instruments,
    "kraken_funding_rates": kraken_funding_rates,
    "kraken_open_interest": kraken_open_interest,
    "kraken_system_status": kraken_system_status,
    "kraken_top_movers": kraken_top_movers,
    "kraken_market_snapshot": kraken_market_snapshot,
    "kraken_full_analysis": kraken_full_analysis,
    
    # =========================================================================
    # EXCHANGE TOOLS - GATE.IO
    # =========================================================================
    "gateio_futures_contracts": gateio_futures_contracts_tool,
    "gateio_futures_contract": gateio_futures_contract_tool,
    "gateio_futures_ticker": gateio_futures_ticker_tool,
    "gateio_all_futures_tickers": gateio_all_futures_tickers_tool,
    "gateio_futures_orderbook": gateio_futures_orderbook_tool,
    "gateio_futures_trades": gateio_futures_trades_tool,
    "gateio_futures_klines": gateio_futures_klines_tool,
    "gateio_funding_rate": gateio_funding_rate_tool,
    "gateio_all_funding_rates": gateio_all_funding_rates_tool,
    "gateio_contract_stats": gateio_contract_stats_tool,
    "gateio_open_interest": gateio_open_interest_tool,
    "gateio_liquidations": gateio_liquidations_tool,
    "gateio_insurance_fund": gateio_insurance_fund_tool,
    "gateio_risk_limit_tiers": gateio_risk_limit_tiers_tool,
    "gateio_delivery_contracts": gateio_delivery_contracts_tool,
    "gateio_delivery_ticker": gateio_delivery_ticker_tool,
    "gateio_options_underlyings": gateio_options_underlyings_tool,
    "gateio_options_expirations": gateio_options_expirations_tool,
    "gateio_options_contracts": gateio_options_contracts_tool,
    "gateio_options_tickers": gateio_options_tickers_tool,
    "gateio_options_underlying_ticker": gateio_options_underlying_ticker_tool,
    "gateio_options_orderbook": gateio_options_orderbook_tool,
    "gateio_market_snapshot": gateio_market_snapshot_tool,
    "gateio_top_movers": gateio_top_movers_tool,
    "gateio_full_analysis": gateio_full_analysis_tool,
    "gateio_perpetuals": gateio_perpetuals_tool,
    
    # =========================================================================
    # EXCHANGE TOOLS - HYPERLIQUID
    # =========================================================================
    "hyperliquid_meta": hyperliquid_meta_tool,
    "hyperliquid_all_mids": hyperliquid_all_mids_tool,
    "hyperliquid_ticker": hyperliquid_ticker_tool,
    "hyperliquid_all_tickers": hyperliquid_all_tickers_tool,
    "hyperliquid_orderbook": hyperliquid_orderbook_tool,
    "hyperliquid_klines": hyperliquid_klines_tool,
    "hyperliquid_funding_rate": hyperliquid_funding_rate_tool,
    "hyperliquid_all_funding_rates": hyperliquid_all_funding_rates_tool,
    "hyperliquid_open_interest": hyperliquid_open_interest_tool,
    "hyperliquid_top_movers": hyperliquid_top_movers_tool,
    "hyperliquid_exchange_stats": hyperliquid_exchange_stats_tool,
    "hyperliquid_spot_meta": hyperliquid_spot_meta_tool,
    "hyperliquid_spot_meta_and_ctxs": hyperliquid_spot_meta_and_ctxs_tool,
    "hyperliquid_market_snapshot": hyperliquid_market_snapshot_tool,
    "hyperliquid_full_analysis": hyperliquid_full_analysis_tool,
    "hyperliquid_perpetuals": hyperliquid_perpetuals_tool,
    "hyperliquid_recent_trades": hyperliquid_recent_trades_tool,
    
    # =========================================================================
    # EXCHANGE TOOLS - DERIBIT
    # =========================================================================
    "deribit_instruments": deribit_instruments_tool,
    "deribit_currencies": deribit_currencies_tool,
    "deribit_ticker": deribit_ticker_tool,
    "deribit_perpetual_ticker": deribit_perpetual_ticker_tool,
    "deribit_all_perpetual_tickers": deribit_all_perpetual_tickers_tool,
    "deribit_futures_tickers": deribit_futures_tickers_tool,
    "deribit_orderbook": deribit_orderbook_tool,
    "deribit_trades": deribit_trades_tool,
    "deribit_trades_by_currency": deribit_trades_by_currency_tool,
    "deribit_index_price": deribit_index_price_tool,
    "deribit_index_names": deribit_index_names_tool,
    "deribit_funding_rate": deribit_funding_rate_tool,
    "deribit_funding_history": deribit_funding_history_tool,
    "deribit_funding_analysis": deribit_funding_analysis_tool,
    "deribit_historical_volatility": deribit_historical_volatility_tool,
    "deribit_dvol": deribit_dvol_tool,
    "deribit_klines": deribit_klines_tool,
    "deribit_open_interest": deribit_open_interest_tool,
    "deribit_options_summary": deribit_options_summary_tool,
    "deribit_options_chain": deribit_options_chain_tool,
    "deribit_option_ticker": deribit_option_ticker_tool,
    "deribit_top_options": deribit_top_options_tool,
    "deribit_market_snapshot": deribit_market_snapshot_tool,
    "deribit_full_analysis": deribit_full_analysis_tool,
    "deribit_exchange_stats": deribit_exchange_stats_tool,
    "deribit_book_summary": deribit_book_summary_tool,
    "deribit_settlements": deribit_settlements_tool,
    
    # =========================================================================
    # FORECASTING TOOLS
    # =========================================================================
    "forecast_with_darts_statistical": forecast_with_darts_statistical,
    "forecast_with_darts_ml": forecast_with_darts_ml,
    "forecast_with_darts_dl": forecast_with_darts_dl,
    "forecast_quick": forecast_quick,
    "forecast_zero_shot": forecast_zero_shot,
    "list_darts_models": list_darts_models,
    "route_forecast_request": route_forecast_request,
    "ensemble_forecast_simple": ensemble_forecast_simple,
    "ensemble_forecast_advanced": ensemble_forecast_advanced,
    "ensemble_auto_select": ensemble_auto_select,
    "compare_all_models": compare_all_models,
    "auto_model_select": auto_model_select,
    "explain_forecast_features": explain_forecast_features,
    "explain_model_decision": explain_model_decision,
    "backtest_model": backtest_model,
    "compare_models_backtest": compare_models_backtest,
    
    # =========================================================================
    # PRODUCTION FORECASTING TOOLS
    # =========================================================================
    "tune_model_hyperparameters": tune_model_hyperparameters,
    "get_parameter_space": get_parameter_space,
    "cross_validate_forecast_model": cross_validate_forecast_model,
    "list_cv_strategies": list_cv_strategies,
    "check_model_drift": check_model_drift,
    "get_model_health_report": get_model_health_report,
    "monitor_prediction_quality": monitor_prediction_quality,
    
    # =========================================================================
    # MODEL REGISTRY TOOLS
    # =========================================================================
    "registry_list_models": registry_list_models,
    "registry_get_model_info": registry_get_model_info,
    "registry_recommend_model": registry_recommend_model,
    "registry_get_rankings": registry_get_rankings,
    "registry_get_stats": registry_get_stats,
    "registry_compare_models": registry_compare_models,
    "registry_register_result": registry_register_result,
    
    # =========================================================================
    # STREAMING CONTROL TOOLS
    # =========================================================================
    "start_streaming": start_streaming,
    "stop_streaming": stop_streaming,
    "get_streaming_status": get_streaming_status,
    "get_streaming_health": get_streaming_health,
    "get_streaming_alerts": get_streaming_alerts,
    "configure_streaming": configure_streaming,
    "get_realtime_analytics_status": get_realtime_analytics_status,
    "get_stream_forecast": get_stream_forecast,
    
    # =========================================================================
    # ANALYTICS TOOLS
    # =========================================================================
    "compute_alpha_signals": compute_alpha_signals,
    "get_institutional_pressure": get_institutional_pressure,
    "compute_squeeze_probability": compute_squeeze_probability,
    "analyze_leverage_positioning": analyze_leverage_positioning,
    "compute_oi_flow_decomposition": compute_oi_flow_decomposition,
    "compute_leverage_index": compute_leverage_index,
    "compute_funding_stress": compute_funding_stress,
    "detect_market_regime": detect_market_regime,
    "detect_event_risk": detect_event_risk,
    "forecast_timeseries": forecast_timeseries,
    "detect_anomalies": detect_anomalies,
    "detect_change_points": detect_change_points,
    "detect_trend": detect_trend,
    "backtest_forecast_model": backtest_forecast_model,
    "forecast_with_prophet": forecast_with_prophet,
    "analyze_price_stream": analyze_price_stream,
    
    # =========================================================================
    # INSTITUTIONAL FEATURE TOOLS (Phase 4 Week 1)
    # =========================================================================
    "get_price_features": get_price_features,
    "get_spread_dynamics": get_spread_dynamics,
    "get_price_efficiency_metrics": get_price_efficiency_metrics,
    "get_orderbook_features": get_orderbook_features,
    "get_depth_imbalance": get_depth_imbalance,
    "get_wall_detection": get_wall_detection,
    "get_trade_features": get_trade_features,
    "get_cvd_analysis": get_cvd_analysis,
    "get_whale_detection": get_whale_detection,
    "get_funding_features": get_funding_features,
    "get_funding_sentiment": get_funding_sentiment,
    "get_oi_features": get_oi_features,
    "get_leverage_risk": get_leverage_risk,
    "get_liquidation_features": get_liquidation_features,
    "get_mark_price_features": get_mark_price_features,
    "get_ticker_features": get_ticker_features,
    
    # =========================================================================
    # COMPOSITE INTELLIGENCE TOOLS (Phase 4 Week 2)
    # =========================================================================
    "get_smart_accumulation_signal": get_smart_accumulation_signal,
    "get_smart_money_flow": get_smart_money_flow,
    "get_short_squeeze_probability": get_short_squeeze_probability,
    "get_stop_hunt_detector": get_stop_hunt_detector,
    "get_momentum_quality_signal": get_momentum_quality_signal,
    "get_momentum_exhaustion": get_momentum_exhaustion,
    "get_market_maker_activity": get_market_maker_activity,
    "get_liquidation_cascade_risk": get_liquidation_cascade_risk,
    "get_institutional_phase": get_institutional_phase,
    "get_aggregated_intelligence": get_aggregated_intelligence,
    "get_execution_quality": get_execution_quality,
    
    # =========================================================================
    # VISUALIZATION TOOLS (Phase 4 Week 3)
    # =========================================================================
    "get_feature_candles": get_feature_candles,
    "get_liquidity_heatmap": get_liquidity_heatmap,
    "get_signal_dashboard": get_signal_dashboard,
    "get_regime_visualization": get_regime_visualization,
    "get_correlation_matrix": get_correlation_matrix,
    
    # =========================================================================
    # FEATURE QUERY TOOLS (Phase 4 Week 4)
    # =========================================================================
    "query_historical_features": query_historical_features,
    "export_features_csv": export_features_csv,
    "get_feature_statistics": get_feature_statistics,
    "get_feature_correlation_analysis": get_feature_correlation_analysis,
}

# =============================================================================
# TOOL CATEGORIES FOR DOCUMENTATION
# =============================================================================

TOOL_CATEGORIES = {
    "exchange_binance_futures": [
        "binance_get_ticker", "binance_get_prices", "binance_get_orderbook",
        "binance_get_trades", "binance_get_klines", "binance_get_open_interest",
        "binance_get_open_interest_history", "binance_get_funding_rate",
        "binance_get_premium_index", "binance_get_long_short_ratio",
        "binance_get_taker_volume", "binance_get_basis", "binance_get_liquidations",
        "binance_market_snapshot", "binance_full_analysis",
    ],
    "exchange_binance_spot": [
        "binance_spot_ticker", "binance_spot_price", "binance_spot_orderbook",
        "binance_spot_trades", "binance_spot_klines", "binance_spot_avg_price",
        "binance_spot_book_ticker", "binance_spot_agg_trades",
        "binance_spot_exchange_info", "binance_spot_rolling_ticker",
        "binance_spot_all_tickers", "binance_spot_snapshot",
        "binance_spot_full_analysis",
    ],
    "exchange_bybit": [
        "bybit_spot_ticker", "bybit_spot_orderbook", "bybit_spot_trades",
        "bybit_spot_klines", "bybit_all_spot_tickers", "bybit_futures_ticker",
        "bybit_futures_orderbook", "bybit_open_interest", "bybit_funding_rate",
        "bybit_long_short_ratio", "bybit_historical_volatility",
        "bybit_insurance_fund", "bybit_all_perpetual_tickers",
        "bybit_derivatives_analysis", "bybit_market_snapshot",
        "bybit_instruments_info", "bybit_options_overview", "bybit_risk_limit",
        "bybit_announcements", "bybit_full_market_analysis",
    ],
    "exchange_okx": [
        "okx_ticker", "okx_all_tickers", "okx_index_ticker", "okx_orderbook",
        "okx_trades", "okx_klines", "okx_funding_rate", "okx_funding_rate_history",
        "okx_open_interest", "okx_oi_volume", "okx_long_short_ratio",
        "okx_taker_volume", "okx_instruments", "okx_mark_price",
        "okx_insurance_fund", "okx_platform_volume", "okx_options_summary",
        "okx_market_snapshot", "okx_full_analysis", "okx_top_movers",
    ],
    "exchange_kraken": [
        "kraken_spot_ticker", "kraken_all_spot_tickers", "kraken_spot_orderbook",
        "kraken_spot_trades", "kraken_spot_klines", "kraken_spread",
        "kraken_assets", "kraken_spot_pairs", "kraken_futures_ticker",
        "kraken_all_futures_tickers", "kraken_futures_orderbook",
        "kraken_futures_trades", "kraken_futures_klines",
        "kraken_futures_instruments", "kraken_funding_rates",
        "kraken_open_interest", "kraken_system_status", "kraken_top_movers",
        "kraken_market_snapshot", "kraken_full_analysis",
    ],
    "exchange_gateio": [
        "gateio_futures_contracts", "gateio_futures_contract",
        "gateio_futures_ticker", "gateio_all_futures_tickers",
        "gateio_futures_orderbook", "gateio_futures_trades",
        "gateio_futures_klines", "gateio_funding_rate",
        "gateio_all_funding_rates", "gateio_contract_stats",
        "gateio_open_interest", "gateio_liquidations", "gateio_insurance_fund",
        "gateio_risk_limit_tiers", "gateio_delivery_contracts",
        "gateio_delivery_ticker", "gateio_options_underlyings",
        "gateio_options_expirations", "gateio_options_contracts",
        "gateio_options_tickers", "gateio_options_underlying_ticker",
        "gateio_options_orderbook", "gateio_market_snapshot",
        "gateio_top_movers", "gateio_full_analysis", "gateio_perpetuals",
    ],
    "exchange_hyperliquid": [
        "hyperliquid_meta", "hyperliquid_all_mids", "hyperliquid_ticker",
        "hyperliquid_all_tickers", "hyperliquid_orderbook", "hyperliquid_klines",
        "hyperliquid_funding_rate", "hyperliquid_all_funding_rates",
        "hyperliquid_open_interest", "hyperliquid_top_movers",
        "hyperliquid_exchange_stats", "hyperliquid_spot_meta",
        "hyperliquid_spot_meta_and_ctxs", "hyperliquid_market_snapshot",
        "hyperliquid_full_analysis", "hyperliquid_perpetuals",
        "hyperliquid_recent_trades",
    ],
    "exchange_deribit": [
        "deribit_instruments", "deribit_currencies", "deribit_ticker",
        "deribit_perpetual_ticker", "deribit_all_perpetual_tickers",
        "deribit_futures_tickers", "deribit_orderbook", "deribit_trades",
        "deribit_trades_by_currency", "deribit_index_price",
        "deribit_index_names", "deribit_funding_rate", "deribit_funding_history",
        "deribit_funding_analysis", "deribit_historical_volatility",
        "deribit_dvol", "deribit_klines", "deribit_open_interest",
        "deribit_options_summary", "deribit_options_chain",
        "deribit_option_ticker", "deribit_top_options",
        "deribit_market_snapshot", "deribit_full_analysis",
        "deribit_exchange_stats", "deribit_book_summary", "deribit_settlements",
    ],
    "forecasting": [
        "forecast_with_darts_statistical", "forecast_with_darts_ml",
        "forecast_with_darts_dl", "forecast_quick", "forecast_zero_shot",
        "list_darts_models", "route_forecast_request",
        "ensemble_forecast_simple", "ensemble_forecast_advanced",
        "ensemble_auto_select", "compare_all_models", "auto_model_select",
        "explain_forecast_features", "explain_model_decision",
        "backtest_model", "compare_models_backtest",
    ],
    "production_forecasting": [
        "tune_model_hyperparameters", "get_parameter_space",
        "cross_validate_forecast_model", "list_cv_strategies",
        "check_model_drift", "get_model_health_report",
        "monitor_prediction_quality",
    ],
    "model_registry": [
        "registry_list_models", "registry_get_model_info",
        "registry_recommend_model", "registry_get_rankings",
        "registry_get_stats", "registry_compare_models",
        "registry_register_result",
    ],
    "streaming": [
        "start_streaming", "stop_streaming", "get_streaming_status",
        "get_streaming_health", "get_streaming_alerts", "configure_streaming",
        "get_realtime_analytics_status", "get_stream_forecast",
    ],
    "analytics": [
        "compute_alpha_signals", "get_institutional_pressure",
        "compute_squeeze_probability", "analyze_leverage_positioning",
        "compute_oi_flow_decomposition", "compute_leverage_index",
        "compute_funding_stress", "detect_market_regime", "detect_event_risk",
        "forecast_timeseries", "detect_anomalies", "detect_change_points",
        "detect_trend", "backtest_forecast_model", "forecast_with_prophet",
        "analyze_price_stream",
    ],
    "institutional_features": [
        "get_price_features", "get_spread_dynamics", "get_price_efficiency_metrics",
        "get_orderbook_features", "get_depth_imbalance", "get_wall_detection",
        "get_trade_features", "get_cvd_analysis", "get_whale_detection",
        "get_funding_features", "get_funding_sentiment",
        "get_oi_features", "get_leverage_risk",
        "get_liquidation_features", "get_mark_price_features",
        "get_ticker_features",
    ],
    "composite_intelligence": [
        "get_smart_accumulation_signal", "get_smart_money_flow",
        "get_short_squeeze_probability", "get_stop_hunt_detector",
        "get_momentum_quality_signal", "get_momentum_exhaustion",
        "get_market_maker_activity", "get_liquidation_cascade_risk",
        "get_institutional_phase", "get_aggregated_intelligence",
        "get_execution_quality",
    ],
    "visualization": [
        "get_feature_candles", "get_liquidity_heatmap", "get_signal_dashboard",
        "get_regime_visualization", "get_correlation_matrix",
    ],
    "feature_query": [
        "query_historical_features", "export_features_csv",
        "get_feature_statistics", "get_feature_correlation_analysis",
    ],
}

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="MCP Crypto Order Flow Server - HTTP API",
    description="""
    HTTP REST API wrapper for MCP Crypto Order Flow Server tools.
    
    This API exposes 252+ MCP tools for:
    - 8 Exchange integrations (Binance, Bybit, OKX, Kraken, Gate.io, Hyperliquid, Deribit)
    - 38+ Darts forecasting models
    - 139 institutional features
    - 15 composite signals
    - Real-time streaming control
    - Model health monitoring
    
    Designed for Sibyl UI integration.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Streamlit runs on different ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ToolRequest(BaseModel):
    """Request model for tool invocation"""
    params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


class ToolResponse(BaseModel):
    """Response model for tool results"""
    success: bool
    tool_name: str
    result: Any
    execution_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    total_tools: int
    categories: int
    timestamp: str


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_model=Dict[str, Any])
async def root():
    """API information endpoint"""
    return {
        "name": "MCP Crypto Order Flow Server - HTTP API",
        "version": "1.0.0",
        "total_tools": len(TOOL_REGISTRY),
        "categories": len(TOOL_CATEGORIES),
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "tools_list": "/tools",
            "call_tool": "/tools/{tool_name}",
            "tool_schema": "/tools/{tool_name}/schema",
            "features": "/features/{symbol}",
            "signals": "/signals/{symbol}",
            "forecast": "/forecast/{symbol}",
            "streaming": "/streaming/status",
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        total_tools=len(TOOL_REGISTRY),
        categories=len(TOOL_CATEGORIES),
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/tools")
async def list_tools(category: Optional[str] = None):
    """
    List all available tools, optionally filtered by category.
    
    Categories:
    - exchange_binance_futures, exchange_binance_spot, exchange_bybit
    - exchange_okx, exchange_kraken, exchange_gateio
    - exchange_hyperliquid, exchange_deribit
    - forecasting, production_forecasting, model_registry
    - streaming, analytics, institutional_features
    - composite_intelligence, visualization, feature_query
    """
    if category:
        if category not in TOOL_CATEGORIES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown category: {category}. Available: {list(TOOL_CATEGORIES.keys())}"
            )
        tools = TOOL_CATEGORIES[category]
        return {
            "category": category,
            "tool_count": len(tools),
            "tools": tools
        }
    
    return {
        "total_tools": len(TOOL_REGISTRY),
        "categories": {
            name: {"count": len(tools), "tools": tools}
            for name, tools in TOOL_CATEGORIES.items()
        }
    }


@app.get("/tools/{tool_name}/schema")
async def get_tool_schema(tool_name: str):
    """Get the parameter schema for a tool"""
    if tool_name not in TOOL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")
    
    tool_func = TOOL_REGISTRY[tool_name]
    sig = inspect.signature(tool_func)
    
    params = {}
    for name, param in sig.parameters.items():
        param_info = {
            "required": param.default == inspect.Parameter.empty,
            "default": None if param.default == inspect.Parameter.empty else param.default,
        }
        
        # Get type annotation
        if param.annotation != inspect.Parameter.empty:
            param_info["type"] = str(param.annotation)
        
        params[name] = param_info
    
    return {
        "tool_name": tool_name,
        "description": tool_func.__doc__ or "No description available",
        "parameters": params
    }


@app.post("/tools/{tool_name}", response_model=ToolResponse)
async def call_tool(tool_name: str, request: ToolRequest = Body(default=ToolRequest())):
    """
    Call any MCP tool by name with provided parameters.
    
    Example:
    ```
    POST /tools/get_price_features
    {
        "params": {
            "symbol": "BTCUSDT",
            "exchange": "binance"
        }
    }
    ```
    """
    if tool_name not in TOOL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")
    
    tool_func = TOOL_REGISTRY[tool_name]
    start_time = datetime.utcnow()
    
    try:
        # Check if async
        if asyncio.iscoroutinefunction(tool_func):
            result = await tool_func(**request.params)
        else:
            result = tool_func(**request.params)
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ToolResponse(
            success=True,
            tool_name=tool_name,
            result=result,
            execution_time_ms=execution_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {e}")
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ToolResponse(
            success=False,
            tool_name=tool_name,
            result={"error": str(e), "type": type(e).__name__},
            execution_time_ms=execution_time,
            timestamp=datetime.utcnow().isoformat()
        )


# =============================================================================
# CONVENIENCE ENDPOINTS - Aggregated Data for Sibyl
# =============================================================================

@app.get("/features/{symbol}")
async def get_all_features(
    symbol: str = "BTCUSDT",
    exchange: str = Query(default="binance", description="Exchange name")
):
    """
    Get ALL institutional features for a symbol in one call.
    Returns 139+ features across 8 categories.
    """
    results = {}
    
    feature_tools = [
        ("prices", "get_price_features"),
        ("orderbook", "get_orderbook_features"),
        ("trades", "get_trade_features"),
        ("funding", "get_funding_features"),
        ("oi", "get_oi_features"),
        ("liquidations", "get_liquidation_features"),
        ("mark_prices", "get_mark_price_features"),
    ]
    
    for category, tool_name in feature_tools:
        try:
            tool_func = TOOL_REGISTRY[tool_name]
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(symbol=symbol, exchange=exchange)
            else:
                result = tool_func(symbol=symbol, exchange=exchange)
            results[category] = result
        except Exception as e:
            results[category] = {"error": str(e)}
    
    return {
        "symbol": symbol,
        "exchange": exchange,
        "timestamp": datetime.utcnow().isoformat(),
        "features": results
    }


@app.get("/signals/{symbol}")
async def get_all_signals(
    symbol: str = "BTCUSDT",
    exchange: str = Query(default="binance", description="Exchange name")
):
    """
    Get ALL composite signals for a symbol in one call.
    Returns 15 composite intelligence signals.
    """
    results = {}
    
    signal_tools = [
        ("smart_accumulation", "get_smart_accumulation_signal"),
        ("smart_money_flow", "get_smart_money_flow"),
        ("short_squeeze", "get_short_squeeze_probability"),
        ("stop_hunt", "get_stop_hunt_detector"),
        ("momentum_quality", "get_momentum_quality_signal"),
        ("momentum_exhaustion", "get_momentum_exhaustion"),
        ("market_maker", "get_market_maker_activity"),
        ("cascade_risk", "get_liquidation_cascade_risk"),
        ("institutional_phase", "get_institutional_phase"),
        ("aggregated", "get_aggregated_intelligence"),
    ]
    
    for signal_name, tool_name in signal_tools:
        try:
            tool_func = TOOL_REGISTRY[tool_name]
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(symbol=symbol, exchange=exchange)
            else:
                result = tool_func(symbol=symbol, exchange=exchange)
            results[signal_name] = result
        except Exception as e:
            results[signal_name] = {"error": str(e)}
    
    return {
        "symbol": symbol,
        "exchange": exchange,
        "timestamp": datetime.utcnow().isoformat(),
        "signals": results
    }


@app.get("/forecast/{symbol}")
async def get_forecast(
    symbol: str = "BTCUSDT",
    exchange: str = Query(default="binance", description="Exchange name"),
    horizon: int = Query(default=24, description="Forecast horizon in hours"),
    priority: str = Query(default="balanced", description="Priority: realtime, fast, balanced, accurate")
):
    """
    Get intelligent forecast for a symbol using auto-routed model.
    """
    try:
        tool_func = TOOL_REGISTRY["route_forecast_request"]
        if asyncio.iscoroutinefunction(tool_func):
            result = await tool_func(
                symbol=symbol,
                exchange=exchange,
                horizon=horizon,
                priority=priority
            )
        else:
            result = tool_func(
                symbol=symbol,
                exchange=exchange,
                horizon=horizon,
                priority=priority
            )
        
        return {
            "symbol": symbol,
            "exchange": exchange,
            "horizon": horizon,
            "priority": priority,
            "timestamp": datetime.utcnow().isoformat(),
            "forecast": result
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "exchange": exchange,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.get("/streaming/status")
async def get_streaming_status_endpoint():
    """Get streaming system status and health"""
    results = {}
    
    streaming_tools = [
        ("status", "get_streaming_status"),
        ("health", "get_streaming_health"),
        ("alerts", "get_streaming_alerts"),
    ]
    
    for name, tool_name in streaming_tools:
        try:
            tool_func = TOOL_REGISTRY[tool_name]
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func()
            else:
                result = tool_func()
            results[name] = result
        except Exception as e:
            results[name] = {"error": str(e)}
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "streaming": results
    }


@app.get("/dashboard/{symbol}")
async def get_dashboard_data(
    symbol: str = "BTCUSDT",
    exchange: str = Query(default="binance", description="Exchange name")
):
    """
    Get complete dashboard data for Sibyl main view.
    Aggregates features, signals, forecast, and streaming health.
    """
    # Fetch all data in parallel
    features_task = get_all_features(symbol, exchange)
    signals_task = get_all_signals(symbol, exchange)
    forecast_task = get_forecast(symbol, exchange, horizon=24, priority="fast")
    streaming_task = get_streaming_status_endpoint()
    
    features, signals, forecast, streaming = await asyncio.gather(
        features_task, signals_task, forecast_task, streaming_task,
        return_exceptions=True
    )
    
    return {
        "symbol": symbol,
        "exchange": exchange,
        "timestamp": datetime.utcnow().isoformat(),
        "features": features if not isinstance(features, Exception) else {"error": str(features)},
        "signals": signals if not isinstance(signals, Exception) else {"error": str(signals)},
        "forecast": forecast if not isinstance(forecast, Exception) else {"error": str(forecast)},
        "streaming": streaming if not isinstance(streaming, Exception) else {"error": str(streaming)},
    }


# =============================================================================
# ADDITIONAL CONVENIENCE ENDPOINTS - Regime, Analytics, Backtest, Cross-Exchange
# =============================================================================

@app.get("/regimes/{symbol}")
async def get_regime_data(
    symbol: str = "BTCUSDT",
    exchange: str = Query(default="binance", description="Exchange name"),
    include_history: bool = Query(default=False, description="Include regime transition history")
):
    """
    Get market regime analysis for a symbol.
    Returns current regime, transitions, and change points.
    """
    results = {}
    
    # Get current regime detection
    try:
        tool_func = TOOL_REGISTRY.get("detect_market_regime")
        if tool_func:
            if asyncio.iscoroutinefunction(tool_func):
                results["current_regime"] = await tool_func(symbol=symbol, exchange=exchange)
            else:
                results["current_regime"] = tool_func(symbol=symbol, exchange=exchange)
    except Exception as e:
        results["current_regime"] = {"error": str(e)}
    
    # Get regime visualization
    try:
        tool_func = TOOL_REGISTRY.get("get_regime_visualization")
        if tool_func:
            if asyncio.iscoroutinefunction(tool_func):
                results["visualization"] = await tool_func(symbol=symbol, exchange=exchange)
            else:
                results["visualization"] = tool_func(symbol=symbol, exchange=exchange)
    except Exception as e:
        results["visualization"] = {"error": str(e)}
    
    # Get change points if history requested
    if include_history:
        try:
            tool_func = TOOL_REGISTRY.get("detect_change_points")
            if tool_func:
                if asyncio.iscoroutinefunction(tool_func):
                    results["change_points"] = await tool_func(symbol=symbol, exchange=exchange)
                else:
                    results["change_points"] = tool_func(symbol=symbol, exchange=exchange)
        except Exception as e:
            results["change_points"] = {"error": str(e)}
    
    return {
        "symbol": symbol,
        "exchange": exchange,
        "timestamp": datetime.utcnow().isoformat(),
        "regimes": results
    }


@app.get("/analytics/{symbol}")
async def get_analytics_data(
    symbol: str = "BTCUSDT",
    exchange: str = Query(default="binance", description="Exchange name"),
    analysis_type: str = Query(default="comprehensive", description="Type: comprehensive, alpha, leverage, timeseries")
):
    """
    Get comprehensive analytics for a symbol.
    Returns alpha signals, leverage analysis, and timeseries metrics.
    """
    results = {}
    
    # Define analytics based on type
    if analysis_type in ["comprehensive", "alpha"]:
        alpha_tools = [
            ("alpha_signals", "compute_alpha_signals"),
            ("institutional_pressure", "get_institutional_pressure"),
            ("squeeze_probability", "compute_squeeze_probability"),
        ]
        for name, tool_name in alpha_tools:
            try:
                tool_func = TOOL_REGISTRY.get(tool_name)
                if tool_func:
                    if asyncio.iscoroutinefunction(tool_func):
                        results[name] = await tool_func(symbol=symbol, exchange=exchange)
                    else:
                        results[name] = tool_func(symbol=symbol, exchange=exchange)
            except Exception as e:
                results[name] = {"error": str(e)}
    
    if analysis_type in ["comprehensive", "leverage"]:
        leverage_tools = [
            ("leverage_positioning", "analyze_leverage_positioning"),
            ("leverage_index", "compute_leverage_index"),
            ("funding_stress", "compute_funding_stress"),
            ("oi_flow", "compute_oi_flow_decomposition"),
        ]
        for name, tool_name in leverage_tools:
            try:
                tool_func = TOOL_REGISTRY.get(tool_name)
                if tool_func:
                    if asyncio.iscoroutinefunction(tool_func):
                        results[name] = await tool_func(symbol=symbol, exchange=exchange)
                    else:
                        results[name] = tool_func(symbol=symbol, exchange=exchange)
            except Exception as e:
                results[name] = {"error": str(e)}
    
    if analysis_type in ["comprehensive", "timeseries"]:
        ts_tools = [
            ("anomalies", "detect_anomalies"),
            ("trend", "detect_trend"),
            ("event_risk", "detect_event_risk"),
        ]
        for name, tool_name in ts_tools:
            try:
                tool_func = TOOL_REGISTRY.get(tool_name)
                if tool_func:
                    if asyncio.iscoroutinefunction(tool_func):
                        results[name] = await tool_func(symbol=symbol, exchange=exchange)
                    else:
                        results[name] = tool_func(symbol=symbol, exchange=exchange)
            except Exception as e:
                results[name] = {"error": str(e)}
    
    return {
        "symbol": symbol,
        "exchange": exchange,
        "analysis_type": analysis_type,
        "timestamp": datetime.utcnow().isoformat(),
        "analytics": results
    }


@app.get("/backtest/{symbol}")
async def get_backtest_data(
    symbol: str = "BTCUSDT",
    exchange: str = Query(default="binance", description="Exchange name"),
    model_type: str = Query(default="auto", description="Model type: auto, statistical, ml, dl"),
    periods: int = Query(default=30, description="Number of periods for backtesting")
):
    """
    Get backtesting results for forecast models.
    Returns model performance metrics and comparison.
    """
    results = {}
    
    # Get backtest results
    try:
        tool_func = TOOL_REGISTRY.get("backtest_model")
        if tool_func:
            if asyncio.iscoroutinefunction(tool_func):
                results["backtest"] = await tool_func(
                    symbol=symbol,
                    exchange=exchange,
                    model_type=model_type,
                    periods=periods
                )
            else:
                results["backtest"] = tool_func(
                    symbol=symbol,
                    exchange=exchange,
                    model_type=model_type,
                    periods=periods
                )
    except Exception as e:
        results["backtest"] = {"error": str(e)}
    
    # Get model health report
    try:
        tool_func = TOOL_REGISTRY.get("get_model_health_report")
        if tool_func:
            if asyncio.iscoroutinefunction(tool_func):
                results["model_health"] = await tool_func()
            else:
                results["model_health"] = tool_func()
    except Exception as e:
        results["model_health"] = {"error": str(e)}
    
    # Get prediction quality
    try:
        tool_func = TOOL_REGISTRY.get("monitor_prediction_quality")
        if tool_func:
            if asyncio.iscoroutinefunction(tool_func):
                results["prediction_quality"] = await tool_func(symbol=symbol)
            else:
                results["prediction_quality"] = tool_func(symbol=symbol)
    except Exception as e:
        results["prediction_quality"] = {"error": str(e)}
    
    return {
        "symbol": symbol,
        "exchange": exchange,
        "model_type": model_type,
        "periods": periods,
        "timestamp": datetime.utcnow().isoformat(),
        "backtest_results": results
    }


@app.get("/cross-exchange/{symbol}")
async def get_cross_exchange_data(
    symbol: str = "BTCUSDT",
    exchanges: str = Query(default="binance,bybit,okx", description="Comma-separated exchange list")
):
    """
    Get cross-exchange comparison data for a symbol.
    Returns prices, funding rates, and open interest across exchanges.
    """
    exchange_list = [e.strip().lower() for e in exchanges.split(",")]
    results = {
        "prices": {},
        "funding_rates": {},
        "open_interest": {},
        "arbitrage": {}
    }
    
    # Collect data from each exchange
    for exch in exchange_list:
        # Price data
        try:
            ticker_tool = f"{exch}_get_ticker" if exch == "binance" else f"{exch}_ticker" if exch == "hyperliquid" else f"{exch}_futures_ticker"
            tool_func = TOOL_REGISTRY.get(ticker_tool)
            if tool_func:
                if asyncio.iscoroutinefunction(tool_func):
                    results["prices"][exch] = await tool_func(symbol=symbol)
                else:
                    results["prices"][exch] = tool_func(symbol=symbol)
        except Exception as e:
            results["prices"][exch] = {"error": str(e)}
        
        # Funding rate
        try:
            funding_tool = f"{exch}_get_funding_rate" if exch == "binance" else f"{exch}_funding_rate"
            tool_func = TOOL_REGISTRY.get(funding_tool)
            if tool_func:
                if asyncio.iscoroutinefunction(tool_func):
                    results["funding_rates"][exch] = await tool_func(symbol=symbol)
                else:
                    results["funding_rates"][exch] = tool_func(symbol=symbol)
        except Exception as e:
            results["funding_rates"][exch] = {"error": str(e)}
        
        # Open interest
        try:
            oi_tool = f"{exch}_get_open_interest" if exch == "binance" else f"{exch}_open_interest"
            tool_func = TOOL_REGISTRY.get(oi_tool)
            if tool_func:
                if asyncio.iscoroutinefunction(tool_func):
                    results["open_interest"][exch] = await tool_func(symbol=symbol)
                else:
                    results["open_interest"][exch] = tool_func(symbol=symbol)
        except Exception as e:
            results["open_interest"][exch] = {"error": str(e)}
    
    return {
        "symbol": symbol,
        "exchanges": exchange_list,
        "timestamp": datetime.utcnow().isoformat(),
        "cross_exchange": results
    }


@app.get("/model-comparison")
async def get_model_comparison(
    symbol: str = Query(default="BTCUSDT", description="Trading symbol"),
    exchange: str = Query(default="binance", description="Exchange name"),
    horizon: int = Query(default=24, description="Forecast horizon in hours")
):
    """
    Compare multiple forecasting models on a symbol.
    Returns performance metrics, rankings, and recommendations.
    """
    results = {}
    
    # Compare all models
    try:
        tool_func = TOOL_REGISTRY.get("compare_all_models")
        if tool_func:
            if asyncio.iscoroutinefunction(tool_func):
                results["comparison"] = await tool_func(
                    symbol=symbol,
                    exchange=exchange,
                    horizon=horizon
                )
            else:
                results["comparison"] = tool_func(
                    symbol=symbol,
                    exchange=exchange,
                    horizon=horizon
                )
    except Exception as e:
        results["comparison"] = {"error": str(e)}
    
    # Get model rankings
    try:
        tool_func = TOOL_REGISTRY.get("registry_get_rankings")
        if tool_func:
            if asyncio.iscoroutinefunction(tool_func):
                results["rankings"] = await tool_func()
            else:
                results["rankings"] = tool_func()
    except Exception as e:
        results["rankings"] = {"error": str(e)}
    
    # Get recommendation
    try:
        tool_func = TOOL_REGISTRY.get("registry_recommend_model")
        if tool_func:
            if asyncio.iscoroutinefunction(tool_func):
                results["recommendation"] = await tool_func(
                    symbol=symbol,
                    horizon=horizon
                )
            else:
                results["recommendation"] = tool_func(
                    symbol=symbol,
                    horizon=horizon
                )
    except Exception as e:
        results["recommendation"] = {"error": str(e)}
    
    # Get auto-selected model
    try:
        tool_func = TOOL_REGISTRY.get("auto_model_select")
        if tool_func:
            if asyncio.iscoroutinefunction(tool_func):
                results["auto_select"] = await tool_func(
                    symbol=symbol,
                    exchange=exchange,
                    horizon=horizon
                )
            else:
                results["auto_select"] = tool_func(
                    symbol=symbol,
                    exchange=exchange,
                    horizon=horizon
                )
    except Exception as e:
        results["auto_select"] = {"error": str(e)}
    
    return {
        "symbol": symbol,
        "exchange": exchange,
        "horizon": horizon,
        "timestamp": datetime.utcnow().isoformat(),
        "model_comparison": results
    }


@app.get("/ticker-features/{symbol}")
async def get_ticker_features_endpoint(
    symbol: str = "BTCUSDT",
    exchange: str = Query(default="binance", description="Exchange name")
):
    """
    Get ticker-based 24h market features for a symbol.
    Returns 10 features: volume_24h, high_24h, low_24h, price_change, etc.
    """
    try:
        tool_func = TOOL_REGISTRY.get("get_ticker_features")
        if tool_func:
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(symbol=symbol, exchange=exchange)
            else:
                result = tool_func(symbol=symbol, exchange=exchange)
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": datetime.utcnow().isoformat(),
                "ticker_features": result
            }
        else:
            return {
                "symbol": symbol,
                "exchange": exchange,
                "error": "get_ticker_features tool not found",
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        return {
            "symbol": symbol,
            "exchange": exchange,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# =============================================================================
# SYSTEM ENDPOINTS - Health, Exchanges, Symbols, Tools
# =============================================================================

@app.get("/")
async def root():
    """API root - health check and info"""
    return {
        "name": "MCP Crypto Order Flow Server",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "total_tools": len(TOOL_REGISTRY),
        "total_categories": len(TOOL_CATEGORIES),
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """Comprehensive system health check"""
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {}
    }
    
    # Check tool registry
    health["components"]["tool_registry"] = {
        "status": "healthy",
        "total_tools": len(TOOL_REGISTRY),
        "categories": len(TOOL_CATEGORIES)
    }
    
    # Check each category
    for category, tools in TOOL_CATEGORIES.items():
        available = sum(1 for t in tools if t in TOOL_REGISTRY)
        health["components"][category] = {
            "status": "healthy" if available == len(tools) else "degraded",
            "available": available,
            "total": len(tools)
        }
    
    # Overall status
    degraded = [c for c, v in health["components"].items() if v.get("status") == "degraded"]
    if degraded:
        health["status"] = "degraded"
        health["degraded_components"] = degraded
    
    return health


@app.get("/exchanges")
async def list_exchanges():
    """List all supported exchanges with their capabilities"""
    exchanges = [
        {
            "id": "binance",
            "name": "Binance Futures",
            "type": "futures",
            "tools": len([t for t in TOOL_REGISTRY if t.startswith("binance_") and "spot" not in t]),
            "features": ["orderbook", "trades", "funding", "oi", "liquidations", "mark_price", "ticker"]
        },
        {
            "id": "binance_spot",
            "name": "Binance Spot",
            "type": "spot",
            "tools": len([t for t in TOOL_REGISTRY if "binance_spot" in t]),
            "features": ["orderbook", "trades", "ticker"]
        },
        {
            "id": "bybit",
            "name": "Bybit",
            "type": "futures",
            "tools": len([t for t in TOOL_REGISTRY if t.startswith("bybit_")]),
            "features": ["orderbook", "trades", "funding", "oi", "liquidations", "mark_price", "ticker"]
        },
        {
            "id": "okx",
            "name": "OKX",
            "type": "futures",
            "tools": len([t for t in TOOL_REGISTRY if t.startswith("okx_")]),
            "features": ["orderbook", "trades", "funding", "oi", "liquidations", "mark_price", "ticker"]
        },
        {
            "id": "kraken",
            "name": "Kraken",
            "type": "futures",
            "tools": len([t for t in TOOL_REGISTRY if t.startswith("kraken_")]),
            "features": ["orderbook", "trades", "funding", "oi", "ticker"]
        },
        {
            "id": "gateio",
            "name": "Gate.io",
            "type": "futures",
            "tools": len([t for t in TOOL_REGISTRY if t.startswith("gateio_")]),
            "features": ["orderbook", "trades", "funding", "oi", "liquidations", "ticker"]
        },
        {
            "id": "hyperliquid",
            "name": "Hyperliquid",
            "type": "perps",
            "tools": len([t for t in TOOL_REGISTRY if t.startswith("hyperliquid_")]),
            "features": ["orderbook", "trades", "funding", "oi"]
        },
        {
            "id": "deribit",
            "name": "Deribit",
            "type": "options",
            "tools": len([t for t in TOOL_REGISTRY if t.startswith("deribit_")]),
            "features": ["orderbook", "trades", "funding", "oi", "options", "greeks", "volatility"]
        }
    ]
    
    return {
        "exchanges": exchanges,
        "total": len(exchanges),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/symbols/{exchange}")
async def list_symbols(
    exchange: str,
    market_type: str = Query(default="futures", description="Market type: spot, futures, perps, options")
):
    """List available symbols for an exchange - ONLY system-supported symbols"""
    
    # System supported symbols (9 trading pairs)
    SUPPORTED_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ARUSDT",  # Majors
        "BRETTUSDT", "POPCATUSDT", "WIFUSDT", "PNUTUSDT"       # Memes
    ]
    
    # Exchange-specific symbol formats
    exchange_symbols = {
        "binance": SUPPORTED_SYMBOLS,
        "binance_spot": SUPPORTED_SYMBOLS,
        "bybit": SUPPORTED_SYMBOLS,
        "okx": [s.replace("USDT", "-USDT-SWAP") for s in SUPPORTED_SYMBOLS[:5]],  # Only majors
        "kraken": ["XXBTZUSD", "XETHZUSD", "SOLUSD", "XXRPZUSD", "ARUSD"],  # Only majors
        "gateio": SUPPORTED_SYMBOLS,
        "hyperliquid": SUPPORTED_SYMBOLS,
        "deribit": ["BTC-PERPETUAL", "ETH-PERPETUAL"]  # Options/perps only
    }
    
    symbols = exchange_symbols.get(exchange.lower(), SUPPORTED_SYMBOLS)
    
    return {
        "exchange": exchange,
        "market_type": market_type,
        "symbols": symbols,
        "total": len(symbols),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/tools")
async def list_tools_endpoint(
    category: Optional[str] = Query(default=None, description="Filter by category")
):
    """List all available tools"""
    if category:
        tools = TOOL_CATEGORIES.get(category, [])
        return {
            "category": category,
            "tools": tools,
            "total": len(tools),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Return all tools grouped by category
    return {
        "categories": {cat: len(tools) for cat, tools in TOOL_CATEGORIES.items()},
        "tools_by_category": TOOL_CATEGORIES,
        "total_tools": len(TOOL_REGISTRY),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/tools/{tool_name}/schema")
async def get_tool_schema(tool_name: str):
    """Get schema/documentation for a specific tool"""
    import inspect
    
    if tool_name not in TOOL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")
    
    tool_func = TOOL_REGISTRY[tool_name]
    
    # Extract function signature and docstring
    sig = inspect.signature(tool_func)
    doc = tool_func.__doc__ or "No documentation available"
    
    params = {}
    for param_name, param in sig.parameters.items():
        param_info = {
            "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "any",
            "required": param.default == inspect.Parameter.empty
        }
        if param.default != inspect.Parameter.empty:
            param_info["default"] = str(param.default)
        params[param_name] = param_info
    
    return {
        "tool_name": tool_name,
        "description": doc.split("\n")[0] if doc else "",
        "documentation": doc,
        "parameters": params,
        "is_async": asyncio.iscoroutinefunction(tool_func),
        "timestamp": datetime.utcnow().isoformat()
    }


# =============================================================================
    
    uvicorn.run(app, host=host, port=port)
