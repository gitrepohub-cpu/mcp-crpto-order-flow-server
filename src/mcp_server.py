"""
MCP Server for Crypto Arbitrage Analysis.
Provides real-time arbitrage analysis across multiple cryptocurrency exchanges.

This server connects to the Go-based crypto-futures-arbitrage-scanner via WebSocket
and exposes MCP tools for AI assistants to analyze arbitrage opportunities.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastmcp import FastMCP

# Ensure project root is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging before imports
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from libraries
logging.getLogger("fastmcp").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)

# Import crypto arbitrage tools
from src.tools.crypto_arbitrage_tool import (
    analyze_crypto_arbitrage,
    get_exchange_prices,
    get_spread_matrix,
    get_recent_opportunities,
    arbitrage_scanner_health,
    # New data stream tools
    get_orderbooks,
    get_trades,
    get_funding_rates,
    get_liquidations,
    get_open_interest,
    get_mark_prices,
    get_ticker_24h,
    get_market_summary
)

# Import Binance Futures REST tools
from src.tools.binance_futures_tools import (
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

# Import Bybit REST tools
from src.tools.bybit_tools import (
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

# Import Binance Spot REST tools
from src.tools.binance_spot_tools import (
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

# Import OKX REST tools
from src.tools.okx_tools import (
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

# Import Kraken REST tools
from src.tools.kraken_tools import (
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

# Import Gate.io REST tools
from src.tools.gateio_tools import (
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

# Import Hyperliquid REST tools
from src.tools.hyperliquid_tools import (
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

# Import Deribit REST tools
from src.tools.deribit_tools import (
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

# Import DuckDB Historical Data Tools
from src.tools.duckdb_historical_tools import (
    get_historical_prices,
    get_historical_trades,
    get_historical_funding_rates,
    get_historical_liquidations,
    get_historical_open_interest,
    get_database_stats,
    query_custom_historical,
)

# Import Live + Historical Combined Tools
from src.tools.live_historical_tools import (
    get_market_snapshot_full,
    get_price_with_history,
    get_funding_arbitrage_analysis,
    get_liquidation_heatmap,
    compare_live_vs_historical,
)

# Import Institutional Feature Tools (Phase 4 Week 1 - 15 Tools)
from src.tools.institutional_feature_tools import (
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

# Import Composite Intelligence Tools (Phase 4 Week 2 - 10 Tools)
from src.tools.composite_intelligence_tools import (
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

# Import Visualization Tools (Phase 4 Week 3 - 5 Tools)
from src.tools.visualization_tools import (
    get_feature_candles,
    get_liquidity_heatmap,
    get_signal_dashboard,
    get_regime_visualization,
    get_correlation_matrix,
)

# Import Feature Calculation Framework
from src.features.registry import FeatureRegistry

# Initialize MCP server
mcp = FastMCP("Crypto Arbitrage Analysis Server")

# Initialize and register Feature Calculators
# The registry auto-discovers calculators from src/features/calculators/
try:
    feature_registry = FeatureRegistry()
    feature_registry.discover_calculators()
    feature_registry.register_all_with_mcp(mcp)
    logger.info(f"Registered {len(feature_registry.calculators)} feature calculators")
except Exception as e:
    logger.warning(f"Feature calculator registration failed: {e}")


# ============================================================================
# CRYPTO ARBITRAGE MCP TOOLS
# ============================================================================

@mcp.tool()
async def analyze_crypto_arbitrage_tool(
    symbol: str = "BTCUSDT",
    min_profit_threshold: float = 0.05,
    include_spreads: bool = True,
    include_opportunities: bool = True
) -> str:
    """
    Comprehensive crypto futures arbitrage analysis across multiple exchanges.
    
    Analyzes real-time price discrepancies across 10+ exchanges including:
    - Futures: Binance, Bybit, OKX, Kraken, Gate.io, Hyperliquid, Paradex
    - Spot: Binance, Bybit
    - Oracle: Pyth Network
    
    Args:
        symbol: Trading pair to analyze. Supported: BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT
        min_profit_threshold: Minimum profit % to highlight opportunities (default 0.05%)
        include_spreads: Include pairwise spread matrix between all exchanges
        include_opportunities: Include list of recent arbitrage opportunities
    
    Returns:
        XML analysis including market status, prices, spreads, and recommendations
    
    Example:
        "Analyze BTC arbitrage opportunities" → analyze_crypto_arbitrage_tool(symbol="BTCUSDT")
    """
    try:
        logger.info(f"Analyzing crypto arbitrage for {symbol}")
        result = await analyze_crypto_arbitrage(
            symbol=symbol,
            min_profit_threshold=min_profit_threshold,
            include_spreads=include_spreads,
            include_opportunities=include_opportunities
        )
        return result
    except Exception as e:
        logger.error(f"Error analyzing crypto arbitrage for {symbol}: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="ANALYSIS_FAILED">
  <message>Failed to analyze crypto arbitrage for {symbol}</message>
  <details>{str(e)}</details>
  <suggestions>
    <suggestion>Ensure the Go arbitrage scanner is running on port 8082</suggestion>
    <suggestion>Run: cd crypto-futures-arbitrage-scanner &amp;&amp; go run main.go</suggestion>
  </suggestions>
</error>"""


@mcp.tool()
async def get_crypto_prices(symbol: str = None) -> str:
    """
    Get current real-time prices from all connected cryptocurrency exchanges.
    
    Args:
        symbol: Specific trading pair (BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT).
                If None, returns prices for all tracked symbols.
    
    Returns:
        XML with current prices from each exchange, sorted by price.
    
    Example:
        "What's the current BTC price on all exchanges?" → get_crypto_prices(symbol="BTCUSDT")
    """
    try:
        logger.info(f"Getting crypto prices for {symbol or 'all symbols'}")
        result = await get_exchange_prices(symbol=symbol)
        return result
    except Exception as e:
        logger.error(f"Error getting crypto prices: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="PRICE_FETCH_FAILED">
  <message>Failed to get crypto prices</message>
  <details>{str(e)}</details>
</error>"""


@mcp.tool()
async def get_crypto_spreads(symbol: str = "BTCUSDT") -> str:
    """
    Get the pairwise spread matrix showing price differences between all exchanges.
    
    The spread matrix shows the profit percentage if you were to:
    - Buy on exchange A (row)
    - Sell on exchange B (column)
    
    Args:
        symbol: Trading pair to analyze (BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT)
    
    Returns:
        XML spread matrix with best opportunities and full exchange comparison.
    
    Example:
        "Show me the spread matrix for BTC" → get_crypto_spreads(symbol="BTCUSDT")
    """
    try:
        logger.info(f"Getting spread matrix for {symbol}")
        result = await get_spread_matrix(symbol=symbol)
        return result
    except Exception as e:
        logger.error(f"Error getting spread matrix: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="SPREAD_FETCH_FAILED">
  <message>Failed to get spread matrix for {symbol}</message>
  <details>{str(e)}</details>
</error>"""


@mcp.tool()
async def get_arbitrage_opportunities(
    symbol: str = None,
    min_profit: float = 0.0,
    limit: int = 20
) -> str:
    """
    Get recent arbitrage opportunities detected by the scanner.
    
    Args:
        symbol: Filter by trading pair (None = all symbols)
        min_profit: Minimum profit percentage to include (e.g., 0.05 = 0.05%)
        limit: Maximum number of opportunities to return (default 20)
    
    Returns:
        XML list of arbitrage opportunities with buy/sell exchanges and profit.
    
    Example:
        "Show me the best arbitrage opportunities" → get_arbitrage_opportunities(min_profit=0.1)
    """
    try:
        logger.info(f"Getting arbitrage opportunities: symbol={symbol}, min_profit={min_profit}")
        result = await get_recent_opportunities(
            symbol=symbol,
            min_profit=min_profit,
            limit=limit
        )
        return result
    except Exception as e:
        logger.error(f"Error getting arbitrage opportunities: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="OPPORTUNITIES_FETCH_FAILED">
  <message>Failed to get arbitrage opportunities</message>
  <details>{str(e)}</details>
</error>"""


@mcp.tool()
async def crypto_scanner_health() -> str:
    """
    Check health and connectivity of the crypto arbitrage scanner.
    
    Verifies WebSocket connection to Go scanner backend, tracked symbols,
    connected exchanges, and cached opportunities.
    
    Returns:
        XML health status with connection details and troubleshooting tips.
    
    Example:
        "Is the arbitrage scanner working?" → crypto_scanner_health()
    """
    try:
        logger.info("Checking crypto arbitrage scanner health")
        result = await arbitrage_scanner_health()
        return result
    except Exception as e:
        logger.error(f"Crypto scanner health check failed: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<health_check status="ERROR">
  <error>{str(e)}</error>
  <suggestions>
    <suggestion>Ensure the Go arbitrage scanner is running</suggestion>
    <suggestion>Run: cd crypto-futures-arbitrage-scanner &amp;&amp; go run main.go</suggestion>
  </suggestions>
</health_check>"""


@mcp.tool()
async def compare_exchange_prices(
    symbol: str = "BTCUSDT",
    exchange1: str = "binance_futures",
    exchange2: str = "bybit_futures"
) -> str:
    """
    Compare prices between two specific exchanges.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT)
        exchange1: First exchange ID
        exchange2: Second exchange ID
        
    Exchange IDs: binance_futures, binance_spot, bybit_futures, bybit_spot,
                  okx_futures, kraken_futures, gate_futures, hyperliquid_futures,
                  paradex_futures, pyth
    
    Returns:
        XML comparison with prices and spread percentage.
    
    Example:
        "Compare BTC on Binance vs Bybit" → compare_exchange_prices("BTCUSDT", "binance_futures", "bybit_futures")
    """
    try:
        from src.storage.websocket_client import get_arbitrage_client
        from src.formatters.crypto_xml_formatter import CryptoArbitrageFormatter
        
        client = get_arbitrage_client()
        formatter = CryptoArbitrageFormatter()
        
        if not client._started:
            await client.connect()
            await asyncio.sleep(1.0)
        
        prices = await client.get_prices_snapshot(symbol)
        symbol_prices = prices.get(symbol, {})
        
        price1_data = symbol_prices.get(exchange1, {})
        price2_data = symbol_prices.get(exchange2, {})
        
        price1 = price1_data.get("price", 0) if isinstance(price1_data, dict) else price1_data
        price2 = price2_data.get("price", 0) if isinstance(price2_data, dict) else price2_data
        
        if not price1 or not price2:
            missing = []
            if not price1:
                missing.append(exchange1)
            if not price2:
                missing.append(exchange2)
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<comparison_error>
  <message>Missing price data for: {', '.join(missing)}</message>
  <available_sources>{', '.join(symbol_prices.keys())}</available_sources>
</comparison_error>"""
        
        spread_pct = ((price2 - price1) / price1) * 100
        
        name1 = formatter.SOURCE_NAMES.get(exchange1, exchange1)
        name2 = formatter.SOURCE_NAMES.get(exchange2, exchange2)
        
        recommendation = ""
        if abs(spread_pct) < 0.02:
            recommendation = "Spread is negligible - no arbitrage opportunity"
        elif spread_pct > 0:
            recommendation = f"Potential: BUY on {name1}, SELL on {name2}"
        else:
            recommendation = f"Potential: BUY on {name2}, SELL on {name1}"
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<exchange_comparison symbol="{symbol}">
  <exchange1 name="{name1}" id="{exchange1}">
    <price>{formatter._format_price(price1)}</price>
    <raw>{price1}</raw>
  </exchange1>
  <exchange2 name="{name2}" id="{exchange2}">
    <price>{formatter._format_price(price2)}</price>
    <raw>{price2}</raw>
  </exchange2>
  <spread>
    <absolute>{formatter._format_price(abs(price2 - price1))}</absolute>
    <percentage>{spread_pct:.4f}%</percentage>
    <direction>{"exchange2 higher" if spread_pct > 0 else "exchange1 higher"}</direction>
  </spread>
  <recommendation>{recommendation}</recommendation>
</exchange_comparison>"""
        
    except Exception as e:
        logger.error(f"Error comparing exchange prices: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error>
  <message>Failed to compare prices</message>
  <details>{str(e)}</details>
</error>"""


# ============================================================================
# NEW DATA STREAM MCP TOOLS
# ============================================================================

@mcp.tool()
async def get_orderbook_data(
    symbol: str = None,
    exchange: str = None
) -> str:
    """
    Get real-time orderbook depth from all connected exchanges.
    
    Shows top 10 bid/ask levels, spread, and depth imbalance for each exchange.
    Useful for analyzing market microstructure and liquidity.
    
    Args:
        symbol: Specific symbol (BTCUSDT, ETHUSDT, etc.) or None for all
        exchange: Specific exchange ID or None for all exchanges
    
    Returns:
        XML with orderbook depth including spread, bids, asks, and imbalance.
    
    Example:
        "Show me the BTC orderbook on all exchanges" → get_orderbook_data(symbol="BTCUSDT")
    """
    try:
        logger.info(f"Getting orderbook data for {symbol or 'all'} on {exchange or 'all exchanges'}")
        return await get_orderbooks(symbol=symbol, exchange=exchange)
    except Exception as e:
        logger.error(f"Error getting orderbook data: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="ORDERBOOK_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_trades(
    symbol: str = None,
    exchange: str = None,
    limit: int = 50
) -> str:
    """
    Get recent trades from all connected exchanges.
    
    Shows individual trades with price, quantity, side (buy/sell), and aggregated
    volume statistics including buy/sell ratio.
    
    Args:
        symbol: Specific symbol or None for all
        exchange: Specific exchange or None for all
        limit: Maximum trades per exchange (default 50)
    
    Returns:
        XML with recent trades and volume statistics.
    
    Example:
        "Show me recent BTC trades" → get_trades(symbol="BTCUSDT")
    """
    try:
        logger.info(f"Getting recent trades for {symbol or 'all'}")
        return await get_trades(symbol=symbol, exchange=exchange, limit=limit)
    except Exception as e:
        logger.error(f"Error getting trades: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="TRADES_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_funding_rate_data(symbol: str = None) -> str:
    """
    Get funding rates from all futures exchanges.
    
    Funding rates indicate market sentiment - positive rates mean longs pay shorts
    (bullish sentiment), negative rates mean shorts pay longs (bearish sentiment).
    
    Args:
        symbol: Specific symbol or None for all tracked symbols
    
    Returns:
        XML with funding rates, annualized rates, and sentiment indicators.
    
    Example:
        "What are the current BTC funding rates?" → get_funding_rate_data(symbol="BTCUSDT")
    """
    try:
        logger.info(f"Getting funding rates for {symbol or 'all'}")
        return await get_funding_rates(symbol=symbol)
    except Exception as e:
        logger.error(f"Error getting funding rates: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="FUNDING_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_liquidation_data(symbol: str = None, limit: int = 20) -> str:
    """
    Get recent liquidation events from exchanges.
    
    Shows forced liquidations (margin calls) with side, price, quantity, and value.
    Large liquidations often indicate market volatility and potential reversals.
    
    Args:
        symbol: Specific symbol or None for all
        limit: Maximum liquidations to return (default 20)
    
    Returns:
        XML with liquidation events and aggregated long/short liquidation totals.
    
    Example:
        "Show me recent BTC liquidations" → get_liquidation_data(symbol="BTCUSDT")
    """
    try:
        logger.info(f"Getting liquidations for {symbol or 'all'}")
        return await get_liquidations(symbol=symbol, limit=limit)
    except Exception as e:
        logger.error(f"Error getting liquidations: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="LIQUIDATIONS_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_open_interest_data(symbol: str = None) -> str:
    """
    Get open interest from all futures exchanges.
    
    Open interest represents total outstanding derivative contracts.
    Rising OI with price = strong trend, Falling OI = weakening trend.
    
    Args:
        symbol: Specific symbol or None for all
    
    Returns:
        XML with open interest per exchange and totals.
    
    Example:
        "What's the BTC open interest across exchanges?" → get_open_interest_data(symbol="BTCUSDT")
    """
    try:
        logger.info(f"Getting open interest for {symbol or 'all'}")
        return await get_open_interest(symbol=symbol)
    except Exception as e:
        logger.error(f"Error getting open interest: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="OI_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_mark_price_data(symbol: str = None) -> str:
    """
    Get mark prices and basis from futures exchanges.
    
    Mark price is the fair price used for liquidations. Basis is the difference
    between futures and spot - positive basis (contango) means bullish sentiment.
    
    Args:
        symbol: Specific symbol or None for all
    
    Returns:
        XML with mark prices, index prices, basis, and basis percentage.
    
    Example:
        "What's the BTC futures basis?" → get_mark_price_data(symbol="BTCUSDT")
    """
    try:
        logger.info(f"Getting mark prices for {symbol or 'all'}")
        return await get_mark_prices(symbol=symbol)
    except Exception as e:
        logger.error(f"Error getting mark prices: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="MARK_PRICE_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_24h_ticker_data(symbol: str = None) -> str:
    """
    Get 24-hour ticker statistics from all exchanges.
    
    Shows volume, high/low, price change percentage, and trade counts
    for the last 24 hours.
    
    Args:
        symbol: Specific symbol or None for all
    
    Returns:
        XML with 24h statistics per exchange.
    
    Example:
        "What's BTC 24h volume across exchanges?" → get_24h_ticker_data(symbol="BTCUSDT")
    """
    try:
        logger.info(f"Getting 24h ticker for {symbol or 'all'}")
        return await get_ticker_24h(symbol=symbol)
    except Exception as e:
        logger.error(f"Error getting 24h ticker: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="TICKER_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_market_overview(symbol: str = "BTCUSDT") -> str:
    """
    Get comprehensive market overview combining all data for a symbol.
    
    Aggregates prices, orderbooks, funding rates, open interest, 24h stats,
    recent trades, and liquidations into a single comprehensive view.
    
    Args:
        symbol: Symbol to analyze (default BTCUSDT)
    
    Returns:
        XML with complete market summary across all exchanges.
    
    Example:
        "Give me a complete BTC market overview" → get_market_overview(symbol="BTCUSDT")
    """
    try:
        logger.info(f"Getting market overview for {symbol}")
        return await get_market_summary(symbol=symbol)
    except Exception as e:
        logger.error(f"Error getting market overview: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="SUMMARY_FAILED"><message>{str(e)}</message></error>"""


# ============================================================================
# DUCKDB HISTORICAL DATA MCP TOOLS
# ============================================================================

@mcp.tool()
async def get_historical_price_data(
    symbol: str = "BTCUSDT",
    exchange: str = None,
    market_type: str = "futures",
    minutes: int = 60,
    limit: int = 1000
) -> str:
    """
    Query historical price data from DuckDB database.
    
    Retrieves stored price data collected by the real-time collector.
    Supports filtering by exchange, market type, and time range.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, SOLUSDT, etc.)
        exchange: Specific exchange or None for all (binance, bybit, okx, kraken, gateio, hyperliquid)
        market_type: 'futures' or 'spot'
        minutes: How many minutes of history to retrieve
        limit: Maximum number of records
    
    Returns:
        XML with historical price data and statistics.
    
    Example:
        "Get last hour of BTC prices from Binance futures" → get_historical_price_data(symbol="BTCUSDT", exchange="binance", minutes=60)
    """
    try:
        logger.info(f"Getting historical prices for {symbol}")
        result = await get_historical_prices(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            minutes=minutes,
            limit=limit
        )
        return result
    except Exception as e:
        logger.error(f"Error getting historical prices: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="HISTORICAL_QUERY_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_historical_trade_data(
    symbol: str = "BTCUSDT",
    exchange: str = None,
    market_type: str = "futures",
    minutes: int = 60,
    side: str = None,
    limit: int = 500
) -> str:
    """
    Query historical trade data from DuckDB database.
    
    Retrieves stored trades with volume analysis. Can filter by
    buy or sell side to analyze directional flow.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, SOLUSDT, etc.)
        exchange: Specific exchange or None for all
        market_type: 'futures' or 'spot'
        minutes: How many minutes of history
        side: Filter by 'buy' or 'sell' (optional, None for all)
        limit: Maximum number of records
    
    Returns:
        XML with trade data including buy/sell volume stats.
    
    Example:
        "Show BTC buy trades in last hour" → get_historical_trade_data(symbol="BTCUSDT", side="buy", minutes=60)
    """
    try:
        logger.info(f"Getting historical trades for {symbol}")
        result = await get_historical_trades(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            minutes=minutes,
            side=side,
            limit=limit
        )
        return result
    except Exception as e:
        logger.error(f"Error getting historical trades: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="HISTORICAL_QUERY_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_historical_funding_data(
    symbol: str = "BTCUSDT",
    exchange: str = None,
    hours: int = 24
) -> str:
    """
    Query historical funding rate data from DuckDB database.
    
    Retrieves funding rate history with cumulative funding calculations
    and annualized rate analysis across exchanges.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
        exchange: Specific exchange or None for all
        hours: Hours of history to retrieve
    
    Returns:
        XML with funding rate history and cumulative analysis.
    
    Example:
        "Show BTC funding rate history for last 24 hours" → get_historical_funding_data(symbol="BTCUSDT", hours=24)
    """
    try:
        logger.info(f"Getting historical funding rates for {symbol}")
        result = await get_historical_funding_rates(
            symbol=symbol,
            exchange=exchange,
            hours=hours
        )
        return result
    except Exception as e:
        logger.error(f"Error getting historical funding: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="HISTORICAL_QUERY_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_historical_liquidation_data(
    symbol: str = "BTCUSDT",
    exchange: str = None,
    hours: int = 24,
    min_value: float = 10000
) -> str:
    """
    Query historical liquidation data from DuckDB database.
    
    Retrieves liquidation history with long/short breakdown.
    Essential for identifying market stress and cascade events.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
        exchange: Specific exchange or None for all
        hours: Hours of history to retrieve
        min_value: Minimum liquidation value in USD (default $10,000)
    
    Returns:
        XML with liquidation history and summary statistics.
    
    Example:
        "Show large BTC liquidations over last 24 hours" → get_historical_liquidation_data(symbol="BTCUSDT", hours=24, min_value=100000)
    """
    try:
        logger.info(f"Getting historical liquidations for {symbol}")
        result = await get_historical_liquidations(
            symbol=symbol,
            exchange=exchange,
            hours=hours,
            min_value=min_value
        )
        return result
    except Exception as e:
        logger.error(f"Error getting historical liquidations: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="HISTORICAL_QUERY_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_historical_oi_data(
    symbol: str = "BTCUSDT",
    exchange: str = None,
    hours: int = 24
) -> str:
    """
    Query historical open interest data from DuckDB database.
    
    Retrieves OI history to track positioning changes over time.
    Shows how leverage and market exposure has evolved.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
        exchange: Specific exchange or None for all
        hours: Hours of history to retrieve
    
    Returns:
        XML with OI history and change statistics.
    
    Example:
        "Show BTC open interest history" → get_historical_oi_data(symbol="BTCUSDT", hours=24)
    """
    try:
        logger.info(f"Getting historical OI for {symbol}")
        result = await get_historical_open_interest(
            symbol=symbol,
            exchange=exchange,
            hours=hours
        )
        return result
    except Exception as e:
        logger.error(f"Error getting historical OI: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="HISTORICAL_QUERY_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_database_statistics() -> str:
    """
    Get statistics about the stored market data in DuckDB.
    
    Shows database size, record counts per table, date ranges,
    and available symbols/exchanges in the historical data.
    
    Returns:
        XML with database statistics and data inventory.
    
    Example:
        "How much data do we have stored?" → get_database_statistics()
    """
    try:
        logger.info("Getting database statistics")
        result = await get_database_stats()
        return result
    except Exception as e:
        logger.error(f"Error getting database stats: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="DATABASE_QUERY_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def query_historical_analytics(
    symbol: str = "BTCUSDT",
    query_type: str = "price_ohlc",
    exchange: str = "binance",
    market_type: str = "futures",
    hours: int = 24,
    aggregation: str = "1h"
) -> str:
    """
    Run advanced analytical queries on historical data.
    
    Supports multiple query types for in-depth analysis:
    - price_ohlc: OHLC candlestick aggregation at custom intervals
    - volume_profile: Volume distribution by price level
    - volatility: Rolling volatility analysis
    - funding_cumulative: Cumulative funding costs
    - liquidation_cascade: Liquidation cascade detection
    
    Args:
        symbol: Trading pair to analyze
        query_type: Type of analysis - price_ohlc, volume_profile, volatility, funding_cumulative, liquidation_cascade
        exchange: Exchange to query
        market_type: 'futures' or 'spot'
        hours: Hours of historical data to analyze
        aggregation: Time bucket ('5m', '15m', '1h', '4h', '1d')
    
    Returns:
        XML with analytical results based on query type.
    
    Example:
        "Generate 1-hour OHLC candles for BTC" → query_historical_analytics(symbol="BTCUSDT", query_type="price_ohlc", aggregation="1h")
    """
    try:
        logger.info(f"Running {query_type} analytics for {symbol}")
        result = await query_custom_historical(
            query_type=query_type,
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            hours=hours,
            aggregation=aggregation
        )
        return result
    except Exception as e:
        logger.error(f"Error running analytics query: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="ANALYTICS_QUERY_FAILED"><message>{str(e)}</message></error>"""


# ============================================================================
# LIVE + HISTORICAL COMBINED MCP TOOLS
# ============================================================================

@mcp.tool()
async def get_full_market_snapshot(
    symbol: str = "BTCUSDT",
    include_historical: bool = True,
    historical_minutes: int = 60
) -> str:
    """
    Get comprehensive market snapshot combining LIVE and HISTORICAL data.
    
    Combines real-time streaming data with historical context for
    complete market intelligence including:
    - Live prices from all exchanges
    - Live funding rates
    - Live orderbooks
    - Historical price statistics (range, volatility)
    - Historical trade flow (buy/sell volume)
    - Recent liquidations summary
    - Market analysis and signals
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
        include_historical: Whether to include historical context
        historical_minutes: Minutes of historical data for context
    
    Returns:
        XML with combined live + historical market intelligence.
    
    Example:
        "Give me full BTC market analysis" → get_full_market_snapshot(symbol="BTCUSDT")
    """
    try:
        logger.info(f"Getting full market snapshot for {symbol}")
        return await get_market_snapshot_full(
            symbol=symbol,
            include_historical=include_historical,
            historical_minutes=historical_minutes
        )
    except Exception as e:
        logger.error(f"Error getting full snapshot: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="SNAPSHOT_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_price_with_historical_context(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    market_type: str = "futures",
    historical_minutes: int = 30
) -> str:
    """
    Get current live price with historical price context.
    
    Combines the latest live price with historical statistics
    including OHLC, percentiles, and price change analysis.
    
    Args:
        symbol: Trading pair
        exchange: Exchange to query
        market_type: 'futures' or 'spot'
        historical_minutes: Minutes of history for context
    
    Returns:
        XML with live price + historical range + change analysis.
    
    Example:
        "What's BTC price now vs last hour?" → get_price_with_historical_context(symbol="BTCUSDT", historical_minutes=60)
    """
    try:
        logger.info(f"Getting price with history for {symbol}")
        return await get_price_with_history(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type,
            historical_minutes=historical_minutes
        )
    except Exception as e:
        logger.error(f"Error getting price with history: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="PRICE_HISTORY_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def analyze_funding_arbitrage(
    symbol: str = "BTCUSDT",
    historical_hours: int = 24
) -> str:
    """
    Analyze funding rate arbitrage opportunities across exchanges.
    
    Combines live funding rates with historical patterns to identify
    profitable funding arbitrage opportunities (delta-neutral carry).
    
    Args:
        symbol: Trading pair to analyze
        historical_hours: Hours of funding history to analyze
    
    Returns:
        XML with funding arbitrage opportunities and historical patterns.
    
    Example:
        "Find funding arbitrage opportunities for BTC" → analyze_funding_arbitrage(symbol="BTCUSDT")
    """
    try:
        logger.info(f"Analyzing funding arbitrage for {symbol}")
        return await get_funding_arbitrage_analysis(
            symbol=symbol,
            historical_hours=historical_hours
        )
    except Exception as e:
        logger.error(f"Error analyzing funding arbitrage: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="FUNDING_ANALYSIS_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_liquidation_heatmap_analysis(
    symbol: str = "BTCUSDT",
    hours: int = 24,
    price_buckets: int = 20
) -> str:
    """
    Generate liquidation heatmap showing where liquidations occurred by price level.
    
    Identifies price zones with heavy liquidation activity, useful for
    understanding support/resistance and stop-loss clustering.
    
    Args:
        symbol: Trading pair to analyze
        hours: Hours of liquidation data to include
        price_buckets: Number of price buckets for the heatmap
    
    Returns:
        XML heatmap of liquidations by price level with intensity.
    
    Example:
        "Show BTC liquidation heatmap" → get_liquidation_heatmap_analysis(symbol="BTCUSDT")
    """
    try:
        logger.info(f"Generating liquidation heatmap for {symbol}")
        return await get_liquidation_heatmap(
            symbol=symbol,
            hours=hours,
            price_buckets=price_buckets
        )
    except Exception as e:
        logger.error(f"Error generating liquidation heatmap: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="HEATMAP_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def detect_price_anomalies(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    market_type: str = "futures"
) -> str:
    """
    Compare current live price against historical averages to detect anomalies.
    
    Uses z-score analysis across multiple timeframes (5m, 15m, 1h, 4h, 24h)
    to identify significant price deviations that may indicate trading opportunities.
    
    Args:
        symbol: Trading pair to analyze
        exchange: Exchange to compare
        market_type: 'futures' or 'spot'
    
    Returns:
        XML comparison with deviation analysis and signals.
    
    Example:
        "Is BTC price unusually high or low right now?" → detect_price_anomalies(symbol="BTCUSDT")
    """
    try:
        logger.info(f"Detecting price anomalies for {symbol}")
        return await compare_live_vs_historical(
            symbol=symbol,
            exchange=exchange,
            market_type=market_type
        )
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="ANOMALY_DETECTION_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def list_feature_calculators() -> str:
    """
    List all available advanced feature calculators.
    
    Shows all registered plugin-based feature calculators with their
    descriptions, categories, and parameters.
    
    Returns:
        XML listing of all available feature calculators.
    
    Example:
        "What advanced analytics features are available?" → list_feature_calculators()
    """
    try:
        calculators = feature_registry.list_calculators()
        
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<feature_calculators count="{len(calculators)}">',
        ]
        
        # Group by category
        categories = {}
        for calc in calculators:
            cat = calc.get('category', 'general')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(calc)
        
        for cat, calcs in sorted(categories.items()):
            xml_parts.append(f'  <category name="{cat}">')
            for calc in calcs:
                xml_parts.append(f'    <calculator name="{calc["name"]}" version="{calc["version"]}">')
                xml_parts.append(f'      <description>{calc["description"]}</description>')
                xml_parts.append(f'      <mcp_tool>calculate_{calc["name"]}</mcp_tool>')
                xml_parts.append('      <parameters>')
                for pname, pschema in calc.get('parameters', {}).items():
                    ptype = pschema.get('type', 'str')
                    pdefault = pschema.get('default', 'None')
                    pdesc = pschema.get('description', '')
                    xml_parts.append(f'        <param name="{pname}" type="{ptype}" default="{pdefault}">{pdesc}</param>')
                xml_parts.append('      </parameters>')
                xml_parts.append('    </calculator>')
            xml_parts.append('  </category>')
        
        xml_parts.append('</feature_calculators>')
        return '\n'.join(xml_parts)
        
    except Exception as e:
        logger.error(f"Error listing feature calculators: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="LIST_FAILED"><message>{str(e)}</message></error>"""


# ============================================================================
# ADVANCED ANALYTICS MCP TOOLS (Feature Intelligence Framework)
# ============================================================================

# Lazy-loaded analytics engine (initialized on first use to improve startup time)
_feature_engine = None

def _get_feature_engine():
    """Get or create the feature engine (lazy initialization)."""
    global _feature_engine
    if _feature_engine is None:
        from src.analytics import FeatureEngine
        _feature_engine = FeatureEngine()
    return _feature_engine


@mcp.tool()
async def get_market_intelligence(
    symbol: str = "BTCUSDT",
    layers: str = "all"
) -> str:
    """
    Get comprehensive market intelligence using the Advanced Feature Framework.
    
    Computes 5 layers of analytics:
    1. Order Flow & Microstructure - Liquidity imbalance, vacuum, persistence
    2. Leverage & Positioning - OI flow, liquidation pressure, funding stress
    3. Cross-Exchange Intelligence - Price leadership, arbitrage, flow sync
    4. Regime & Volatility - Market regime detection, event risk
    5. Alpha Signals - Institutional pressure, squeeze probability, absorption
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT)
        layers: Which layers to compute - "all", "order_flow", "leverage", 
                "cross_exchange", "regime", "alpha_signals", or comma-separated
    
    Returns:
        XML with complete market intelligence analysis.
    
    Example:
        "Analyze BTC market intelligence" → get_market_intelligence(symbol="BTCUSDT")
    """
    try:
        logger.info(f"Computing market intelligence for {symbol}")
        
        # Get data from exchange client
        from src.storage.direct_exchange_client import get_direct_client
        client = get_direct_client()
        
        if not client._started:
            await client.connect()
            await asyncio.sleep(2.0)
        
        # Gather all data
        prices = await client.get_prices_snapshot(symbol)
        orderbooks = await client.get_orderbooks(symbol)
        trades = await client.get_trades(symbol)
        funding = await client.get_funding_rates(symbol)
        oi = await client.get_open_interest(symbol)
        liquidations = await client.get_liquidations(symbol)
        mark_prices = await client.get_mark_prices(symbol)
        tickers = await client.get_ticker_24h(symbol)
        
        # Prepare data for feature engine
        data = {
            "prices": prices,
            "orderbook": orderbooks,
            "trades": trades.get("trades", []),
            "all_exchange_trades": trades.get("by_exchange", {}),
            "funding_rates": funding,
            "open_interest": oi,
            "liquidations": liquidations.get("liquidations", []),
            "mark_prices": mark_prices,
            "ticker_24h": tickers
        }
        
        # Compute features
        feature_engine = _get_feature_engine()
        result = await feature_engine.compute_all_features(symbol, data)
        
        # Format as XML
        return _format_intelligence_xml(result, layers)
        
    except Exception as e:
        logger.error(f"Error computing market intelligence: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="INTELLIGENCE_FAILED">
  <message>Failed to compute market intelligence for {symbol}</message>
  <details>{str(e)}</details>
</error>"""


@mcp.tool()
async def get_institutional_pressure(symbol: str = "BTCUSDT") -> str:
    """
    Detect institutional buying/selling pressure.
    
    Combines orderbook imbalance, trade delta, OI flow, funding skew, and basis
    to compute a composite institutional pressure score from -100 to +100.
    
    Args:
        symbol: Trading pair to analyze
    
    Returns:
        XML with pressure score, direction, strength, and component breakdown.
    
    Example:
        "What's the institutional pressure on BTC?" → get_institutional_pressure("BTCUSDT")
    """
    try:
        logger.info(f"Computing institutional pressure for {symbol}")
        
        from src.storage.direct_exchange_client import get_direct_client
        client = get_direct_client()
        
        if not client._started:
            await client.connect()
            await asyncio.sleep(2.0)
        
        # Get required data
        prices = await client.get_prices_snapshot(symbol)
        orderbooks = await client.get_orderbooks(symbol)
        trades = await client.get_trades(symbol)
        funding = await client.get_funding_rates(symbol)
        oi = await client.get_open_interest(symbol)
        mark_prices = await client.get_mark_prices(symbol)
        
        # Compute layer features
        from src.analytics import OrderFlowAnalytics, LeverageAnalytics, CrossExchangeAnalytics
        
        order_flow = OrderFlowAnalytics()
        leverage = LeverageAnalytics()
        cross_exchange = CrossExchangeAnalytics()
        
        of_features = {
            "liquidity_imbalance": order_flow.compute_liquidity_imbalance(symbol, orderbooks),
            "trade_aggression": order_flow.compute_trade_aggression(symbol, trades.get("trades", []))
        }
        
        lev_features = {
            "oi_flow": leverage.compute_oi_flow_decomposition(symbol, oi, trades.get("trades", [])),
            "funding_stress": leverage.compute_funding_stress(symbol, funding),
            "basis_regime": leverage.compute_basis_regime(symbol, mark_prices, prices)
        }
        
        ce_features = {
            "flow_synchronization": cross_exchange.compute_flow_synchronization(
                symbol, trades.get("by_exchange", {})
            )
        }
        
        # Compute pressure
        from src.analytics import AlphaSignalEngine
        alpha = AlphaSignalEngine()
        pressure = alpha.compute_institutional_pressure(symbol, of_features, lev_features, ce_features)
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<institutional_pressure symbol="{symbol}">
  <score>{pressure['pressure_score']}</score>
  <direction>{pressure['pressure_direction']}</direction>
  <strength>{pressure['pressure_strength']}</strength>
  <confidence>{pressure['confidence']}%</confidence>
  <trend>{pressure['trend']}</trend>
  <recommendation>{pressure['recommendation']}</recommendation>
  <components>
    <orderbook_imbalance>{pressure['components'].get('orderbook_imbalance', 0)}</orderbook_imbalance>
    <trade_delta>{pressure['components'].get('trade_delta', 0)}</trade_delta>
    <oi_flow>{pressure['components'].get('oi_flow', 0)}</oi_flow>
    <funding_skew>{pressure['components'].get('funding_skew', 0)}</funding_skew>
    <basis_trend>{pressure['components'].get('basis_trend', 0)}</basis_trend>
    <flow_sync>{pressure['components'].get('flow_sync', 0)}</flow_sync>
  </components>
</institutional_pressure>"""
        
    except Exception as e:
        logger.error(f"Error computing institutional pressure: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="PRESSURE_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_squeeze_probability(symbol: str = "BTCUSDT") -> str:
    """
    Compute probability of a short or long squeeze.
    
    Analyzes leverage buildup, funding extremes, liquidation clusters, and 
    liquidity vacuum to predict potential squeeze events.
    
    Args:
        symbol: Trading pair to analyze
    
    Returns:
        XML with squeeze probability, type, trigger levels, and components.
    
    Example:
        "What's the squeeze risk for BTC?" → get_squeeze_probability("BTCUSDT")
    """
    try:
        logger.info(f"Computing squeeze probability for {symbol}")
        
        from src.storage.direct_exchange_client import get_direct_client
        client = get_direct_client()
        
        if not client._started:
            await client.connect()
            await asyncio.sleep(2.0)
        
        # Get required data
        prices = await client.get_prices_snapshot(symbol)
        orderbooks = await client.get_orderbooks(symbol)
        trades = await client.get_trades(symbol)
        funding = await client.get_funding_rates(symbol)
        oi = await client.get_open_interest(symbol)
        liquidations = await client.get_liquidations(symbol)
        mark_prices = await client.get_mark_prices(symbol)
        
        # Get current price
        current_price = 0.0
        for ex_data in prices.values():
            if symbol in ex_data:
                current_price = float(ex_data[symbol].get("price", 0))
                break
        
        # Compute features
        from src.analytics import OrderFlowAnalytics, LeverageAnalytics, RegimeAnalytics
        
        order_flow = OrderFlowAnalytics()
        leverage = LeverageAnalytics()
        regime = RegimeAnalytics()
        
        of_features = {
            "liquidity_vacuum": order_flow.compute_liquidity_vacuum(symbol, orderbooks)
        }
        
        lev_features = {
            "leverage_index": leverage.compute_leverage_index(symbol, oi, trades.get("trades", [])),
            "funding_stress": leverage.compute_funding_stress(symbol, funding),
            "liquidation_pressure": leverage.compute_liquidation_pressure(
                symbol, liquidations.get("liquidations", []), current_price
            )
        }
        
        regime_features = {
            "regime": regime.detect_regime(symbol, of_features, lev_features, prices)
        }
        
        # Compute squeeze probability
        from src.analytics import AlphaSignalEngine
        alpha = AlphaSignalEngine()
        squeeze = alpha.compute_squeeze_probability(symbol, of_features, lev_features, regime_features)
        
        # Format trigger levels
        triggers_xml = ""
        for level in squeeze.get("trigger_levels", [])[:5]:
            triggers_xml += f"""
    <level price="{level.get('price', 0)}" pct_from_current="{level.get('pct_from_current', 0):.2f}%" value_at_risk="{level.get('value_at_risk', 0):.0f}" />"""
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<squeeze_probability symbol="{symbol}">
  <probability>{squeeze['squeeze_probability']:.1f}%</probability>
  <squeeze_type>{squeeze['squeeze_type']}</squeeze_type>
  <intensity>{squeeze['squeeze_intensity']:.1f}</intensity>
  <time_to_squeeze>{squeeze['time_to_squeeze']}</time_to_squeeze>
  <warning>{squeeze.get('warning', '')}</warning>
  <components>
    <leverage_risk>{squeeze['components'].get('leverage_risk', 0):.1f}</leverage_risk>
    <funding_crowding>{squeeze['components'].get('funding_crowding', 0):.1f}</funding_crowding>
    <crowded_side>{squeeze['components'].get('crowded_side', 'NEUTRAL')}</crowded_side>
    <liq_dominance>{squeeze['components'].get('liq_dominance', 'NONE')}</liq_dominance>
    <vacuum_score>{squeeze['components'].get('vacuum_score', 0):.1f}</vacuum_score>
  </components>
  <trigger_levels>{triggers_xml}
  </trigger_levels>
</squeeze_probability>"""
        
    except Exception as e:
        logger.error(f"Error computing squeeze probability: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="SQUEEZE_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_market_regime(symbol: str = "BTCUSDT") -> str:
    """
    Detect current market regime and risk conditions.
    
    Classifies market into: ACCUMULATION, DISTRIBUTION, BREAKOUT, SQUEEZE,
    MEAN_REVERSION, CHAOS, or CONSOLIDATION.
    
    Also computes event risk level and active warnings.
    
    Args:
        symbol: Trading pair to analyze
    
    Returns:
        XML with current regime, confidence, event risk, and warnings.
    
    Example:
        "What's the current market regime for BTC?" → get_market_regime("BTCUSDT")
    """
    try:
        logger.info(f"Detecting market regime for {symbol}")
        
        from src.storage.direct_exchange_client import get_direct_client
        client = get_direct_client()
        
        if not client._started:
            await client.connect()
            await asyncio.sleep(2.0)
        
        # Get required data
        prices = await client.get_prices_snapshot(symbol)
        orderbooks = await client.get_orderbooks(symbol)
        trades = await client.get_trades(symbol)
        funding = await client.get_funding_rates(symbol)
        oi = await client.get_open_interest(symbol)
        liquidations = await client.get_liquidations(symbol)
        mark_prices = await client.get_mark_prices(symbol)
        tickers = await client.get_ticker_24h(symbol)
        
        # Get current price
        current_price = 0.0
        for ex_data in prices.values():
            if symbol in ex_data:
                current_price = float(ex_data[symbol].get("price", 0))
                break
        
        # Compute features
        from src.analytics import OrderFlowAnalytics, LeverageAnalytics, CrossExchangeAnalytics, RegimeAnalytics
        
        order_flow = OrderFlowAnalytics()
        leverage_analytics = LeverageAnalytics()
        cross_exchange = CrossExchangeAnalytics()
        regime_analytics = RegimeAnalytics()
        
        of_features = {
            "liquidity_imbalance": order_flow.compute_liquidity_imbalance(symbol, orderbooks),
            "liquidity_vacuum": order_flow.compute_liquidity_vacuum(symbol, orderbooks),
            "trade_aggression": order_flow.compute_trade_aggression(symbol, trades.get("trades", []))
        }
        
        lev_features = {
            "oi_flow": leverage_analytics.compute_oi_flow_decomposition(symbol, oi, trades.get("trades", [])),
            "leverage_index": leverage_analytics.compute_leverage_index(symbol, oi, trades.get("trades", [])),
            "funding_stress": leverage_analytics.compute_funding_stress(symbol, funding),
            "liquidation_pressure": leverage_analytics.compute_liquidation_pressure(
                symbol, liquidations.get("liquidations", []), current_price
            )
        }
        
        ce_features = {
            "spread_arbitrage": cross_exchange.compute_spread_arbitrage(symbol, prices, orderbooks),
            "flow_synchronization": cross_exchange.compute_flow_synchronization(
                symbol, trades.get("by_exchange", {})
            )
        }
        
        # Detect regime
        regime = regime_analytics.detect_regime(symbol, of_features, lev_features, prices)
        event_risk = regime_analytics.detect_event_risk(symbol, of_features, lev_features, ce_features)
        volatility = regime_analytics.compute_volatility_state(symbol, prices, tickers)
        
        # Format warnings
        warnings_xml = ""
        for warning in event_risk.get("active_warnings", [])[:5]:
            warnings_xml += f"\n    <warning>{warning}</warning>"
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<market_regime symbol="{symbol}">
  <current_regime>{regime['current_regime']}</current_regime>
  <regime_confidence>{regime['confidence']:.1f}%</regime_confidence>
  <regime_duration>{regime.get('regime_duration', 0)}</regime_duration>
  
  <event_risk>
    <risk_level>{event_risk['risk_level']}</risk_level>
    <risk_score>{event_risk['risk_score']:.1f}</risk_score>
    <recommended_action>{event_risk.get('recommended_action', '')}</recommended_action>
  </event_risk>
  
  <volatility_state>
    <percentile>{volatility.get('volatility_percentile', 0):.0f}</percentile>
    <state>{volatility.get('volatility_state', 'NORMAL')}</state>
    <expansion_probability>{volatility.get('expansion_probability', 0):.1f}%</expansion_probability>
  </volatility_state>
  
  <active_warnings>{warnings_xml}
  </active_warnings>
  
  <regime_characteristics>
    <description>{regime.get('characteristics', {}).get('description', '')}</description>
    <typical_behavior>{regime.get('characteristics', {}).get('typical_behavior', '')}</typical_behavior>
    <risk_level>{regime.get('characteristics', {}).get('risk_level', '')}</risk_level>
  </regime_characteristics>
</market_regime>"""
        
    except Exception as e:
        logger.error(f"Error detecting market regime: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="REGIME_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_liquidity_analysis(symbol: str = "BTCUSDT") -> str:
    """
    Analyze orderbook liquidity and microstructure.
    
    Computes liquidity imbalance at multiple depth levels, vacuum zones,
    support/resistance persistence, and microstructure efficiency.
    
    Args:
        symbol: Trading pair to analyze
    
    Returns:
        XML with liquidity imbalance, vacuum score, persistence zones, efficiency.
    
    Example:
        "Analyze BTC orderbook liquidity" → get_liquidity_analysis("BTCUSDT")
    """
    try:
        logger.info(f"Analyzing liquidity for {symbol}")
        
        from src.storage.direct_exchange_client import get_direct_client
        client = get_direct_client()
        
        if not client._started:
            await client.connect()
            await asyncio.sleep(2.0)
        
        # Get required data
        prices = await client.get_prices_snapshot(symbol)
        orderbooks = await client.get_orderbooks(symbol)
        trades = await client.get_trades(symbol)
        
        # Get current price
        current_price = 0.0
        for ex_data in prices.values():
            if symbol in ex_data:
                current_price = float(ex_data[symbol].get("price", 0))
                break
        
        # Compute features
        from src.analytics import OrderFlowAnalytics
        
        order_flow = OrderFlowAnalytics()
        
        imbalance = order_flow.compute_liquidity_imbalance(symbol, orderbooks)
        vacuum = order_flow.compute_liquidity_vacuum(symbol, orderbooks)
        persistence = order_flow.compute_orderbook_persistence(symbol, orderbooks)
        trade_agg = order_flow.compute_trade_aggression(symbol, trades.get("trades", []))
        efficiency = order_flow.compute_microstructure_efficiency(symbol, trades.get("trades", []), current_price)
        
        # Format support/resistance zones
        support_xml = ""
        for zone in persistence.get("support_zones", [])[:3]:
            support_xml += f"""
      <zone price="{zone.get('price', 0)}" reliability="{zone.get('reliability', 0):.0f}%" avg_size="{zone.get('avg_size', 0):.2f}" />"""
        
        resistance_xml = ""
        for zone in persistence.get("resistance_zones", [])[:3]:
            resistance_xml += f"""
      <zone price="{zone.get('price', 0)}" reliability="{zone.get('reliability', 0):.0f}%" avg_size="{zone.get('avg_size', 0):.2f}" />"""
        
        agg = imbalance.get("aggregated", {})
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<liquidity_analysis symbol="{symbol}">
  <imbalance>
    <ratio>{agg.get('imbalance_ratio', 0):.4f}</ratio>
    <signal>{agg.get('signal', 'NEUTRAL')}</signal>
    <depth_5_tick>{imbalance.get('depth_bands', {}).get('5_tick', {}).get('imbalance', 0):.4f}</depth_5_tick>
    <depth_20_tick>{imbalance.get('depth_bands', {}).get('20_tick', {}).get('imbalance', 0):.4f}</depth_20_tick>
    <depth_50_tick>{imbalance.get('depth_bands', {}).get('50_tick', {}).get('imbalance', 0):.4f}</depth_50_tick>
  </imbalance>
  
  <vacuum>
    <score>{vacuum.get('vacuum_score', 0):.1f}</score>
    <signal>{vacuum.get('signal', 'NORMAL')}</signal>
    <breakout_probability>{vacuum.get('breakout_probability', 0):.1f}%</breakout_probability>
    <thin_zones_count>{len(vacuum.get('thin_zones', []))}</thin_zones_count>
  </vacuum>
  
  <persistence>
    <support_reliability>{persistence.get('support_reliability', 0):.0f}%</support_reliability>
    <resistance_reliability>{persistence.get('resistance_reliability', 0):.0f}%</resistance_reliability>
    <spoof_probability>{persistence.get('spoof_probability', 0):.0f}%</spoof_probability>
    <support_zones>{support_xml}
    </support_zones>
    <resistance_zones>{resistance_xml}
    </resistance_zones>
  </persistence>
  
  <trade_aggression>
    <delta>{trade_agg.get('delta', 0):.2f}</delta>
    <delta_pct>{trade_agg.get('delta_pct', 0):.1f}%</delta_pct>
    <signal>{trade_agg.get('signal', 'NEUTRAL')}</signal>
    <large_trade_count>{trade_agg.get('large_trade_count', 0)}</large_trade_count>
    <absorption_detected>{trade_agg.get('absorption_detected', False)}</absorption_detected>
  </trade_aggression>
  
  <microstructure>
    <price_impact>{efficiency.get('price_impact', 0):.6f}</price_impact>
    <slippage_per_1m>{efficiency.get('slippage_per_1m', 0):.4f}</slippage_per_1m>
    <efficiency_score>{efficiency.get('efficiency_score', 0):.1f}</efficiency_score>
    <micro_volatility>{efficiency.get('micro_volatility', 0):.6f}</micro_volatility>
  </microstructure>
</liquidity_analysis>"""
        
    except Exception as e:
        logger.error(f"Error analyzing liquidity: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="LIQUIDITY_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_leverage_analysis(symbol: str = "BTCUSDT") -> str:
    """
    Analyze leverage, positioning, and risk flows.
    
    Computes OI flow decomposition, leverage index, liquidation pressure,
    funding stress, and basis regime.
    
    Args:
        symbol: Trading pair to analyze
    
    Returns:
        XML with leverage metrics, liquidation clusters, funding analysis.
    
    Example:
        "Analyze BTC leverage and risk" → get_leverage_analysis("BTCUSDT")
    """
    try:
        logger.info(f"Analyzing leverage for {symbol}")
        
        from src.storage.direct_exchange_client import get_direct_client
        client = get_direct_client()
        
        if not client._started:
            await client.connect()
            await asyncio.sleep(2.0)
        
        # Get required data
        prices = await client.get_prices_snapshot(symbol)
        trades = await client.get_trades(symbol)
        funding = await client.get_funding_rates(symbol)
        oi = await client.get_open_interest(symbol)
        liquidations = await client.get_liquidations(symbol)
        mark_prices = await client.get_mark_prices(symbol)
        
        # Get current price
        current_price = 0.0
        for ex_data in prices.values():
            if symbol in ex_data:
                current_price = float(ex_data[symbol].get("price", 0))
                break
        
        # Compute features
        from src.analytics import LeverageAnalytics
        
        leverage = LeverageAnalytics()
        
        oi_flow = leverage.compute_oi_flow_decomposition(symbol, oi, trades.get("trades", []))
        lev_index = leverage.compute_leverage_index(symbol, oi, trades.get("trades", []))
        liq_pressure = leverage.compute_liquidation_pressure(
            symbol, liquidations.get("liquidations", []), current_price
        )
        funding_stress = leverage.compute_funding_stress(symbol, funding)
        basis = leverage.compute_basis_regime(symbol, mark_prices, prices)
        
        # Format liquidation zones
        zones_xml = ""
        for zone in liq_pressure.get("pressure_zones", [])[:5]:
            zones_xml += f"""
      <zone price="{zone.get('price', 0)}" pct_from_current="{zone.get('pct_from_current', 0):.2f}%" value="{zone.get('total_value', 0):.0f}" dominant="{zone.get('dominant_side', 'N/A')}" />"""
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<leverage_analysis symbol="{symbol}">
  <oi_flow>
    <position_intent>{oi_flow.get('position_intent', 'UNKNOWN')}</position_intent>
    <position_intent_score>{oi_flow.get('position_intent_score', 0):.1f}</position_intent_score>
    <total_oi>{oi_flow.get('total_oi', 0):.0f}</total_oi>
    <oi_change_pct>{oi_flow.get('oi_change_pct', 0):.2f}%</oi_change_pct>
  </oi_flow>
  
  <leverage_index>
    <oi_volume_ratio>{lev_index.get('oi_volume_ratio', 0):.2f}</oi_volume_ratio>
    <leverage_zscore>{lev_index.get('leverage_zscore', 0):.2f}</leverage_zscore>
    <cascade_probability>{lev_index.get('cascade_probability', 0):.1f}%</cascade_probability>
    <signal>{lev_index.get('signal', 'NORMAL')}</signal>
  </leverage_index>
  
  <liquidation_pressure>
    <total_long_value>{liq_pressure.get('long_liquidation_value', 0):.0f}</total_long_value>
    <total_short_value>{liq_pressure.get('short_liquidation_value', 0):.0f}</total_short_value>
    <dominant_side>{liq_pressure.get('dominant_side', 'NEUTRAL')}</dominant_side>
    <cascade_risk>{liq_pressure.get('cascade_risk', 'LOW')}</cascade_risk>
    <pressure_zones>{zones_xml}
    </pressure_zones>
  </liquidation_pressure>
  
  <funding_stress>
    <avg_funding_rate>{funding_stress.get('avg_funding_rate', 0):.6f}</avg_funding_rate>
    <funding_zscore>{funding_stress.get('funding_zscore', 0):.2f}</funding_zscore>
    <carry_crowding_index>{funding_stress.get('carry_crowding_index', 0):.1f}</carry_crowding_index>
    <stress_level>{funding_stress.get('stress_level', 'NORMAL')}</stress_level>
    <sentiment>{funding_stress.get('sentiment', 'NEUTRAL')}</sentiment>
  </funding_stress>
  
  <basis_regime>
    <avg_basis_pct>{basis.get('avg_basis_pct', 0):.4f}%</avg_basis_pct>
    <regime>{basis.get('regime', 'UNKNOWN')}</regime>
    <signal>{basis.get('signal', 'NEUTRAL')}</signal>
  </basis_regime>
</leverage_analysis>"""
        
    except Exception as e:
        logger.error(f"Error analyzing leverage: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="LEVERAGE_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_cross_exchange_analysis(symbol: str = "BTCUSDT") -> str:
    """
    Analyze cross-exchange dynamics and arbitrage intelligence.
    
    Detects price leadership between exchanges, spread arbitrage opportunities,
    and flow synchronization patterns.
    
    Args:
        symbol: Trading pair to analyze
    
    Returns:
        XML with price leadership, arbitrage opportunities, flow sync.
    
    Example:
        "Analyze cross-exchange dynamics for BTC" → get_cross_exchange_analysis("BTCUSDT")
    """
    try:
        logger.info(f"Analyzing cross-exchange dynamics for {symbol}")
        
        from src.storage.direct_exchange_client import get_direct_client
        client = get_direct_client()
        
        if not client._started:
            await client.connect()
            await asyncio.sleep(2.0)
        
        # Get required data
        prices = await client.get_prices_snapshot(symbol)
        orderbooks = await client.get_orderbooks(symbol)
        trades = await client.get_trades(symbol)
        
        # Compute features
        from src.analytics import CrossExchangeAnalytics
        
        cross_exchange = CrossExchangeAnalytics()
        
        leadership = cross_exchange.compute_price_leadership(symbol, prices)
        arbitrage = cross_exchange.compute_spread_arbitrage(symbol, prices, orderbooks)
        flow_sync = cross_exchange.compute_flow_synchronization(symbol, trades.get("by_exchange", {}))
        
        # Format leadership
        ranking_xml = ""
        for ex in leadership.get("exchange_ranking", [])[:5]:
            ranking_xml += f"""
      <exchange name="{ex.get('exchange', '')}" leadership_score="{ex.get('leadership_score', 0):.1f}" avg_lead_ms="{ex.get('avg_lead_ms', 0):.0f}" />"""
        
        # Format opportunities
        opps_xml = ""
        for opp in arbitrage.get("opportunities", [])[:5]:
            opps_xml += f"""
      <opportunity buy="{opp.get('buy_exchange', '')}" sell="{opp.get('sell_exchange', '')}" spread_pct="{opp.get('spread_pct', 0):.4f}%" fill_probability="{opp.get('fill_probability', 0):.0f}%" />"""
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<cross_exchange_analysis symbol="{symbol}">
  <price_leadership>
    <leader>{leadership.get('leader', 'UNKNOWN')}</leader>
    <lead_time_ms>{leadership.get('lead_time_ms', 0):.0f}</lead_time_ms>
    <granger_signal>{leadership.get('granger_signal', 'NONE')}</granger_signal>
    <exchange_ranking>{ranking_xml}
    </exchange_ranking>
  </price_leadership>
  
  <spread_arbitrage>
    <max_spread_pct>{arbitrage.get('max_spread_pct', 0):.4f}%</max_spread_pct>
    <arbitrage_active>{arbitrage.get('arbitrage_active', False)}</arbitrage_active>
    <avg_fill_probability>{arbitrage.get('avg_fill_probability', 0):.0f}%</avg_fill_probability>
    <opportunities>{opps_xml}
    </opportunities>
  </spread_arbitrage>
  
  <flow_synchronization>
    <synchronization_score>{flow_sync.get('synchronization_score', 0):.1f}</synchronization_score>
    <dominant_direction>{flow_sync.get('dominant_direction', 'NEUTRAL')}</dominant_direction>
    <coordinated_flow_detected>{flow_sync.get('coordinated_flow_detected', False)}</coordinated_flow_detected>
    <signal>{flow_sync.get('signal', 'NEUTRAL')}</signal>
  </flow_synchronization>
</cross_exchange_analysis>"""
        
    except Exception as e:
        logger.error(f"Error analyzing cross-exchange: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="CROSS_EXCHANGE_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def get_analytics_summary() -> str:
    """
    Get summary of available analytics features and computation stats.
    
    Returns description of all analytics layers and recent computation statistics.
    
    Returns:
        XML with available features and engine statistics.
    
    Example:
        "What analytics are available?" → get_analytics_summary()
    """
    try:
        feature_engine = _get_feature_engine()
        stats = feature_engine.get_computation_stats()
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<analytics_summary>
  <engine_stats>
    <total_computations>{stats['total_computations']}</total_computations>
    <last_computation>{stats['last_computation'] or 'Never'}</last_computation>
    <avg_computation_time_ms>{stats['avg_computation_time_ms']}</avg_computation_time_ms>
    <symbols_tracked>{', '.join(stats['symbols_tracked']) or 'None'}</symbols_tracked>
  </engine_stats>
  
  <available_layers>
    <layer name="Order Flow &amp; Microstructure">
      <features>liquidity_imbalance, liquidity_vacuum, orderbook_persistence, trade_aggression, microstructure_efficiency</features>
      <tool>get_liquidity_analysis</tool>
    </layer>
    
    <layer name="Leverage &amp; Positioning">
      <features>oi_flow, leverage_index, liquidation_pressure, funding_stress, basis_regime</features>
      <tool>get_leverage_analysis</tool>
    </layer>
    
    <layer name="Cross-Exchange Intelligence">
      <features>price_leadership, spread_arbitrage, flow_synchronization</features>
      <tool>get_cross_exchange_analysis</tool>
    </layer>
    
    <layer name="Regime &amp; Volatility">
      <features>regime_detection, event_risk, volatility_state</features>
      <tool>get_market_regime</tool>
    </layer>
    
    <layer name="Alpha Signals">
      <features>institutional_pressure, squeeze_probability, smart_money_absorption, composite_signal</features>
      <tools>get_institutional_pressure, get_squeeze_probability, get_market_intelligence</tools>
    </layer>
  </available_layers>
  
  <usage_examples>
    <example>get_market_intelligence(symbol="BTCUSDT") - Full analysis</example>
    <example>get_institutional_pressure(symbol="ETHUSDT") - Pressure score</example>
    <example>get_squeeze_probability(symbol="BTCUSDT") - Squeeze risk</example>
    <example>get_market_regime(symbol="SOLUSDT") - Regime detection</example>
  </usage_examples>
</analytics_summary>"""
        
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="SUMMARY_FAILED"><message>{str(e)}</message></error>"""


# ============================================================================
# STREAMING ANALYSIS MCP TOOLS
# ============================================================================

# Lazy-loaded streaming analyzer (initialized on first use to improve startup time)
_streaming_analyzer = None

def _get_streaming_analyzer():
    """Get or create the streaming analyzer (lazy initialization)."""
    global _streaming_analyzer
    if _streaming_analyzer is None:
        from src.analytics.streaming_analyzer import StreamingAnalyzer
        _streaming_analyzer = StreamingAnalyzer()
    return _streaming_analyzer


@mcp.tool()
async def stream_and_analyze(
    symbol: str = "BTCUSDT",
    duration: int = 30
) -> str:
    """
    Stream market data for a specified time and perform comprehensive analysis.
    
    Collects real-time data (prices, orderbooks, trades, funding, liquidations)
    over the specified duration and computes analytics including:
    - Price movement analysis (direction, volatility, range)
    - Volume analysis (buy/sell ratio, large trades)
    - Orderbook analysis (depth, spread, imbalance)
    - Flow analysis (delta, CVD, aggressor detection)
    - Regime detection (breakout, consolidation, etc.)
    - Trading signals with confidence and recommendations
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT)
        duration: Analysis duration in seconds (5-300, default 30)
    
    Returns:
        XML with comprehensive analysis of the streaming period.
    
    Examples:
        "Analyze BTC for 30 seconds" → stream_and_analyze(symbol="BTCUSDT", duration=30)
        "Stream ETH data for 1 minute" → stream_and_analyze(symbol="ETHUSDT", duration=60)
        "Quick 10 second BTC analysis" → stream_and_analyze(symbol="BTCUSDT", duration=10)
    """
    try:
        # Validate duration
        duration = max(5, min(300, duration))
        
        logger.info(f"Starting {duration}s streaming analysis for {symbol}")
        
        from src.storage.direct_exchange_client import get_direct_client
        client = get_direct_client()
        
        if not client._started:
            await client.connect()
            await asyncio.sleep(2.0)
        
        # Run streaming analysis
        analyzer = _get_streaming_analyzer()
        result = await analyzer.analyze_stream(
            symbol=symbol,
            duration_seconds=duration,
            client=client
        )
        
        # Format as XML
        return _format_streaming_analysis_xml(result)
        
    except Exception as e:
        logger.error(f"Error in streaming analysis: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="STREAM_ANALYSIS_FAILED">
  <message>Failed to stream and analyze {symbol}</message>
  <details>{str(e)}</details>
</error>"""


@mcp.tool()
async def quick_analyze(symbol: str = "BTCUSDT") -> str:
    """
    Quick 10-second market snapshot analysis.
    
    Fast analysis that streams data for 10 seconds and provides:
    - Current price direction
    - Buy/sell flow
    - Orderbook imbalance
    - Quick trading signal
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT)
    
    Returns:
        XML with quick analysis results.
    
    Example:
        "Quick check on BTC" → quick_analyze(symbol="BTCUSDT")
    """
    try:
        logger.info(f"Starting quick 10s analysis for {symbol}")
        
        from src.storage.direct_exchange_client import get_direct_client
        client = get_direct_client()
        
        if not client._started:
            await client.connect()
            await asyncio.sleep(2.0)
        
        analyzer = _get_streaming_analyzer()
        result = await analyzer.analyze_stream(
            symbol=symbol,
            duration_seconds=10,
            client=client,
            sample_interval=0.5
        )
        
        # Simplified output
        price = result.get("price_analysis", {})
        flow = result.get("flow_analysis", {})
        signals = result.get("signals", {})
        regime = result.get("regime_analysis", {})
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<quick_analysis symbol="{symbol}" duration="10s">
  <price>
    <current>{price.get('end_price', 0)}</current>
    <change_pct>{price.get('change_pct', 0):.4f}%</change_pct>
    <direction>{price.get('direction', 'UNKNOWN')}</direction>
  </price>
  <flow>
    <delta_pct>{flow.get('delta_pct', 0):.2f}%</delta_pct>
    <aggressor>{flow.get('aggressor', 'UNKNOWN')}</aggressor>
    <signal>{flow.get('flow_signal', 'NEUTRAL')}</signal>
  </flow>
  <regime>{regime.get('detected_regime', 'UNKNOWN')}</regime>
  <signal>
    <bias>{signals.get('overall_bias', 'NEUTRAL')}</bias>
    <confidence>{signals.get('confidence', 50)}%</confidence>
    <recommendation>{signals.get('recommendation', 'N/A')}</recommendation>
  </signal>
</quick_analysis>"""
        
    except Exception as e:
        logger.error(f"Error in quick analysis: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="QUICK_ANALYSIS_FAILED"><message>{str(e)}</message></error>"""


@mcp.tool()
async def analyze_for_duration(
    symbol: str = "BTCUSDT",
    minutes: float = 1.0,
    focus: str = "all"
) -> str:
    """
    Analyze market for a specified number of minutes.
    
    Flexible duration analysis with focus options:
    - "all": Complete analysis (default)
    - "price": Focus on price movement
    - "flow": Focus on order flow and delta
    - "regime": Focus on market regime detection
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT)
        minutes: Duration in minutes (0.1 to 5.0)
        focus: Analysis focus - "all", "price", "flow", or "regime"
    
    Returns:
        XML with analysis based on specified focus.
    
    Examples:
        "Analyze BTC for 2 minutes" → analyze_for_duration(symbol="BTCUSDT", minutes=2.0)
        "1 minute flow analysis on ETH" → analyze_for_duration(symbol="ETHUSDT", minutes=1.0, focus="flow")
    """
    try:
        # Convert minutes to seconds
        duration = int(max(5, min(300, minutes * 60)))
        
        logger.info(f"Starting {minutes}min ({duration}s) analysis for {symbol}, focus={focus}")
        
        from src.storage.direct_exchange_client import get_direct_client
        client = get_direct_client()
        
        if not client._started:
            await client.connect()
            await asyncio.sleep(2.0)
        
        analyzer = _get_streaming_analyzer()
        result = await analyzer.analyze_stream(
            symbol=symbol,
            duration_seconds=duration,
            client=client
        )
        
        # Format based on focus
        if focus == "price":
            return _format_price_focus_xml(result)
        elif focus == "flow":
            return _format_flow_focus_xml(result)
        elif focus == "regime":
            return _format_regime_focus_xml(result)
        else:
            return _format_streaming_analysis_xml(result)
        
    except Exception as e:
        logger.error(f"Error in duration analysis: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="DURATION_ANALYSIS_FAILED"><message>{str(e)}</message></error>"""


def _format_streaming_analysis_xml(result: Dict) -> str:
    """Format full streaming analysis as XML."""
    symbol = result.get("symbol", "UNKNOWN")
    duration = result.get("duration_seconds", 0)
    samples = result.get("samples_collected", 0)
    
    price = result.get("price_analysis", {})
    volume = result.get("volume_analysis", {})
    orderbook = result.get("orderbook_analysis", {})
    funding = result.get("funding_analysis", {})
    liquidations = result.get("liquidation_analysis", {})
    flow = result.get("flow_analysis", {})
    regime = result.get("regime_analysis", {})
    signals = result.get("signals", {})
    data_summary = result.get("data_summary", {})
    errors = result.get("errors", [])
    
    # Format errors
    errors_xml = ""
    if errors:
        errors_xml = f"""
  <errors count="{len(errors)}">"""
        for err in errors[:5]:
            errors_xml += f"""
    <error>{err}</error>"""
        errors_xml += """
  </errors>"""
    
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<streaming_analysis symbol="{symbol}">
  <metadata>
    <duration_seconds>{duration}</duration_seconds>
    <samples_collected>{samples}</samples_collected>
    <analysis_start>{result.get('analysis_start', '')}</analysis_start>
    <analysis_end>{result.get('analysis_end', '')}</analysis_end>
  </metadata>
  
  <data_collected>
    <price_snapshots>{data_summary.get('price_snapshots', 0)}</price_snapshots>
    <orderbook_snapshots>{data_summary.get('orderbook_snapshots', 0)}</orderbook_snapshots>
    <trades>{data_summary.get('trades_collected', 0)}</trades>
    <liquidations>{data_summary.get('liquidations_collected', 0)}</liquidations>
  </data_collected>
  
  <price_analysis>
    <start_price>{price.get('start_price', 0)}</start_price>
    <end_price>{price.get('end_price', 0)}</end_price>
    <high>{price.get('high', 0)}</high>
    <low>{price.get('low', 0)}</low>
    <change_pct>{price.get('change_pct', 0):.4f}%</change_pct>
    <range_pct>{price.get('range_pct', 0):.4f}%</range_pct>
    <direction>{price.get('direction', 'UNKNOWN')}</direction>
    <volatility>{price.get('volatility', 0):.4f}</volatility>
  </price_analysis>
  
  <volume_analysis>
    <total_volume>{volume.get('total_volume', 0):.4f}</total_volume>
    <total_value_usd>{volume.get('total_value_usd', 0):.2f}</total_value_usd>
    <total_trades>{volume.get('total_trades', 0)}</total_trades>
    <buy_volume>{volume.get('buy_volume', 0):.4f}</buy_volume>
    <sell_volume>{volume.get('sell_volume', 0):.4f}</sell_volume>
    <buy_sell_ratio>{volume.get('buy_sell_ratio', 0):.4f}</buy_sell_ratio>
    <volume_imbalance>{volume.get('volume_imbalance', 0):.2f}%</volume_imbalance>
    <large_trade_count>{volume.get('large_trade_count', 0)}</large_trade_count>
  </volume_analysis>
  
  <orderbook_analysis>
    <avg_spread_pct>{orderbook.get('avg_spread_pct', 0):.6f}%</avg_spread_pct>
    <bid_depth_usd>{orderbook.get('bid_depth_usd', 0):.2f}</bid_depth_usd>
    <ask_depth_usd>{orderbook.get('ask_depth_usd', 0):.2f}</ask_depth_usd>
    <depth_imbalance>{orderbook.get('depth_imbalance', 0):.4f}</depth_imbalance>
    <imbalance_signal>{orderbook.get('imbalance_signal', 'NEUTRAL')}</imbalance_signal>
  </orderbook_analysis>
  
  <flow_analysis>
    <delta>{flow.get('delta', 0):.4f}</delta>
    <delta_pct>{flow.get('delta_pct', 0):.2f}%</delta_pct>
    <cvd_final>{flow.get('cvd_final', 0):.4f}</cvd_final>
    <cvd_trend>{flow.get('cvd_trend', 'FLAT')}</cvd_trend>
    <aggressor>{flow.get('aggressor', 'BALANCED')}</aggressor>
    <flow_signal>{flow.get('flow_signal', 'NEUTRAL')}</flow_signal>
  </flow_analysis>
  
  <funding_analysis>
    <avg_rate>{funding.get('avg_funding_rate', 0):.8f}</avg_rate>
    <annualized_pct>{funding.get('annualized_rate', 0):.2f}%</annualized_pct>
    <sentiment>{funding.get('sentiment', 'NEUTRAL')}</sentiment>
  </funding_analysis>
  
  <liquidation_analysis>
    <total_liquidations>{liquidations.get('total_liquidations', 0)}</total_liquidations>
    <long_liquidations>{liquidations.get('long_liquidations', 0)}</long_liquidations>
    <short_liquidations>{liquidations.get('short_liquidations', 0)}</short_liquidations>
    <total_value_usd>{liquidations.get('total_value_usd', 0):.2f}</total_value_usd>
    <dominant_side>{liquidations.get('dominant_side', 'BALANCED')}</dominant_side>
    <cascade_risk>{liquidations.get('cascade_risk', 'LOW')}</cascade_risk>
  </liquidation_analysis>
  
  <regime_analysis>
    <detected_regime>{regime.get('detected_regime', 'UNKNOWN')}</detected_regime>
    <description>{regime.get('description', '')}</description>
    <price_direction>{regime.get('price_direction', 'UNKNOWN')}</price_direction>
    <flow_direction>{regime.get('flow_direction', 'NEUTRAL')}</flow_direction>
    <volatility_state>{regime.get('volatility_state', 'NORMAL')}</volatility_state>
  </regime_analysis>
  
  <signals>
    <overall_bias>{signals.get('overall_bias', 'NEUTRAL')}</overall_bias>
    <confidence>{signals.get('confidence', 50)}%</confidence>
    <active_signals>{', '.join(signals.get('active_signals', []))}</active_signals>
    <recommendation>{signals.get('recommendation', 'N/A')}</recommendation>
  </signals>
{errors_xml}
</streaming_analysis>"""


def _format_price_focus_xml(result: Dict) -> str:
    """Format price-focused analysis."""
    symbol = result.get("symbol", "UNKNOWN")
    price = result.get("price_analysis", {})
    regime = result.get("regime_analysis", {})
    
    exchange_xml = ""
    for ex, stats in price.get("by_exchange", {}).items():
        exchange_xml += f"""
    <exchange name="{ex}">
      <start>{stats.get('start_price', 0)}</start>
      <end>{stats.get('end_price', 0)}</end>
      <change_pct>{stats.get('change_pct', 0):.4f}%</change_pct>
      <volatility>{stats.get('volatility', 0):.4f}</volatility>
    </exchange>"""
    
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<price_analysis symbol="{symbol}" duration="{result.get('duration_seconds', 0)}s">
  <summary>
    <start_price>{price.get('start_price', 0)}</start_price>
    <end_price>{price.get('end_price', 0)}</end_price>
    <high>{price.get('high', 0)}</high>
    <low>{price.get('low', 0)}</low>
    <change_pct>{price.get('change_pct', 0):.4f}%</change_pct>
    <range_pct>{price.get('range_pct', 0):.4f}%</range_pct>
    <direction>{price.get('direction', 'UNKNOWN')}</direction>
    <volatility>{price.get('volatility', 0):.4f}</volatility>
    <samples>{price.get('price_samples', 0)}</samples>
  </summary>
  <by_exchange>{exchange_xml}
  </by_exchange>
  <regime>{regime.get('detected_regime', 'UNKNOWN')}</regime>
</price_analysis>"""


def _format_flow_focus_xml(result: Dict) -> str:
    """Format flow-focused analysis."""
    symbol = result.get("symbol", "UNKNOWN")
    flow = result.get("flow_analysis", {})
    volume = result.get("volume_analysis", {})
    signals = result.get("signals", {})
    
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<flow_analysis symbol="{symbol}" duration="{result.get('duration_seconds', 0)}s">
  <order_flow>
    <delta>{flow.get('delta', 0):.4f}</delta>
    <delta_pct>{flow.get('delta_pct', 0):.2f}%</delta_pct>
    <cvd_final>{flow.get('cvd_final', 0):.4f}</cvd_final>
    <cvd_trend>{flow.get('cvd_trend', 'FLAT')}</cvd_trend>
    <aggressor>{flow.get('aggressor', 'BALANCED')}</aggressor>
    <flow_signal>{flow.get('flow_signal', 'NEUTRAL')}</flow_signal>
  </order_flow>
  <volume>
    <buy_volume>{volume.get('buy_volume', 0):.4f}</buy_volume>
    <sell_volume>{volume.get('sell_volume', 0):.4f}</sell_volume>
    <buy_sell_ratio>{volume.get('buy_sell_ratio', 0):.4f}</buy_sell_ratio>
    <volume_imbalance>{volume.get('volume_imbalance', 0):.2f}%</volume_imbalance>
    <large_trades>{volume.get('large_trade_count', 0)}</large_trades>
  </volume>
  <signal>
    <bias>{signals.get('overall_bias', 'NEUTRAL')}</bias>
    <confidence>{signals.get('confidence', 50)}%</confidence>
    <recommendation>{signals.get('recommendation', 'N/A')}</recommendation>
  </signal>
</flow_analysis>"""


def _format_regime_focus_xml(result: Dict) -> str:
    """Format regime-focused analysis."""
    symbol = result.get("symbol", "UNKNOWN")
    regime = result.get("regime_analysis", {})
    price = result.get("price_analysis", {})
    flow = result.get("flow_analysis", {})
    liquidations = result.get("liquidation_analysis", {})
    signals = result.get("signals", {})
    
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<regime_analysis symbol="{symbol}" duration="{result.get('duration_seconds', 0)}s">
  <regime>
    <detected>{regime.get('detected_regime', 'UNKNOWN')}</detected>
    <description>{regime.get('description', '')}</description>
    <volatility_state>{regime.get('volatility_state', 'NORMAL')}</volatility_state>
  </regime>
  <supporting_data>
    <price_direction>{price.get('direction', 'UNKNOWN')}</price_direction>
    <price_change>{price.get('change_pct', 0):.4f}%</price_change>
    <flow_direction>{flow.get('flow_signal', 'NEUTRAL')}</flow_direction>
    <delta_pct>{flow.get('delta_pct', 0):.2f}%</delta_pct>
    <liquidations>{liquidations.get('total_liquidations', 0)}</liquidations>
  </supporting_data>
  <signal>
    <bias>{signals.get('overall_bias', 'NEUTRAL')}</bias>
    <confidence>{signals.get('confidence', 50)}%</confidence>
    <recommendation>{signals.get('recommendation', 'N/A')}</recommendation>
  </signal>
</regime_analysis>"""


def _format_intelligence_xml(result: dict, layers: str) -> str:
    """Format full intelligence result as XML."""
    import json
    
    symbol = result.get("symbol", "UNKNOWN")
    timestamp = result.get("timestamp", "")
    computation_time = result.get("computation_time_ms", 0)
    errors = result.get("errors", [])
    summary = result.get("summary", {})
    layer_data = result.get("layers", {})
    
    # Summary section
    summary_xml = f"""
  <summary>
    <market_bias>{summary.get('market_bias', 'NEUTRAL')}</market_bias>
    <market_condition>{summary.get('market_condition', 'NORMAL')}</market_condition>
    <risk_level>{summary.get('risk_level', 'LOW')}</risk_level>
    <scores>
      <institutional_pressure>{summary.get('scores', {}).get('institutional_pressure', 0)}</institutional_pressure>
      <squeeze_probability>{summary.get('scores', {}).get('squeeze_probability', 0)}</squeeze_probability>
      <absorption_strength>{summary.get('scores', {}).get('absorption_strength', 0)}</absorption_strength>
      <event_risk_score>{summary.get('scores', {}).get('event_risk_score', 0)}</event_risk_score>
    </scores>
  </summary>"""
    
    # Key signals
    signals_xml = ""
    for sig in summary.get("key_signals", [])[:5]:
        signals_xml += f"""
    <signal type="{sig.get('type', '')}">{json.dumps(sig)}</signal>"""
    
    # Warnings
    warnings_xml = ""
    for warn in summary.get("warnings", [])[:5]:
        warnings_xml += f"""
    <warning>{warn}</warning>"""
    
    # Composite signal
    composite = layer_data.get("alpha_signals", {}).get("composite_signal", {})
    composite_xml = f"""
  <composite_signal>
    <signal>{composite.get('signal', 'NEUTRAL')}</signal>
    <confidence>{composite.get('confidence', 0)}</confidence>
    <urgency>{composite.get('urgency', 'LOW')}</urgency>
    <risk_reward>{composite.get('risk_reward', 'UNKNOWN')}</risk_reward>
    <action_plan>{composite.get('action_plan', '')}</action_plan>
  </composite_signal>"""
    
    # Errors section
    errors_xml = ""
    if errors:
        errors_xml = f"""
  <errors count="{len(errors)}">"""
        for err in errors[:10]:
            errors_xml += f"""
    <error>{err}</error>"""
        errors_xml += """
  </errors>"""
    
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<market_intelligence symbol="{symbol}" timestamp="{timestamp}" computation_time_ms="{computation_time}">
{summary_xml}
  
  <key_signals>{signals_xml}
  </key_signals>
  
  <active_warnings>{warnings_xml}
  </active_warnings>
{composite_xml}
{errors_xml}
</market_intelligence>"""


# ============================================================================
# BINANCE FUTURES REST API MCP TOOLS
# ============================================================================

@mcp.tool()
async def binance_ticker_tool(symbol: str = None) -> str:
    """
    Get Binance Futures 24h ticker statistics via REST API.
    
    Returns comprehensive 24h statistics including price, volume, and trade count
    directly from Binance Futures REST API.
    
    Args:
        symbol: Specific symbol (e.g., "BTCUSDT") or leave empty for all major pairs
    
    Returns:
        JSON with 24h ticker data including price changes, volume, high/low
    """
    import json
    try:
        result = await binance_get_ticker(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_orderbook_tool(symbol: str = "BTCUSDT", depth: int = 100) -> str:
    """
    Get Binance Futures orderbook depth via REST API (up to 1000 levels).
    
    Retrieves full orderbook with bid/ask analysis including:
    - All price levels with quantities
    - Spread analysis
    - Depth imbalance metrics
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT)
        depth: Number of levels (5, 10, 20, 50, 100, 500, 1000)
    
    Returns:
        JSON with orderbook data and liquidity metrics
    """
    import json
    try:
        result = await binance_get_orderbook(symbol, depth)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_trades_tool(symbol: str = "BTCUSDT", limit: int = 500) -> str:
    """
    Get recent aggregated trades from Binance Futures via REST API.
    
    Retrieves recent trades with analysis including:
    - Buy/sell volume breakdown
    - VWAP calculation
    - Large trade detection
    
    Args:
        symbol: Trading pair
        limit: Number of trades (max 1000)
    
    Returns:
        JSON with trade data and volume analysis
    """
    import json
    try:
        result = await binance_get_trades(symbol, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_klines_tool(symbol: str = "BTCUSDT", interval: str = "1m", 
                              limit: int = 200) -> str:
    """
    Get OHLCV candlestick data from Binance Futures via REST API.
    
    Retrieves historical candles with technical analysis including:
    - Full OHLCV data
    - SMA calculations
    - Volume analysis
    - Trend detection
    
    Args:
        symbol: Trading pair
        interval: Kline interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)
        limit: Number of candles (max 1500)
    
    Returns:
        JSON with candle data and technical summary
    """
    import json
    try:
        result = await binance_get_klines(symbol, interval, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_open_interest_tool(symbol: str = None) -> str:
    """
    Get current open interest from Binance Futures via REST API.
    
    Open interest = total number of outstanding derivative contracts.
    Rising OI + price up = bullish, Rising OI + price down = bearish.
    
    Args:
        symbol: Specific symbol or leave empty for all major pairs
    
    Returns:
        JSON with open interest data
    """
    import json
    try:
        result = await binance_get_open_interest(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_open_interest_history_tool(symbol: str = "BTCUSDT", 
                                             period: str = "5m",
                                             limit: int = 200) -> str:
    """
    Get historical open interest from Binance Futures via REST API.
    
    Shows how open interest has changed over time.
    Useful for identifying accumulation/distribution phases.
    
    Args:
        symbol: Trading pair
        period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
        limit: Number of data points (max 500)
    
    Returns:
        JSON with historical OI and trend analysis
    """
    import json
    try:
        result = await binance_get_open_interest_history(symbol, period, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_funding_rate_tool(symbol: str = "BTCUSDT", limit: int = 100) -> str:
    """
    Get historical funding rates from Binance Futures via REST API.
    
    Funding rate is paid between longs and shorts every 8 hours.
    - Positive = longs pay shorts (market is long-heavy)
    - Negative = shorts pay longs (market is short-heavy)
    
    Args:
        symbol: Trading pair
        limit: Number of funding periods (max 1000)
    
    Returns:
        JSON with funding rate history and sentiment analysis
    """
    import json
    try:
        result = await binance_get_funding_rate(symbol, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_premium_index_tool(symbol: str = None) -> str:
    """
    Get premium index / mark price from Binance Futures via REST API.
    
    Includes:
    - Mark price (used for liquidations)
    - Index price (spot aggregate)
    - Basis (futures - spot premium/discount)
    - Current funding rate
    
    Args:
        symbol: Specific symbol or leave empty for all
    
    Returns:
        JSON with premium index and basis data
    """
    import json
    try:
        result = await binance_get_premium_index(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_long_short_ratio_tool(symbol: str = "BTCUSDT", 
                                        period: str = "5m",
                                        limit: int = 100) -> str:
    """
    Get comprehensive long/short ratio data from Binance Futures via REST API.
    
    Fetches multiple positioning metrics:
    - Top trader account ratio (how many traders are long vs short)
    - Top trader position ratio (position sizes)
    - Global account ratio (all traders)
    - Taker buy/sell ratio (aggressor flow)
    
    Args:
        symbol: Trading pair
        period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
        limit: Number of data points (max 500)
    
    Returns:
        JSON with comprehensive positioning data
    """
    import json
    try:
        result = await binance_get_long_short_ratio(symbol, period, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_taker_volume_tool(symbol: str = "BTCUSDT", 
                                    period: str = "5m",
                                    limit: int = 100) -> str:
    """
    Get taker buy/sell volume ratio from Binance Futures via REST API.
    
    Taker volume shows who is crossing the spread (aggressors):
    - Ratio > 1 = more aggressive buying
    - Ratio < 1 = more aggressive selling
    
    Critical for detecting momentum and order flow.
    
    Args:
        symbol: Trading pair
        period: Time period
        limit: Number of data points
    
    Returns:
        JSON with taker volume analysis
    """
    import json
    try:
        result = await binance_get_taker_volume(symbol, period, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_basis_tool(symbol: str = "BTCUSDT", 
                             period: str = "5m",
                             limit: int = 200) -> str:
    """
    Get futures basis data from Binance via REST API.
    
    Basis = Futures Price - Index/Spot Price
    - Positive = Contango (futures premium)
    - Negative = Backwardation (futures discount)
    
    Annualized basis is used for carry trade calculations.
    
    Args:
        symbol: Trading pair
        period: Time period
        limit: Number of data points
    
    Returns:
        JSON with basis analysis
    """
    import json
    try:
        result = await binance_get_basis(symbol, period, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_liquidations_tool(symbol: str = None, limit: int = 100) -> str:
    """
    Get recent liquidation orders from Binance Futures via REST API.
    
    Liquidations occur when traders get margin called:
    - SELL liquidations = longs getting liquidated
    - BUY liquidations = shorts getting liquidated
    
    Large liquidation clusters can indicate support/resistance levels.
    
    Args:
        symbol: Specific symbol or leave empty for all
        limit: Number of liquidations (max 1000)
    
    Returns:
        JSON with liquidation data and analysis
    """
    import json
    try:
        result = await binance_get_liquidations(symbol, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_snapshot_tool(symbol: str = "BTCUSDT") -> str:
    """
    Get a comprehensive market snapshot from Binance Futures via REST API.
    
    Single call that fetches:
    - Current prices (mark, index, last)
    - 24h statistics
    - Open interest
    - Funding rate
    - Best bid/ask spread
    
    Args:
        symbol: Trading pair
    
    Returns:
        JSON with complete market snapshot
    """
    import json
    try:
        result = await binance_market_snapshot(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_full_analysis_tool(symbol: str = "BTCUSDT") -> str:
    """
    Get complete market analysis from Binance Futures via REST API.
    
    Comprehensive analysis including:
    - Market snapshot (prices, volume, OI)
    - Positioning data (long/short ratios)
    - Historical analysis (OI trend, funding trend, basis)
    - Trading signals based on the data
    
    This is the most comprehensive single-call analysis available.
    
    Args:
        symbol: Trading pair
    
    Returns:
        JSON with full analysis and trading signals
    """
    import json
    try:
        result = await binance_full_analysis(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# BYBIT REST API MCP TOOLS - SPOT MARKET
# ============================================================================

@mcp.tool()
async def bybit_spot_ticker_tool(symbol: str = "BTCUSDT") -> str:
    """
    Get Bybit spot market ticker data.
    
    Returns current price, 24h change, volume, and spread info for spot trading.
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT, ETHUSDT)
    
    Returns:
        JSON with spot ticker data including price, volume, 24h stats
    """
    import json
    try:
        result = await bybit_spot_ticker(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def bybit_spot_orderbook_tool(symbol: str = "BTCUSDT", depth: int = 50) -> str:
    """
    Get Bybit spot orderbook depth.
    
    Returns bid/ask levels with liquidity analysis for spot market.
    
    Args:
        symbol: Trading pair
        depth: Number of levels (1-200)
    
    Returns:
        JSON with orderbook data, spread, and imbalance metrics
    """
    import json
    try:
        result = await bybit_spot_orderbook(symbol, depth)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def bybit_spot_trades_tool(symbol: str = "BTCUSDT", limit: int = 100) -> str:
    """
    Get recent Bybit spot trades.
    
    Analyzes recent trade flow to determine buying/selling pressure.
    
    Args:
        symbol: Trading pair
        limit: Number of trades (max 1000)
    
    Returns:
        JSON with trades analysis including buy/sell ratio and large trade detection
    """
    import json
    try:
        result = await bybit_spot_trades(symbol, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def bybit_spot_klines_tool(symbol: str = "BTCUSDT", 
                                  interval: str = "60", 
                                  limit: int = 100) -> str:
    """
    Get Bybit spot klines/candlesticks.
    
    OHLCV data with trend analysis for spot market.
    
    Args:
        symbol: Trading pair
        interval: Kline interval (1,3,5,15,30,60,120,240,360,720,D,W,M)
        limit: Number of klines (max 1000)
    
    Returns:
        JSON with OHLCV data and trend analysis
    """
    import json
    try:
        result = await bybit_spot_klines(symbol, interval, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def bybit_all_spot_tickers_tool() -> str:
    """
    Get all Bybit spot market tickers.
    
    Returns summary of all spot pairs including top gainers, losers, and volume leaders.
    
    Returns:
        JSON with all spot tickers and market summary
    """
    import json
    try:
        result = await bybit_all_spot_tickers()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# BYBIT REST API MCP TOOLS - FUTURES/LINEAR PERPETUAL
# ============================================================================

@mcp.tool()
async def bybit_futures_ticker_tool(symbol: str = "BTCUSDT") -> str:
    """
    Get Bybit USDT perpetual futures ticker.
    
    Comprehensive futures data including:
    - Price, mark price, index price
    - Funding rate and next funding time
    - Open interest and value
    - Basis/premium calculation
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT)
    
    Returns:
        JSON with futures ticker and derivatives metrics
    """
    import json
    try:
        result = await bybit_futures_ticker(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def bybit_futures_orderbook_tool(symbol: str = "BTCUSDT", depth: int = 100) -> str:
    """
    Get Bybit futures orderbook depth (up to 500 levels).
    
    Includes bid/ask imbalance, spread analysis, and wall detection.
    
    Args:
        symbol: Trading pair
        depth: Number of levels (1-500)
    
    Returns:
        JSON with orderbook data and liquidity analysis
    """
    import json
    try:
        result = await bybit_futures_orderbook(symbol, depth)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def bybit_open_interest_tool(symbol: str = "BTCUSDT", 
                                    interval: str = "1h", 
                                    limit: int = 48) -> str:
    """
    Get Bybit open interest history.
    
    OI analysis for understanding position building/unwinding:
    - Rising OI + Rising Price = New longs entering (bullish)
    - Rising OI + Falling Price = New shorts entering (bearish)
    - Falling OI = Positions closing
    
    Args:
        symbol: Trading pair
        interval: Data interval (5min, 15min, 30min, 1h, 4h, 1d)
        limit: Number of data points (max 200)
    
    Returns:
        JSON with OI history and trend analysis
    """
    import json
    try:
        result = await bybit_open_interest(symbol, interval, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def bybit_funding_rate_tool(symbol: str = "BTCUSDT", limit: int = 50) -> str:
    """
    Get Bybit funding rate history.
    
    Funding rate indicates market sentiment:
    - Positive = Longs pay shorts (bullish positioning)
    - Negative = Shorts pay longs (bearish positioning)
    - Extreme funding often precedes reversals
    
    Args:
        symbol: Trading pair
        limit: Number of funding periods (max 200)
    
    Returns:
        JSON with funding rate history, annualized rate, and sentiment
    """
    import json
    try:
        result = await bybit_funding_rate(symbol, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def bybit_long_short_ratio_tool(symbol: str = "BTCUSDT", 
                                       period: str = "1h", 
                                       limit: int = 24) -> str:
    """
    Get Bybit long/short account ratio.
    
    Shows the distribution of longs vs shorts:
    - Ratio > 1.5 = Heavy long positioning (contrarian short signal?)
    - Ratio < 0.67 = Heavy short positioning (contrarian long signal?)
    
    Args:
        symbol: Trading pair
        period: Data interval (5min, 15min, 30min, 1h, 4h, 1d)
        limit: Number of data points (max 500)
    
    Returns:
        JSON with L/S ratio history and positioning sentiment
    """
    import json
    try:
        result = await bybit_long_short_ratio(symbol, period, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def bybit_historical_volatility_tool(base_coin: str = "BTC", period: int = 30) -> str:
    """
    Get Bybit historical volatility (options market data).
    
    Historical volatility for volatility regime analysis:
    - High HV = Volatile market, use wider stops
    - Low HV = Compression, breakout expected
    
    Args:
        base_coin: Base coin (BTC, ETH)
        period: Period in days (7, 14, 21, 30, 60, 90, 180, 270)
    
    Returns:
        JSON with historical volatility and regime classification
    """
    import json
    try:
        result = await bybit_historical_volatility(base_coin, period)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def bybit_insurance_fund_tool(coin: str = "USDT") -> str:
    """
    Get Bybit insurance fund balance.
    
    Insurance fund protects traders from auto-deleveraging (ADL).
    Healthy fund = lower ADL risk.
    
    Args:
        coin: Coin to check (USDT, BTC, etc.)
    
    Returns:
        JSON with insurance fund data and health status
    """
    import json
    try:
        result = await bybit_insurance_fund(coin)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def bybit_all_perpetual_tickers_tool() -> str:
    """
    Get all Bybit USDT perpetual tickers.
    
    Returns market overview with:
    - Top gainers and losers
    - Highest open interest
    - Highest volume
    - Highest/most extreme funding rates
    
    Returns:
        JSON with all perpetual tickers and market summary
    """
    import json
    try:
        result = await bybit_all_perpetual_tickers()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# BYBIT REST API MCP TOOLS - ANALYSIS & OPTIONS
# ============================================================================

@mcp.tool()
async def bybit_derivatives_analysis_tool(symbol: str = "BTCUSDT") -> str:
    """
    Get comprehensive Bybit derivatives analysis.
    
    Combines multiple data sources:
    - OI trend and position flow
    - Funding rate sentiment
    - Long/short positioning
    - Price and volume metrics
    - Generated trading signals
    
    Args:
        symbol: Trading pair
    
    Returns:
        JSON with full derivatives analysis and signals
    """
    import json
    try:
        result = await bybit_derivatives_analysis(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def bybit_market_snapshot_tool(symbol: str = "BTCUSDT") -> str:
    """
    Get comprehensive Bybit market snapshot.
    
    Single call for complete market view:
    - Ticker data
    - Orderbook (top levels)
    - Recent trades
    - Open interest
    - Funding history
    
    Args:
        symbol: Trading pair
    
    Returns:
        JSON with complete market snapshot
    """
    import json
    try:
        result = await bybit_market_snapshot(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def bybit_instruments_info_tool(category: str = "linear", 
                                       symbol: str = None) -> str:
    """
    Get Bybit instrument specifications.
    
    Returns contract specifications including:
    - Lot size, tick size, min notional
    - Leverage limits
    - Funding interval
    - Contract type
    
    Args:
        category: Market category (spot, linear, inverse, option)
        symbol: Specific symbol (optional, returns all if empty)
    
    Returns:
        JSON with instrument specifications
    """
    import json
    try:
        result = await bybit_instruments_info(category, symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def bybit_options_overview_tool(base_coin: str = "BTC") -> str:
    """
    Get Bybit options market overview.
    
    Options market data including:
    - Options tickers
    - Historical volatility
    - Delivery prices
    
    Args:
        base_coin: Base coin (BTC, ETH)
    
    Returns:
        JSON with options market overview
    """
    import json
    try:
        result = await bybit_options_overview(base_coin)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def bybit_risk_limit_tool(category: str = "linear", symbol: str = None) -> str:
    """
    Get Bybit risk limit tiers.
    
    Risk limits define max position size at each leverage level:
    - Higher tiers = larger positions but lower max leverage
    - Important for position sizing
    
    Args:
        category: Market category (linear, inverse)
        symbol: Specific symbol (optional)
    
    Returns:
        JSON with risk limit tiers
    """
    import json
    try:
        result = await bybit_risk_limit(category, symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def bybit_announcements_tool(locale: str = "en-US", limit: int = 10) -> str:
    """
    Get Bybit platform announcements.
    
    Stay informed about:
    - New listings
    - Maintenance schedules
    - Product updates
    - Promotions
    
    Args:
        locale: Language (en-US, zh-CN, etc.)
        limit: Number of announcements
    
    Returns:
        JSON with recent announcements
    """
    import json
    try:
        result = await bybit_announcements(locale, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def bybit_full_market_analysis_tool(symbol: str = "BTCUSDT") -> str:
    """
    Get comprehensive Bybit full market analysis.
    
    The most comprehensive single-call analysis for Bybit:
    - Spot data
    - Futures data with funding/OI
    - Positioning data (long/short ratio)
    - Combined signals and analysis
    - Trading recommendations
    
    Args:
        symbol: Trading pair
    
    Returns:
        JSON with full market analysis and trading signals
    """
    import json
    try:
        result = await bybit_full_market_analysis(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# BINANCE SPOT REST API MCP TOOLS
# ============================================================================

@mcp.tool()
async def binance_spot_ticker_tool(symbol: str = "BTCUSDT") -> str:
    """
    Get Binance spot 24hr ticker statistics.
    
    Provides comprehensive 24-hour statistics including:
    - Current price and 24h change
    - High, low, and weighted average price
    - Volume and trade count
    - Bid/ask prices and spread
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT, ETHUSDT)
    
    Returns:
        JSON with complete 24h ticker statistics
    """
    import json
    try:
        result = await binance_spot_ticker(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_spot_price_tool(symbol: str = None) -> str:
    """
    Get current Binance spot price(s).
    
    Returns instant price quotes for spot markets.
    If no symbol provided, returns major pairs.
    
    Args:
        symbol: Trading pair or None for major pairs (BTC, ETH, SOL, XRP, BNB)
    
    Returns:
        JSON with current price data
    """
    import json
    try:
        result = await binance_spot_price(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_spot_orderbook_tool(symbol: str = "BTCUSDT", depth: int = 100) -> str:
    """
    Get Binance spot orderbook depth.
    
    Returns bid/ask levels with:
    - Best bid/ask and spread analysis
    - Volume imbalance calculation
    - Major bid/ask walls detection
    
    Args:
        symbol: Trading pair
        depth: Number of levels (5, 10, 20, 50, 100, 500, 1000, 5000)
    
    Returns:
        JSON with orderbook and liquidity analysis
    """
    import json
    try:
        result = await binance_spot_orderbook(symbol, depth)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_spot_trades_tool(symbol: str = "BTCUSDT", limit: int = 500) -> str:
    """
    Get recent Binance spot trades with flow analysis.
    
    Returns recent trade data with:
    - Buy/sell volume breakdown
    - Large trade detection
    - Trade flow direction signal
    
    Args:
        symbol: Trading pair
        limit: Number of trades (max 1000)
    
    Returns:
        JSON with trades and flow analysis
    """
    import json
    try:
        result = await binance_spot_trades(symbol, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_spot_klines_tool(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 100
) -> str:
    """
    Get Binance spot klines/candlestick data.
    
    Returns OHLCV data with analysis:
    - Price trend detection
    - Volume trend analysis
    - Period high/low range
    
    Args:
        symbol: Trading pair
        interval: Timeframe (1s,1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M)
        limit: Number of candles (max 1000)
    
    Returns:
        JSON with OHLCV data and trend analysis
    """
    import json
    try:
        result = await binance_spot_klines(symbol, interval, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_spot_avg_price_tool(symbol: str = "BTCUSDT") -> str:
    """
    Get Binance spot average price (5-minute window).
    
    Returns the current average price based on the
    last 5 minutes of trading activity.
    
    Args:
        symbol: Trading pair
    
    Returns:
        JSON with average price
    """
    import json
    try:
        result = await binance_spot_avg_price(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_spot_book_ticker_tool(symbol: str = None) -> str:
    """
    Get Binance spot best bid/ask prices.
    
    Returns the best bid and ask prices with quantities.
    Useful for spread analysis and execution planning.
    
    Args:
        symbol: Trading pair or None for all
    
    Returns:
        JSON with best bid/ask data
    """
    import json
    try:
        result = await binance_spot_book_ticker(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_spot_agg_trades_tool(symbol: str = "BTCUSDT", limit: int = 500) -> str:
    """
    Get Binance spot aggregate trades.
    
    Returns compressed/aggregated trade data:
    - Trades that execute at same price/time are combined
    - More efficient for volume analysis
    
    Args:
        symbol: Trading pair
        limit: Number of trades (max 1000)
    
    Returns:
        JSON with aggregate trades and analysis
    """
    import json
    try:
        result = await binance_spot_agg_trades(symbol, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_spot_exchange_info_tool(symbol: str = None) -> str:
    """
    Get Binance spot exchange information.
    
    Returns exchange rules and symbol specifications:
    - Trading rules (min/max qty, tick size)
    - Symbol status and filters
    - Available order types
    
    Args:
        symbol: Specific symbol or None for exchange summary
    
    Returns:
        JSON with exchange/symbol information
    """
    import json
    try:
        result = await binance_spot_exchange_info(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_spot_rolling_ticker_tool(symbol: str = "BTCUSDT", window: str = "1d") -> str:
    """
    Get Binance spot rolling window ticker.
    
    Returns price change statistics for a custom window:
    - Flexible window sizes (1m-59m, 1h-23h, 1d-7d)
    - Volume and trade count for window
    
    Args:
        symbol: Trading pair
        window: Window size (e.g., "1h", "4h", "1d", "7d")
    
    Returns:
        JSON with rolling window statistics
    """
    import json
    try:
        result = await binance_spot_rolling_ticker(symbol, window)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_spot_all_tickers_tool() -> str:
    """
    Get all Binance spot tickers with top movers.
    
    Returns market overview including:
    - Top gainers and losers
    - Highest volume pairs
    - Market summary statistics
    
    Returns:
        JSON with all tickers and top movers
    """
    import json
    try:
        result = await binance_spot_all_tickers()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_spot_snapshot_tool(symbol: str = "BTCUSDT") -> str:
    """
    Get comprehensive Binance spot market snapshot.
    
    Combines multiple data sources:
    - Current ticker and price data
    - Orderbook snapshot
    - Recent trades summary
    
    Args:
        symbol: Trading pair
    
    Returns:
        JSON with complete market snapshot
    """
    import json
    try:
        result = await binance_spot_snapshot(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def binance_spot_full_analysis_tool(symbol: str = "BTCUSDT") -> str:
    """
    Get complete Binance spot market analysis.
    
    The most comprehensive single-call analysis for Binance spot:
    - Ticker, orderbook, and trade data
    - Buy/sell flow analysis
    - Orderbook imbalance signals
    - Technical indicators and trends
    - Trading recommendations
    
    Args:
        symbol: Trading pair
    
    Returns:
        JSON with full market analysis and signals
    """
    import json
    try:
        result = await binance_spot_full_analysis(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# OKX REST API MCP TOOLS
# ============================================================================

@mcp.tool()
async def okx_ticker_tool(inst_id: str = "BTC-USDT-SWAP") -> str:
    """
    Get OKX ticker data for a specific instrument.
    
    Supports all OKX instrument types:
    - Perpetuals: BTC-USDT-SWAP, ETH-USDT-SWAP
    - Spot: BTC-USDT, ETH-USDT
    - Futures: BTC-USDT-240329 (expiry date)
    
    Args:
        inst_id: Instrument ID (e.g., BTC-USDT-SWAP)
    
    Returns:
        JSON with price, volume, and 24h stats
    """
    import json
    try:
        result = await okx_ticker(inst_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_all_tickers_tool(inst_type: str = "SWAP") -> str:
    """
    Get all OKX tickers for an instrument type.
    
    Args:
        inst_type: SPOT, SWAP, FUTURES, OPTION, MARGIN
    
    Returns:
        JSON with all tickers and top volume pairs
    """
    import json
    try:
        result = await okx_all_tickers(inst_type)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_index_ticker_tool(inst_id: str = "BTC-USD") -> str:
    """
    Get OKX index ticker.
    
    Args:
        inst_id: Index instrument ID (e.g., BTC-USD, ETH-USD)
    
    Returns:
        JSON with index price data
    """
    import json
    try:
        result = await okx_index_ticker(inst_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_orderbook_tool(inst_id: str = "BTC-USDT-SWAP", depth: int = 100) -> str:
    """
    Get OKX orderbook depth.
    
    Returns bid/ask levels with:
    - Best bid/ask and spread analysis
    - Volume imbalance calculation
    - Major bid/ask walls detection
    
    Args:
        inst_id: Instrument ID
        depth: Number of levels (max 400)
    
    Returns:
        JSON with orderbook and liquidity analysis
    """
    import json
    try:
        result = await okx_orderbook(inst_id, depth)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_trades_tool(inst_id: str = "BTC-USDT-SWAP", limit: int = 100) -> str:
    """
    Get recent OKX trades with flow analysis.
    
    Args:
        inst_id: Instrument ID
        limit: Number of trades (max 500)
    
    Returns:
        JSON with trades and buy/sell flow analysis
    """
    import json
    try:
        result = await okx_trades(inst_id, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_klines_tool(
    inst_id: str = "BTC-USDT-SWAP",
    interval: str = "1H",
    limit: int = 100
) -> str:
    """
    Get OKX klines/candlesticks.
    
    Args:
        inst_id: Instrument ID
        interval: 1m,3m,5m,15m,30m,1H,2H,4H,6H,12H,1D,1W,1M
        limit: Number of candles (max 300)
    
    Returns:
        JSON with OHLCV data and trend analysis
    """
    import json
    try:
        result = await okx_klines(inst_id, interval, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_funding_rate_tool(inst_id: str = "BTC-USDT-SWAP") -> str:
    """
    Get current OKX funding rate.
    
    Args:
        inst_id: Perpetual swap instrument ID
    
    Returns:
        JSON with funding rate and sentiment analysis
    """
    import json
    try:
        result = await okx_funding_rate(inst_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_funding_rate_history_tool(
    inst_id: str = "BTC-USDT-SWAP",
    limit: int = 50
) -> str:
    """
    Get OKX funding rate history.
    
    Args:
        inst_id: Perpetual swap instrument ID
        limit: Number of records (max 100)
    
    Returns:
        JSON with historical funding rates and trend
    """
    import json
    try:
        result = await okx_funding_rate_history(inst_id, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_open_interest_tool(
    inst_id: str = None,
    inst_type: str = "SWAP",
    uly: str = None
) -> str:
    """
    Get OKX open interest.
    
    Args:
        inst_id: Specific instrument or None for all
        inst_type: SWAP, FUTURES, OPTION
        uly: Underlying (e.g., BTC-USDT)
    
    Returns:
        JSON with open interest data
    """
    import json
    try:
        result = await okx_open_interest(inst_id, inst_type, uly)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_oi_volume_tool(ccy: str = "BTC", period: str = "1H") -> str:
    """
    Get OKX open interest and volume history.
    
    Args:
        ccy: Currency (BTC, ETH, etc.)
        period: 5m, 1H, 1D
    
    Returns:
        JSON with OI and volume history
    """
    import json
    try:
        result = await okx_oi_volume(ccy, period)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_long_short_ratio_tool(ccy: str = "BTC", period: str = "1H") -> str:
    """
    Get OKX long/short account ratio.
    
    Shows the ratio of accounts holding net long vs net short positions.
    
    Args:
        ccy: Currency (BTC, ETH, etc.)
        period: 5m, 1H, 1D
    
    Returns:
        JSON with long/short ratio and interpretation
    """
    import json
    try:
        result = await okx_long_short_ratio(ccy, period)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_taker_volume_tool(
    ccy: str = "BTC",
    inst_type: str = "SWAP",
    period: str = "1H"
) -> str:
    """
    Get OKX taker buy/sell volume.
    
    Shows aggressive buying vs selling pressure.
    
    Args:
        ccy: Currency (BTC, ETH, etc.)
        inst_type: SPOT, SWAP, FUTURES, OPTION
        period: 5m, 1H, 1D
    
    Returns:
        JSON with taker volume breakdown and signal
    """
    import json
    try:
        result = await okx_taker_volume(ccy, inst_type, period)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_instruments_tool(inst_type: str = "SWAP", inst_id: str = None) -> str:
    """
    Get OKX instrument information.
    
    Args:
        inst_type: SPOT, SWAP, FUTURES, OPTION, MARGIN
        inst_id: Specific instrument ID for details
    
    Returns:
        JSON with instrument specifications
    """
    import json
    try:
        result = await okx_instruments(inst_type, inst_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_mark_price_tool(inst_type: str = "SWAP", inst_id: str = None) -> str:
    """
    Get OKX mark price.
    
    Args:
        inst_type: SWAP, FUTURES, OPTION, MARGIN
        inst_id: Specific instrument ID
    
    Returns:
        JSON with mark price data
    """
    import json
    try:
        result = await okx_mark_price(inst_type, inst_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_insurance_fund_tool(inst_type: str = "SWAP") -> str:
    """
    Get OKX insurance fund balance.
    
    Args:
        inst_type: SWAP, FUTURES, OPTION
    
    Returns:
        JSON with insurance fund data
    """
    import json
    try:
        result = await okx_insurance_fund(inst_type)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_platform_volume_tool() -> str:
    """
    Get OKX 24h platform trading volume.
    
    Returns platform-wide trading volume in USD and CNY.
    
    Returns:
        JSON with platform volume statistics
    """
    import json
    try:
        result = await okx_platform_volume()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_options_summary_tool(uly: str = "BTC-USD") -> str:
    """
    Get OKX options market summary.
    
    Args:
        uly: Underlying (e.g., BTC-USD, ETH-USD)
    
    Returns:
        JSON with options OI, volume, and put/call ratio
    """
    import json
    try:
        result = await okx_options_summary(uly)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_market_snapshot_tool(symbol: str = "BTC") -> str:
    """
    Get comprehensive OKX market snapshot.
    
    Combines perpetual, spot, orderbook, funding, and OI data.
    
    Args:
        symbol: Base symbol (BTC, ETH, etc.)
    
    Returns:
        JSON with complete market snapshot
    """
    import json
    try:
        result = await okx_market_snapshot(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_full_analysis_tool(symbol: str = "BTC") -> str:
    """
    Get complete OKX market analysis with trading signals.
    
    The most comprehensive single-call analysis for OKX:
    - Perpetual and spot data
    - Orderbook and trade flow analysis
    - Long/short ratio and taker volume
    - Combined signals and recommendations
    
    Args:
        symbol: Base symbol (BTC, ETH, etc.)
    
    Returns:
        JSON with full analysis and trading signals
    """
    import json
    try:
        result = await okx_full_analysis(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def okx_top_movers_tool(inst_type: str = "SWAP", limit: int = 10) -> str:
    """
    Get OKX top gainers and losers.
    
    Args:
        inst_type: SWAP, SPOT, FUTURES
        limit: Number of results per category
    
    Returns:
        JSON with top gainers, losers, and volume leaders
    """
    import json
    try:
        result = await okx_top_movers(inst_type, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# KRAKEN REST API MCP TOOLS
# ============================================================================

@mcp.tool()
async def kraken_spot_ticker_tool(pair: str = "XBTUSD") -> str:
    """
    Get Kraken spot ticker data.
    
    Args:
        pair: Trading pair (XBTUSD, ETHUSD, SOLUSD, etc.)
    
    Returns:
        JSON with ticker data and price analysis
    """
    import json
    try:
        result = await kraken_spot_ticker(pair)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_all_spot_tickers_tool(pairs: str = "XBTUSD,ETHUSD,SOLUSD") -> str:
    """
    Get multiple Kraken spot tickers.
    
    Args:
        pairs: Comma-separated trading pairs
    
    Returns:
        JSON with multiple tickers sorted by volume
    """
    import json
    try:
        pair_list = [p.strip() for p in pairs.split(",")]
        result = await kraken_all_spot_tickers(pair_list)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_spot_orderbook_tool(pair: str = "XBTUSD", depth: int = 100) -> str:
    """
    Get Kraken spot order book.
    
    Args:
        pair: Trading pair
        depth: Number of levels (max 500)
    
    Returns:
        JSON with orderbook and liquidity analysis
    """
    import json
    try:
        result = await kraken_spot_orderbook(pair, depth)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_spot_trades_tool(pair: str = "XBTUSD", count: int = 100) -> str:
    """
    Get Kraken spot recent trades.
    
    Args:
        pair: Trading pair
        count: Number of trades
    
    Returns:
        JSON with trades and buy/sell flow analysis
    """
    import json
    try:
        result = await kraken_spot_trades(pair, count)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_spot_klines_tool(pair: str = "XBTUSD", interval: str = "1H") -> str:
    """
    Get Kraken spot OHLC candlestick data.
    
    Args:
        pair: Trading pair
        interval: 1m, 5m, 15m, 30m, 1H, 4H, 1D, 1W
    
    Returns:
        JSON with candlesticks and technical analysis
    """
    import json
    try:
        result = await kraken_spot_klines(pair, interval)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_spread_tool(pair: str = "XBTUSD") -> str:
    """
    Get Kraken recent spread data.
    
    Args:
        pair: Trading pair
    
    Returns:
        JSON with spread history and liquidity analysis
    """
    import json
    try:
        result = await kraken_spread(pair)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_assets_tool() -> str:
    """
    Get Kraken supported assets.
    
    Returns:
        JSON with all supported crypto assets
    """
    import json
    try:
        result = await kraken_assets()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_spot_pairs_tool(info: str = "info") -> str:
    """
    Get Kraken spot trading pairs.
    
    Args:
        info: info, leverage, fees, margin
    
    Returns:
        JSON with trading pair specifications
    """
    import json
    try:
        result = await kraken_spot_pairs(info)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_futures_ticker_tool(symbol: str = "PF_XBTUSD") -> str:
    """
    Get Kraken futures/perpetual ticker.
    
    Args:
        symbol: Futures symbol (PF_XBTUSD = perpetual BTC)
    
    Returns:
        JSON with futures ticker and funding analysis
    """
    import json
    try:
        result = await kraken_futures_ticker(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_all_futures_tickers_tool() -> str:
    """
    Get all Kraken futures/perpetual tickers.
    
    Returns:
        JSON with all perpetuals and market aggregates
    """
    import json
    try:
        result = await kraken_all_futures_tickers()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_futures_orderbook_tool(symbol: str = "PF_XBTUSD") -> str:
    """
    Get Kraken futures order book.
    
    Args:
        symbol: Futures symbol
    
    Returns:
        JSON with orderbook and analysis
    """
    import json
    try:
        result = await kraken_futures_orderbook(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_futures_trades_tool(symbol: str = "PF_XBTUSD") -> str:
    """
    Get Kraken futures recent trades.
    
    Args:
        symbol: Futures symbol
    
    Returns:
        JSON with trades and flow analysis
    """
    import json
    try:
        result = await kraken_futures_trades(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_futures_klines_tool(symbol: str = "PF_XBTUSD", interval: str = "1h") -> str:
    """
    Get Kraken futures candlestick data.
    
    Args:
        symbol: Futures symbol
        interval: 1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d, 1w
    
    Returns:
        JSON with candlesticks and analysis
    """
    import json
    try:
        result = await kraken_futures_klines(symbol, interval)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_futures_instruments_tool() -> str:
    """
    Get Kraken futures instrument specifications.
    
    Returns:
        JSON with all futures instruments and their specs
    """
    import json
    try:
        result = await kraken_futures_instruments()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_funding_rates_tool() -> str:
    """
    Get Kraken perpetual funding rates.
    
    Returns:
        JSON with funding rates for all perpetuals
    """
    import json
    try:
        result = await kraken_funding_rates()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_open_interest_tool() -> str:
    """
    Get Kraken perpetual open interest.
    
    Returns:
        JSON with open interest for all perpetuals
    """
    import json
    try:
        result = await kraken_open_interest()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_system_status_tool() -> str:
    """
    Get Kraken system status.
    
    Returns:
        JSON with system status and server time
    """
    import json
    try:
        result = await kraken_system_status()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_top_movers_tool(limit: int = 10) -> str:
    """
    Get Kraken top gainers and losers.
    
    Args:
        limit: Number of results per category
    
    Returns:
        JSON with top gainers and losers
    """
    import json
    try:
        result = await kraken_top_movers(limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_market_snapshot_tool(symbol: str = "BTC") -> str:
    """
    Get comprehensive Kraken market snapshot.
    
    Args:
        symbol: Base symbol (BTC, ETH, SOL, etc.)
    
    Returns:
        JSON with combined spot and futures data
    """
    import json
    try:
        result = await kraken_market_snapshot(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def kraken_full_analysis_tool(symbol: str = "BTC") -> str:
    """
    Get full Kraken analysis with trading signals.
    
    Args:
        symbol: Base symbol (BTC, ETH, SOL, etc.)
    
    Returns:
        JSON with comprehensive analysis and signals
    """
    import json
    try:
        result = await kraken_full_analysis(symbol)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# GATE.IO FUTURES REST API MCP TOOLS
# ============================================================================

@mcp.tool()
async def gateio_futures_contracts(settle: str = "usdt") -> str:
    """
    Get all Gate.io futures contracts.
    
    Args:
        settle: Settlement currency - 'usdt' or 'btc'
    
    Returns:
        JSON with all futures contract specifications
    """
    import json
    try:
        result = await gateio_futures_contracts_tool(settle)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_futures_contract(contract: str, settle: str = "usdt") -> str:
    """
    Get single Gate.io futures contract info.
    
    Args:
        contract: Contract name (e.g., 'BTC_USDT')
        settle: Settlement currency
    
    Returns:
        JSON with contract specifications and current state
    """
    import json
    try:
        result = await gateio_futures_contract_tool(contract, settle)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_futures_ticker(contract: str, settle: str = "usdt") -> str:
    """
    Get Gate.io futures ticker.
    
    Args:
        contract: Contract name (e.g., 'BTC_USDT')
        settle: Settlement currency
    
    Returns:
        JSON with current ticker data
    """
    import json
    try:
        result = await gateio_futures_ticker_tool(contract, settle)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_all_futures_tickers(settle: str = "usdt") -> str:
    """
    Get all Gate.io futures tickers.
    
    Args:
        settle: Settlement currency - 'usdt' or 'btc'
    
    Returns:
        JSON with all futures tickers sorted by volume
    """
    import json
    try:
        result = await gateio_all_futures_tickers_tool(settle)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_futures_orderbook(contract: str, settle: str = "usdt", limit: int = 50) -> str:
    """
    Get Gate.io futures orderbook.
    
    Args:
        contract: Contract name (e.g., 'BTC_USDT')
        settle: Settlement currency
        limit: Depth limit (max 50)
    
    Returns:
        JSON with orderbook bids and asks
    """
    import json
    try:
        result = await gateio_futures_orderbook_tool(contract, settle, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_futures_trades(contract: str, settle: str = "usdt", limit: int = 100) -> str:
    """
    Get Gate.io futures recent trades.
    
    Args:
        contract: Contract name (e.g., 'BTC_USDT')
        settle: Settlement currency
        limit: Number of trades (max 1000)
    
    Returns:
        JSON with recent trades and volume statistics
    """
    import json
    try:
        result = await gateio_futures_trades_tool(contract, settle, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_futures_klines(contract: str, interval: str = "1h", settle: str = "usdt", limit: int = 100) -> str:
    """
    Get Gate.io futures candlesticks.
    
    Args:
        contract: Contract name (e.g., 'BTC_USDT')
        interval: Kline interval (10s, 1m, 5m, 15m, 30m, 1h, 4h, 8h, 1d, 7d, 30d)
        settle: Settlement currency
        limit: Number of candles (max 2000)
    
    Returns:
        JSON with OHLCV candlestick data
    """
    import json
    try:
        result = await gateio_futures_klines_tool(contract, interval, settle, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_funding_rate(contract: str, settle: str = "usdt", limit: int = 100) -> str:
    """
    Get Gate.io funding rate history.
    
    Args:
        contract: Contract name (e.g., 'BTC_USDT')
        settle: Settlement currency
        limit: Number of records (max 1000)
    
    Returns:
        JSON with funding rate history and annualized rate
    """
    import json
    try:
        result = await gateio_funding_rate_tool(contract, settle, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_all_funding_rates(settle: str = "usdt") -> str:
    """
    Get funding rates for all Gate.io perpetuals.
    
    Args:
        settle: Settlement currency
    
    Returns:
        JSON with all funding rates sorted by absolute value
    """
    import json
    try:
        result = await gateio_all_funding_rates_tool(settle)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_contract_stats(contract: str, settle: str = "usdt", interval: str = "1h", limit: int = 24) -> str:
    """
    Get Gate.io contract statistics (OI, liquidations, L/S ratio).
    
    Args:
        contract: Contract name (e.g., 'BTC_USDT')
        settle: Settlement currency
        interval: Stats interval (5m, 1h, 1d)
        limit: Number of records (max 100)
    
    Returns:
        JSON with contract statistics including OI and liquidations
    """
    import json
    try:
        result = await gateio_contract_stats_tool(contract, settle, interval, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_open_interest(settle: str = "usdt") -> str:
    """
    Get open interest for top Gate.io contracts.
    
    Args:
        settle: Settlement currency
    
    Returns:
        JSON with open interest for top contracts by volume
    """
    import json
    try:
        result = await gateio_open_interest_tool(settle)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_liquidations(settle: str = "usdt", contract: str = None, limit: int = 100) -> str:
    """
    Get Gate.io liquidation history.
    
    Args:
        settle: Settlement currency
        contract: Optional contract filter
        limit: Number of records (max 1000)
    
    Returns:
        JSON with recent liquidations and statistics
    """
    import json
    try:
        result = await gateio_liquidations_tool(settle, contract, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_insurance_fund(settle: str = "usdt") -> str:
    """
    Get Gate.io insurance fund balance.
    
    Args:
        settle: Settlement currency
    
    Returns:
        JSON with insurance fund balance history
    """
    import json
    try:
        result = await gateio_insurance_fund_tool(settle)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_risk_limit_tiers(contract: str, settle: str = "usdt") -> str:
    """
    Get Gate.io risk limit tiers for a contract.
    
    Args:
        contract: Contract name (e.g., 'BTC_USDT')
        settle: Settlement currency
    
    Returns:
        JSON with risk limit tiers and margin requirements
    """
    import json
    try:
        result = await gateio_risk_limit_tiers_tool(contract, settle)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_delivery_contracts(settle: str = "usdt") -> str:
    """
    Get all Gate.io delivery futures contracts.
    
    Args:
        settle: Settlement currency
    
    Returns:
        JSON with delivery futures contracts
    """
    import json
    try:
        result = await gateio_delivery_contracts_tool(settle)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_delivery_ticker(contract: str, settle: str = "usdt") -> str:
    """
    Get Gate.io delivery futures ticker.
    
    Args:
        contract: Contract name
        settle: Settlement currency
    
    Returns:
        JSON with delivery futures ticker
    """
    import json
    try:
        result = await gateio_delivery_ticker_tool(contract, settle)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_options_underlyings() -> str:
    """
    Get Gate.io options underlying assets.
    
    Returns:
        JSON with available underlying assets for options
    """
    import json
    try:
        result = await gateio_options_underlyings_tool()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_options_expirations(underlying: str) -> str:
    """
    Get Gate.io options expiration dates.
    
    Args:
        underlying: Underlying asset (e.g., 'BTC_USDT')
    
    Returns:
        JSON with expiration timestamps
    """
    import json
    try:
        result = await gateio_options_expirations_tool(underlying)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_options_contracts(underlying: str, expiration: int = None) -> str:
    """
    Get Gate.io options contracts.
    
    Args:
        underlying: Underlying asset (e.g., 'BTC_USDT')
        expiration: Optional expiration timestamp filter
    
    Returns:
        JSON with options contracts (calls and puts)
    """
    import json
    try:
        result = await gateio_options_contracts_tool(underlying, expiration)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_options_tickers(underlying: str) -> str:
    """
    Get Gate.io options tickers with Greeks.
    
    Args:
        underlying: Underlying asset (e.g., 'BTC_USDT')
    
    Returns:
        JSON with options tickers including delta, gamma, vega, theta
    """
    import json
    try:
        result = await gateio_options_tickers_tool(underlying)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_options_underlying_ticker(underlying: str) -> str:
    """
    Get Gate.io underlying ticker for options.
    
    Args:
        underlying: Underlying asset (e.g., 'BTC_USDT')
    
    Returns:
        JSON with underlying asset ticker
    """
    import json
    try:
        result = await gateio_options_underlying_ticker_tool(underlying)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_options_orderbook(contract: str, limit: int = 20) -> str:
    """
    Get Gate.io options orderbook.
    
    Args:
        contract: Options contract name
        limit: Depth limit (max 50)
    
    Returns:
        JSON with options orderbook
    """
    import json
    try:
        result = await gateio_options_orderbook_tool(contract, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_market_snapshot(symbol: str = "BTC", settle: str = "usdt") -> str:
    """
    Get comprehensive Gate.io market snapshot.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC')
        settle: Settlement currency
    
    Returns:
        JSON with comprehensive market data including ticker, funding, OI, orderbook
    """
    import json
    try:
        result = await gateio_market_snapshot_tool(symbol, settle)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_top_movers(settle: str = "usdt", limit: int = 10) -> str:
    """
    Get top gainers and losers on Gate.io.
    
    Args:
        settle: Settlement currency
        limit: Number of movers to return
    
    Returns:
        JSON with top gainers and losers by price change
    """
    import json
    try:
        result = await gateio_top_movers_tool(settle, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_full_analysis(symbol: str = "BTC", settle: str = "usdt") -> str:
    """
    Get full Gate.io analysis with trading signals.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC')
        settle: Settlement currency
    
    Returns:
        JSON with comprehensive analysis and signals for funding, basis, L/S ratio, etc.
    """
    import json
    try:
        result = await gateio_full_analysis_tool(symbol, settle)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def gateio_perpetuals(settle: str = "usdt") -> str:
    """
    Get all Gate.io perpetual futures.
    
    Args:
        settle: Settlement currency
    
    Returns:
        JSON with all perpetual contracts and ticker data
    """
    import json
    try:
        result = await gateio_perpetuals_tool(settle)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# HYPERLIQUID REST API MCP TOOLS
# ============================================================================

@mcp.tool()
async def hyperliquid_meta() -> str:
    """
    Get Hyperliquid exchange metadata.
    
    Returns:
        JSON with exchange metadata including all perpetual contracts and universe info
    """
    import json
    try:
        result = await hyperliquid_meta_tool()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def hyperliquid_all_mids() -> str:
    """
    Get all mid prices for Hyperliquid perpetuals.
    
    Returns:
        JSON with all mid prices mapped by coin symbol
    """
    import json
    try:
        result = await hyperliquid_all_mids_tool()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def hyperliquid_ticker(coin: str) -> str:
    """
    Get Hyperliquid ticker for a specific coin.
    
    Args:
        coin: Coin symbol (e.g., 'BTC', 'ETH')
    
    Returns:
        JSON with ticker data including price, funding, OI, volume
    """
    import json
    try:
        result = await hyperliquid_ticker_tool(coin)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def hyperliquid_all_tickers() -> str:
    """
    Get all Hyperliquid perpetual tickers.
    
    Returns:
        JSON with all perpetual tickers sorted by volume (top 50)
    """
    import json
    try:
        result = await hyperliquid_all_tickers_tool()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def hyperliquid_orderbook(coin: str, depth: int = 20) -> str:
    """
    Get Hyperliquid orderbook for a coin.
    
    Args:
        coin: Coin symbol (e.g., 'BTC', 'ETH')
        depth: Number of levels to return (default: 20)
    
    Returns:
        JSON with orderbook including bids, asks, and analysis metrics
    """
    import json
    try:
        result = await hyperliquid_orderbook_tool(coin, depth)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def hyperliquid_klines(coin: str, interval: str = "1h", limit: int = 100) -> str:
    """
    Get Hyperliquid candlestick data.
    
    Args:
        coin: Coin symbol (e.g., 'BTC', 'ETH')
        interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
        limit: Number of candles (approximated by time range, default: 100)
    
    Returns:
        JSON with OHLCV candlestick data
    """
    import json
    try:
        result = await hyperliquid_klines_tool(coin, interval, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def hyperliquid_funding_rate(coin: str, limit: int = 100) -> str:
    """
    Get Hyperliquid funding rate history.
    
    Hyperliquid has hourly funding (unlike most exchanges with 8-hour).
    
    Args:
        coin: Coin symbol (e.g., 'BTC', 'ETH')
        limit: Number of records (hours, default: 100)
    
    Returns:
        JSON with funding rate history and analysis
    """
    import json
    try:
        result = await hyperliquid_funding_rate_tool(coin, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def hyperliquid_all_funding_rates() -> str:
    """
    Get funding rates for all Hyperliquid perpetuals.
    
    Returns:
        JSON with all funding rates sorted by absolute value
    """
    import json
    try:
        result = await hyperliquid_all_funding_rates_tool()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def hyperliquid_open_interest() -> str:
    """
    Get open interest for all Hyperliquid perpetuals.
    
    Returns:
        JSON with open interest data sorted by USD value
    """
    import json
    try:
        result = await hyperliquid_open_interest_tool()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def hyperliquid_top_movers(limit: int = 10) -> str:
    """
    Get top gainers and losers on Hyperliquid.
    
    Args:
        limit: Number of movers per category (default: 10)
    
    Returns:
        JSON with top gainers and losers by price change
    """
    import json
    try:
        result = await hyperliquid_top_movers_tool(limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def hyperliquid_exchange_stats() -> str:
    """
    Get overall Hyperliquid exchange statistics.
    
    Returns:
        JSON with aggregated stats including total OI, volume, and extremes
    """
    import json
    try:
        result = await hyperliquid_exchange_stats_tool()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def hyperliquid_spot_meta() -> str:
    """
    Get Hyperliquid spot market metadata.
    
    Returns:
        JSON with spot market tokens and universe info
    """
    import json
    try:
        result = await hyperliquid_spot_meta_tool()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def hyperliquid_spot_meta_and_ctxs() -> str:
    """
    Get Hyperliquid spot metadata and asset contexts.
    
    Returns:
        JSON with spot meta and current market contexts
    """
    import json
    try:
        result = await hyperliquid_spot_meta_and_ctxs_tool()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def hyperliquid_market_snapshot(coin: str = "BTC") -> str:
    """
    Get comprehensive Hyperliquid market snapshot.
    
    Args:
        coin: Coin symbol (e.g., 'BTC', 'ETH')
    
    Returns:
        JSON with combined ticker, orderbook, and funding data
    """
    import json
    try:
        result = await hyperliquid_market_snapshot_tool(coin)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def hyperliquid_full_analysis(coin: str = "BTC") -> str:
    """
    Get full Hyperliquid analysis with trading signals.
    
    Args:
        coin: Coin symbol (e.g., 'BTC', 'ETH')
    
    Returns:
        JSON with comprehensive analysis and signals for funding, premium, orderbook
    """
    import json
    try:
        result = await hyperliquid_full_analysis_tool(coin)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def hyperliquid_perpetuals() -> str:
    """
    Get all Hyperliquid perpetual contracts.
    
    Returns:
        JSON with all perpetual contracts and current market data
    """
    import json
    try:
        result = await hyperliquid_perpetuals_tool()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def hyperliquid_recent_trades(coin: str) -> str:
    """
    Get recent trade activity approximation for a coin.
    
    Note: Uses orderbook data as proxy since Hyperliquid doesn't expose
    direct trade history via public API.
    
    Args:
        coin: Coin symbol (e.g., 'BTC', 'ETH')
    
    Returns:
        JSON with orderbook summary as trade activity proxy
    """
    import json
    try:
        result = await hyperliquid_recent_trades_tool(coin)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# DERIBIT REST API MCP TOOLS
# ============================================================================

@mcp.tool()
async def deribit_instruments(currency: str = "BTC", kind: str = None) -> str:
    """
    Get all Deribit instruments for a currency.
    
    Args:
        currency: Currency (BTC, ETH, SOL, USDC, USDT)
        kind: Optional filter (future, option, spot)
    
    Returns:
        JSON with instrument specifications
    """
    import json
    try:
        result = await deribit_instruments_tool(currency, kind)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_currencies() -> str:
    """
    Get all supported currencies on Deribit.
    
    Returns:
        JSON with currency information
    """
    import json
    try:
        result = await deribit_currencies_tool()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_ticker(instrument_name: str) -> str:
    """
    Get Deribit ticker for an instrument.
    
    Args:
        instrument_name: Instrument name (e.g., 'BTC-PERPETUAL', 'ETH-28MAR25')
    
    Returns:
        JSON with ticker data including prices, volume, OI, Greeks (for options)
    """
    import json
    try:
        result = await deribit_ticker_tool(instrument_name)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_perpetual_ticker(currency: str = "BTC") -> str:
    """
    Get Deribit perpetual ticker with funding rate.
    
    Args:
        currency: Currency (BTC, ETH, SOL)
    
    Returns:
        JSON with perpetual ticker including price, funding, OI, volume
    """
    import json
    try:
        result = await deribit_perpetual_ticker_tool(currency)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_all_perpetual_tickers() -> str:
    """
    Get tickers for all Deribit perpetuals (BTC, ETH, SOL).
    
    Returns:
        JSON with all perpetual tickers
    """
    import json
    try:
        result = await deribit_all_perpetual_tickers_tool()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_futures_tickers(currency: str = "BTC") -> str:
    """
    Get all futures tickers for a currency.
    
    Args:
        currency: Currency
    
    Returns:
        JSON with all futures tickers including perpetual and dated futures
    """
    import json
    try:
        result = await deribit_futures_tickers_tool(currency)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_orderbook(instrument_name: str, depth: int = 20) -> str:
    """
    Get Deribit orderbook for an instrument.
    
    Args:
        instrument_name: Instrument name
        depth: Number of levels (1-10000)
    
    Returns:
        JSON with orderbook including bids, asks, spread, imbalance
    """
    import json
    try:
        result = await deribit_orderbook_tool(instrument_name, depth)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_trades(instrument_name: str, count: int = 100) -> str:
    """
    Get recent trades for a Deribit instrument.
    
    Args:
        instrument_name: Instrument name
        count: Number of trades (1-1000)
    
    Returns:
        JSON with recent trades and buy/sell analysis
    """
    import json
    try:
        result = await deribit_trades_tool(instrument_name, count)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_trades_by_currency(currency: str = "BTC", kind: str = None, count: int = 100) -> str:
    """
    Get recent trades for a currency.
    
    Args:
        currency: Currency
        kind: Optional filter (future, option)
        count: Number of trades
    
    Returns:
        JSON with recent trades across instruments
    """
    import json
    try:
        result = await deribit_trades_by_currency_tool(currency, kind, count)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_index_price(currency: str = "BTC") -> str:
    """
    Get Deribit index price.
    
    Args:
        currency: Currency (BTC, ETH)
    
    Returns:
        JSON with index price and estimated delivery price
    """
    import json
    try:
        result = await deribit_index_price_tool(currency)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_index_names() -> str:
    """
    Get all available index price names on Deribit.
    
    Returns:
        JSON with list of available indices
    """
    import json
    try:
        result = await deribit_index_names_tool()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_funding_rate(currency: str = "BTC") -> str:
    """
    Get current funding rate for Deribit perpetual.
    
    Deribit has hourly funding (vs 8-hour on most exchanges).
    
    Args:
        currency: Currency (BTC, ETH, SOL)
    
    Returns:
        JSON with current funding rate and annualized rate
    """
    import json
    try:
        result = await deribit_funding_rate_tool(currency)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_funding_history(currency: str = "BTC", hours: int = 24) -> str:
    """
    Get funding rate history.
    
    Args:
        currency: Currency
        hours: Number of hours of history
    
    Returns:
        JSON with funding rate history and statistics
    """
    import json
    try:
        result = await deribit_funding_history_tool(currency, hours)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_funding_analysis(currency: str = "BTC") -> str:
    """
    Get comprehensive funding rate analysis.
    
    Args:
        currency: Currency
    
    Returns:
        JSON with funding analysis and statistics
    """
    import json
    try:
        result = await deribit_funding_analysis_tool(currency)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_historical_volatility(currency: str = "BTC") -> str:
    """
    Get historical volatility data.
    
    Args:
        currency: Currency (BTC or ETH)
    
    Returns:
        JSON with historical volatility time series
    """
    import json
    try:
        result = await deribit_historical_volatility_tool(currency)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_dvol(currency: str = "BTC", hours: int = 24) -> str:
    """
    Get DVOL (Deribit Volatility Index) data.
    
    DVOL is similar to VIX for crypto, showing implied volatility.
    
    Args:
        currency: Currency (BTC or ETH)
        hours: Hours of history
    
    Returns:
        JSON with DVOL time series and OHLC
    """
    import json
    try:
        result = await deribit_dvol_tool(currency, hours)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_klines(instrument_name: str, resolution: str = "60", hours: int = 24) -> str:
    """
    Get OHLCV candlestick data.
    
    Args:
        instrument_name: Instrument name
        resolution: Resolution in minutes (1, 3, 5, 10, 15, 30, 60, 120, 180, 360, 720, 1D)
        hours: Hours of history
    
    Returns:
        JSON with OHLCV candlestick data
    """
    import json
    try:
        result = await deribit_klines_tool(instrument_name, resolution, hours)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_open_interest(currency: str = "BTC") -> str:
    """
    Get open interest for a currency.
    
    Args:
        currency: Currency
    
    Returns:
        JSON with open interest for futures and options
    """
    import json
    try:
        result = await deribit_open_interest_tool(currency)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_options_summary(currency: str = "BTC") -> str:
    """
    Get options market summary.
    
    Deribit is the largest crypto options exchange.
    
    Args:
        currency: Currency
    
    Returns:
        JSON with aggregated options statistics including put/call ratio
    """
    import json
    try:
        result = await deribit_options_summary_tool(currency)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_options_chain(currency: str = "BTC", expiration: str = None) -> str:
    """
    Get options chain.
    
    Args:
        currency: Currency
        expiration: Optional expiration filter (e.g., '28MAR25')
    
    Returns:
        JSON with options chain organized by expiration and strike
    """
    import json
    try:
        result = await deribit_options_chain_tool(currency, expiration)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_option_ticker(instrument_name: str) -> str:
    """
    Get option ticker with Greeks.
    
    Args:
        instrument_name: Option instrument name (e.g., 'BTC-28MAR25-100000-C')
    
    Returns:
        JSON with option ticker including IV, delta, gamma, theta, vega
    """
    import json
    try:
        result = await deribit_option_ticker_tool(instrument_name)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_top_options(currency: str = "BTC", limit: int = 20) -> str:
    """
    Get top options by open interest.
    
    Args:
        currency: Currency
        limit: Number of options to return
    
    Returns:
        JSON with top calls and puts by OI
    """
    import json
    try:
        result = await deribit_top_options_tool(currency, limit)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_market_snapshot(currency: str = "BTC") -> str:
    """
    Get comprehensive Deribit market snapshot.
    
    Args:
        currency: Currency
    
    Returns:
        JSON with combined perpetual, index, volatility, OI data
    """
    import json
    try:
        result = await deribit_market_snapshot_tool(currency)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_full_analysis(currency: str = "BTC") -> str:
    """
    Get full Deribit analysis with trading signals.
    
    Args:
        currency: Currency
    
    Returns:
        JSON with comprehensive analysis including funding, options, volatility signals
    """
    import json
    try:
        result = await deribit_full_analysis_tool(currency)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_exchange_stats() -> str:
    """
    Get overall Deribit exchange statistics.
    
    Returns:
        JSON with aggregated stats for all currencies
    """
    import json
    try:
        result = await deribit_exchange_stats_tool()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_book_summary(currency: str = "BTC", kind: str = None) -> str:
    """
    Get book summary for all instruments.
    
    Args:
        currency: Currency
        kind: Optional filter (future, option)
    
    Returns:
        JSON with book summaries including volume, OI, prices
    """
    import json
    try:
        result = await deribit_book_summary_tool(currency, kind)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def deribit_settlements(currency: str = "BTC", count: int = 20) -> str:
    """
    Get recent settlements.
    
    Args:
        currency: Currency
        count: Number of records
    
    Returns:
        JSON with recent settlement records
    """
    import json
    try:
        result = await deribit_settlements_tool(currency, count)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# INSTITUTIONAL FEATURE TOOLS (Phase 4 Week 1 - 15 Tools)
# ============================================================================

@mcp.tool()
async def institutional_price_features(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Get institutional-grade price features including microprice, spread dynamics, and efficiency metrics.
    
    Calculates 19 price-based features:
    - Microprice (volume-weighted fair price) and deviation
    - Spread dynamics (compression/expansion, z-score)
    - Pressure ratios (bid vs ask)
    - Price efficiency and Hurst exponent
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT", "ETHUSDT")
        exchange: Exchange name (binance, bybit, okx, etc.)
    
    Returns:
        XML analysis with features and interpretations
    
    Example:
        "Get price features for BTC on Binance" → institutional_price_features("BTCUSDT", "binance")
    """
    import json
    try:
        result = await get_price_features(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="institutional_price_features">{str(e)}</error>'


@mcp.tool()
async def institutional_spread_dynamics(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Analyze spread behavior and liquidity conditions.
    
    Features:
    - Spread in basis points
    - Z-score relative to historical distribution
    - Compression/expansion velocity
    - Spike detection for sudden liquidity changes
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML spread analysis with actionable interpretation
    """
    import json
    try:
        result = await get_spread_dynamics(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="institutional_spread_dynamics">{str(e)}</error>'


@mcp.tool()
async def institutional_price_efficiency(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Get price efficiency and trend persistence metrics.
    
    Features:
    - Hurst exponent (trend persistence: >0.5 trending, <0.5 mean reverting)
    - Price efficiency score
    - Tick reversal rate (noise level)
    - Price entropy (randomness measure)
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with regime classification and efficiency metrics
    """
    import json
    try:
        result = await get_price_efficiency_metrics(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="institutional_price_efficiency">{str(e)}</error>'


@mcp.tool()
async def institutional_orderbook_features(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Get full orderbook liquidity structure features.
    
    Calculates 20 orderbook-based features:
    - Multi-level depth imbalance (5, 10, cumulative levels)
    - Liquidity gradient and concentration
    - Absorption and replenishment dynamics
    - Wall detection (pull/push manipulation)
    - Support/resistance strength
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML analysis with orderbook features and liquidity insights
    """
    import json
    try:
        result = await get_orderbook_features(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="institutional_orderbook_features">{str(e)}</error>'


@mcp.tool()
async def institutional_depth_imbalance(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Analyze orderbook depth imbalance at multiple levels.
    
    Calculates:
    - Level 5 imbalance (tight book)
    - Level 10 imbalance (wider book)
    - Cumulative imbalance (full book)
    - Directional bias assessment
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with imbalance metrics and directional bias
    """
    import json
    try:
        result = await get_depth_imbalance(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="institutional_depth_imbalance">{str(e)}</error>'


@mcp.tool()
async def institutional_wall_detection(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Detect orderbook manipulation patterns (spoofing, walls).
    
    Identifies:
    - Pull walls (placed then removed to fake support/resistance)
    - Push walls (aggressive walls to attract order flow)
    - Liquidity persistence (real vs fake liquidity)
    - Manipulation score and alert level
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with wall detection signals and manipulation score
    """
    import json
    try:
        result = await get_wall_detection(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="institutional_wall_detection">{str(e)}</error>'


@mcp.tool()
async def institutional_trade_features(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Get trade flow analysis features.
    
    Calculates 18 trade-based features:
    - CVD (Cumulative Volume Delta) - net buying/selling
    - Buy/sell volume ratios
    - Whale activity detection
    - Trade clustering and aggression
    - Flow toxicity (informed trading)
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with trade flow features and pressure analysis
    """
    import json
    try:
        result = await get_trade_features(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="institutional_trade_features">{str(e)}</error>'


@mcp.tool()
async def institutional_cvd_analysis(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Get detailed CVD (Cumulative Volume Delta) analysis.
    
    CVD measures net buying vs selling pressure:
    - CVD value and normalized score
    - CVD velocity (rate of change)
    - Pressure state (strong buy/sell/neutral)
    - Volume breakdown
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with CVD analysis and pressure assessment
    """
    import json
    try:
        result = await get_cvd_analysis(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="institutional_cvd_analysis">{str(e)}</error>'


@mcp.tool()
async def institutional_whale_detection(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Detect whale/institutional trading activity.
    
    Identifies:
    - Whale volume ratio and direction
    - Large trade clustering
    - Institutional flow indicators
    - Flow toxicity (informed vs uninformed)
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with whale activity metrics and alert levels
    """
    import json
    try:
        result = await get_whale_detection(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="institutional_whale_detection">{str(e)}</error>'


@mcp.tool()
async def institutional_funding_features(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Get funding rate dynamics and carry opportunity features.
    
    Calculates 15 funding-based features:
    - Current and predicted funding rates
    - Funding z-score and extremity
    - Funding velocity and momentum
    - Carry opportunity score
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with funding features and carry analysis
    """
    import json
    try:
        result = await get_funding_features(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="institutional_funding_features">{str(e)}</error>'


@mcp.tool()
async def institutional_funding_sentiment(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Analyze funding rate for market sentiment and positioning.
    
    Provides:
    - Sentiment classification (bullish/bearish)
    - Crowding risk assessment
    - Funding-based reversal signals
    - Strategy implications
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with sentiment analysis and reversal probability
    """
    import json
    try:
        result = await get_funding_sentiment(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="institutional_funding_sentiment">{str(e)}</error>'


@mcp.tool()
async def institutional_oi_features(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Get open interest dynamics and leverage features.
    
    Calculates 18 OI-based features:
    - OI levels and changes
    - Leverage ratios (notional/spot volume)
    - Intent classification (accumulation vs distribution)
    - Cascade potential (liquidation risk)
    - OI divergence from price
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with OI features and leverage analysis
    """
    import json
    try:
        result = await get_oi_features(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="institutional_oi_features">{str(e)}</error>'


@mcp.tool()
async def institutional_leverage_risk(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Analyze leverage and liquidation cascade risk.
    
    Assesses:
    - Current market leverage level and trend
    - Leverage buildup rate
    - Liquidation cascade potential
    - Directional vulnerability (longs vs shorts)
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with leverage risk assessment and warnings
    """
    import json
    try:
        result = await get_leverage_risk(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="institutional_leverage_risk">{str(e)}</error>'


@mcp.tool()
async def institutional_liquidation_features(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Get liquidation cascade and stress features.
    
    Calculates 12 liquidation-based features:
    - Liquidation intensity and velocity
    - Dominant side (longs vs shorts)
    - Cascade patterns and clustering
    - Stress levels and recovery signals
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with liquidation features and cascade analysis
    """
    import json
    try:
        result = await get_liquidation_features(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="institutional_liquidation_features">{str(e)}</error>'


@mcp.tool()
async def institutional_mark_price_features(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Get mark price premium/discount and basis features.
    
    Calculates 10 mark price features:
    - Mark vs index premium/discount
    - Premium z-score and extremity
    - Basis and basis velocity
    - Fair value gap signals
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with mark price features and basis analysis
    """
    import json
    try:
        result = await get_mark_price_features(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="institutional_mark_price_features">{str(e)}</error>'


# ============================================================================
# COMPOSITE INTELLIGENCE TOOLS (Phase 4 Week 2 - 10 Tools)
# ============================================================================

@mcp.tool()
async def composite_smart_accumulation(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Detect institutional accumulation using Smart Money Index.
    
    Combines multiple signals to detect institutional activity:
    - Orderbook absorption (institutions absorbing without price impact)
    - Whale trade activity (large directional trades)
    - Flow toxicity (informed order flow)
    - CVD-price divergence (stealth accumulation/distribution)
    
    Signal Levels:
    - > 0.7: Strong institutional accumulation
    - 0.4-0.7: Moderate institutional activity
    - < 0.4: Low institutional presence
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
    
    Returns:
        XML with accumulation signal, confidence, and action
    """
    import json
    try:
        result = await get_smart_accumulation_signal(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="composite_smart_accumulation">{str(e)}</error>'


@mcp.tool()
async def composite_smart_money_flow(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Get Smart Money Flow Direction for institutional trading direction.
    
    Detects the direction and strength of institutional money flow:
    - BUYING: Net institutional buying pressure
    - SELLING: Net institutional selling pressure
    - NEUTRAL: No clear directional bias
    
    Components analyzed:
    - Aggressive delta (net taker flow)
    - Whale flow direction
    - CVD trend
    - OI-price divergence
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with flow direction, strength, and trade bias
    """
    import json
    try:
        result = await get_smart_money_flow(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="composite_smart_money_flow">{str(e)}</error>'


@mcp.tool()
async def composite_squeeze_probability(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Get Squeeze Probability for short/long squeeze detection.
    
    Combines multiple factors to assess squeeze likelihood:
    - Extreme funding rates (crowded positioning)
    - High leverage ratios
    - Position crowding metrics
    - Price compression patterns
    
    Signal Levels:
    - > 0.7: HIGH squeeze probability - expect violent move
    - 0.4-0.7: Moderate squeeze risk
    - < 0.4: Low squeeze probability
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with squeeze probability, direction, and action
    """
    import json
    try:
        result = await get_short_squeeze_probability(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="composite_squeeze_probability">{str(e)}</error>'


@mcp.tool()
async def composite_stop_hunt_detector(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Detect stop hunting manipulation patterns.
    
    Identifies potential stop hunting activity:
    - Rapid price spikes to known stop levels
    - Volume anomalies during spikes
    - Quick price recovery patterns
    - Orderbook wall movements
    
    Warning Levels:
    - > 0.6: High manipulation probability - beware of stops
    - 0.4-0.6: Some manipulation signs
    - < 0.4: Normal market conditions
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with stop hunt probability and recommendations
    """
    import json
    try:
        result = await get_stop_hunt_detector(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="composite_stop_hunt_detector">{str(e)}</error>'


@mcp.tool()
async def composite_momentum_quality(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Get Momentum Quality Score for trend sustainability assessment.
    
    Evaluates trend strength and sustainability:
    - Volume confirmation (trend supported by volume)
    - Price efficiency (clean vs choppy movement)
    - CVD alignment (order flow supports direction)
    - OI growth (new money entering)
    
    Quality Levels:
    - > 0.7: High quality - sustainable trend
    - 0.5-0.7: Moderate quality - trend may continue
    - < 0.5: Low quality - reversal likely
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with quality score, trend outlook, and strategy
    """
    import json
    try:
        result = await get_momentum_quality_signal(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="composite_momentum_quality">{str(e)}</error>'


@mcp.tool()
async def composite_momentum_exhaustion(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Detect momentum exhaustion for trend reversal warning.
    
    Identifies signs of trend exhaustion:
    - Declining volume on price continuation
    - CVD divergence from price
    - Decreasing OI growth
    - Spread widening (liquidity withdrawal)
    
    Exhaustion Levels:
    - > 0.7: HIGH exhaustion - reversal imminent
    - 0.5-0.7: Moderate exhaustion - trend weakening
    - < 0.5: Low exhaustion - trend healthy
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with exhaustion level and reversal probability
    """
    import json
    try:
        result = await get_momentum_exhaustion(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="composite_momentum_exhaustion">{str(e)}</error>'


@mcp.tool()
async def composite_market_maker_activity(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Analyze Market Maker positioning and activity.
    
    Detects market maker behavior:
    - Wall placement patterns (bid/ask)
    - Spread manipulation
    - Quote stuffing detection
    - Inventory management signals
    
    Activity Types:
    - passive: MM absorbing flow, providing liquidity
    - active: MM taking directional positions
    - defensive: MM reducing exposure
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with MM activity type, inventory bias, and interpretation
    """
    import json
    try:
        result = await get_market_maker_activity(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="composite_market_maker_activity">{str(e)}</error>'


@mcp.tool()
async def composite_liquidation_cascade_risk(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Assess liquidation cascade risk for systemic risk detection.
    
    Evaluates cascade liquidation probability:
    - Liquidation cluster proximity
    - Leverage stress levels
    - Thin liquidity zones
    - Funding pressure direction
    
    Severity Levels:
    - CRITICAL: >0.8 - Imminent cascade risk
    - HIGH: 0.6-0.8 - Elevated risk
    - MODERATE: 0.4-0.6 - Some risk
    - LOW: <0.4 - Normal conditions
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with cascade risk, severity, direction, and action
    """
    import json
    try:
        result = await get_liquidation_cascade_risk(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="composite_liquidation_cascade_risk">{str(e)}</error>'


@mcp.tool()
async def composite_institutional_phase(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Detect institutional market cycle phase.
    
    Identifies the current market cycle phase:
    - ACCUMULATION: Smart money building positions
    - MARKUP: Uptrend phase after accumulation
    - DISTRIBUTION: Smart money exiting positions
    - MARKDOWN: Downtrend phase after distribution
    - NEUTRAL: No clear phase detected
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with phase, intensity, direction, and strategy
    """
    import json
    try:
        result = await get_institutional_phase(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="composite_institutional_phase">{str(e)}</error>'


@mcp.tool()
async def composite_aggregated_intelligence(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Get complete aggregated market intelligence - the MASTER tool.
    
    This comprehensive tool:
    1. Aggregates all 15 composite signals
    2. Ranks signals by importance
    3. Detects and resolves conflicts
    4. Generates actionable trade recommendation
    
    Output includes:
    - Market bias (bullish/bearish/neutral) with confidence
    - Top 5 most important signals ranked
    - Conflict detection and resolution
    - Trade recommendation (direction, strength, entry bias)
    - Risk assessment and urgency
    - Category scores (momentum, flow, risk, timing, execution)
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Comprehensive XML intelligence with trade recommendation
    """
    import json
    try:
        result = await get_aggregated_intelligence(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="composite_aggregated_intelligence">{str(e)}</error>'


@mcp.tool()
async def composite_execution_quality(symbol: str = "BTCUSDT", exchange: str = "binance") -> str:
    """
    Assess execution quality and optimal entry timing.
    
    Evaluates current execution conditions:
    - Spread tightness
    - Liquidity depth
    - Slippage estimate
    - Optimal position sizing
    
    Quality Levels:
    - EXCELLENT: Best execution conditions
    - GOOD: Favorable conditions
    - FAIR: Acceptable conditions
    - POOR: Consider waiting
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        XML with quality score, slippage estimate, and sizing guidance
    """
    import json
    try:
        result = await get_execution_quality(symbol, exchange)
        if result.get("success") and result.get("xml_analysis"):
            return result["xml_analysis"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="composite_execution_quality">{str(e)}</error>'


# ============================================================================
# VISUALIZATION TOOLS (Phase 4 Week 3 - 5 Tools)
# ============================================================================

@mcp.tool()
async def viz_feature_candles(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "5m",
    periods: int = 50,
    overlays: str = "microprice,cvd,depth_imbalance_5,funding_rate,risk_score"
) -> str:
    """
    Get feature-enriched OHLCV candles for visualization.
    
    Returns candlestick data with institutional feature overlays:
    - Price OHLCV data for each period
    - Selected feature values overlaid on each candle
    - Panel mapping for multi-chart rendering
    
    Supported overlays:
    - Price: microprice, vwap, support_strength, resistance_strength
    - Volume: cvd, buy_volume, sell_volume
    - Momentum: momentum_quality, hurst_exponent, momentum_exhaustion
    - Risk: leverage_index, liquidation_cascade_risk, risk_score
    - Orderbook: depth_imbalance_5, depth_imbalance_10, absorption_ratio
    - Funding: funding_rate, funding_zscore
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        timeframe: Candle timeframe ("1m", "5m", "15m", "1h")
        periods: Number of candles to return (max 200)
        overlays: Comma-separated list of feature overlays
    
    Returns:
        XML with candle data and visualization hints
    """
    import json
    try:
        overlay_list = [o.strip() for o in overlays.split(",") if o.strip()]
        result = await get_feature_candles(symbol, exchange, timeframe, periods, overlay_list)
        if result.get("success") and result.get("xml_visualization"):
            return result["xml_visualization"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="viz_feature_candles">{str(e)}</error>'


@mcp.tool()
async def viz_liquidity_heatmap(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    depth_levels: int = 20,
    include_walls: bool = True
) -> str:
    """
    Get liquidity heatmap data for orderbook visualization.
    
    Returns structured data for rendering bid/ask liquidity:
    - Liquidity distribution across price levels
    - Intensity values for heatmap coloring
    - Detected support/resistance walls
    - Imbalance and liquidity metrics
    
    Visualization includes:
    - Bid-side liquidity levels with intensity
    - Ask-side liquidity levels with intensity
    - Wall detection (large orders acting as support/resistance)
    - Color scale hints for rendering
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        depth_levels: Number of price levels per side (max 50)
        include_walls: Whether to detect and include liquidity walls
    
    Returns:
        XML with heatmap data and visualization hints
    """
    import json
    try:
        result = await get_liquidity_heatmap(symbol, exchange, depth_levels, include_walls)
        if result.get("success") and result.get("xml_visualization"):
            return result["xml_visualization"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="viz_liquidity_heatmap">{str(e)}</error>'


@mcp.tool()
async def viz_signal_dashboard(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    include_alerts: bool = True
) -> str:
    """
    Get signal dashboard data for real-time monitoring.
    
    Returns a structured grid of all composite signals:
    - Current signal values and activation status
    - Signal categorization (smart money, momentum, risk, etc.)
    - Overall market bias assessment
    - Active alerts for extreme values
    
    Dashboard categories:
    - Smart Money: Accumulation, flow direction
    - Squeeze Risk: Squeeze probability, stop hunt
    - Momentum: Quality, exhaustion
    - Risk: Liquidation cascade, overall risk
    - Market Maker: Activity index
    
    Color scheme:
    - Strong bullish (green): High confidence bullish
    - Bullish (light green): Moderate bullish
    - Neutral (gray): No clear direction
    - Bearish (light red): Moderate bearish
    - Strong bearish (red): High confidence bearish
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        include_alerts: Whether to generate alerts for extreme values
    
    Returns:
        XML with dashboard grid data and visualization hints
    """
    import json
    try:
        result = await get_signal_dashboard(symbol, exchange, include_alerts)
        if result.get("success") and result.get("xml_visualization"):
            return result["xml_visualization"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="viz_signal_dashboard">{str(e)}</error>'


@mcp.tool()
async def viz_regime_timeline(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    include_history: bool = True,
    include_transitions: bool = True
) -> str:
    """
    Get market regime visualization for timeline rendering.
    
    Returns current regime state plus historical transitions:
    - Current regime identification and confidence
    - Regime characteristics (volatility, trend, liquidity)
    - Transition probability matrix
    - Historical regime timeline
    - Trading strategy implications
    
    Regimes detected:
    - ACCUMULATION: Smart money building positions (bullish)
    - DISTRIBUTION: Smart money exiting (bearish)
    - BREAKOUT: High volatility expansion (trending)
    - SQUEEZE: Forced liquidation cascade (extreme)
    - MEAN_REVERSION: Range-bound behavior
    - CHAOS: Extreme unpredictable volatility
    - CONSOLIDATION: Low volatility, building energy
    
    Color scheme for timeline:
    - Green: Accumulation
    - Red: Distribution/Squeeze
    - Orange: Breakout
    - Indigo: Mean Reversion
    - Purple: Chaos
    - Gray: Consolidation
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        include_history: Include regime change history
        include_transitions: Include transition probabilities
    
    Returns:
        XML with regime data and timeline visualization hints
    """
    import json
    try:
        result = await get_regime_visualization(symbol, exchange, include_history, include_transitions)
        if result.get("success") and result.get("xml_visualization"):
            return result["xml_visualization"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="viz_regime_timeline">{str(e)}</error>'


@mcp.tool()
async def viz_correlation_matrix(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    feature_groups: str = "prices,orderbook,trades,funding,oi",
    include_clusters: bool = True
) -> str:
    """
    Get feature correlation matrix for analysis visualization.
    
    Calculates correlations between institutional features:
    - Correlation values between all feature pairs
    - Top positive correlations (redundant features)
    - Top negative correlations (diversifying features)
    - Feature clusters that move together
    
    Feature groups available:
    - prices: microprice, spread_zscore, pressure_ratio, hurst_exponent
    - orderbook: depth_imbalance, absorption_ratio, support/resistance
    - trades: cvd, whale_activity, flow_toxicity, trade_intensity
    - funding: funding_rate, funding_zscore, funding_momentum
    - oi: oi_delta, leverage_index, cascade_risk
    
    Color scale for matrix:
    - Strong positive (green): 0.7 to 1.0
    - Positive (light green): 0.3 to 0.7
    - Weak/None (gray): -0.3 to 0.3
    - Negative (light red): -0.7 to -0.3
    - Strong negative (red): -1.0 to -0.7
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        feature_groups: Comma-separated list of groups to analyze
        include_clusters: Whether to identify correlated clusters
    
    Returns:
        XML with correlation matrix data and visualization hints
    """
    import json
    try:
        groups = [g.strip() for g in feature_groups.split(",") if g.strip()]
        result = await get_correlation_matrix(symbol, exchange, groups, include_clusters)
        if result.get("success") and result.get("xml_visualization"):
            return result["xml_visualization"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="viz_correlation_matrix">{str(e)}</error>'


# ============================================================================
# FEATURE QUERY TOOLS (Phase 4 Week 4 - 4 Tools)
# ============================================================================

@mcp.tool()
async def query_features(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    feature_type: str = "prices",
    lookback_minutes: int = 60,
    limit: int = 100,
    columns: str = ""
) -> str:
    """
    Query historical feature data by time range.
    
    Retrieves stored feature records for analysis, backtesting,
    or visualization. Results include all calculated features
    for the specified stream type.
    
    Available feature types:
    - prices: Microprice, spread dynamics, pressure ratios
    - orderbook: Depth imbalance, liquidity, absorption
    - trades: CVD, whale detection, market impact
    - funding: Funding rate, stress, carry yield
    - oi: Open interest, leverage, liquidation risk
    - liquidations: Cascade events, severity metrics
    - mark_prices: Basis, premium/discount
    - ticker: Volume, volatility, institutional interest
    - composite: All composite signals
    
    Example query: Query last hour of price features
    ```
    query_features("BTCUSDT", "binance", "prices", 60, 100)
    ```
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        feature_type: Type of features to query
        lookback_minutes: How far back to query (max 1440 = 24h)
        limit: Maximum records to return (max 1000)
        columns: Comma-separated list of specific columns (empty = all)
    
    Returns:
        XML with query results and summary statistics
    """
    import json
    from .tools.feature_query_tools import query_historical_features
    try:
        cols = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
        result = await query_historical_features(symbol, exchange, feature_type, lookback_minutes, limit, cols)
        if result.get("success") and result.get("xml_summary"):
            return result["xml_summary"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="query_features">{str(e)}</error>'


@mcp.tool()
async def export_features_to_csv(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    feature_types: str = "prices,orderbook,trades",
    lookback_minutes: int = 60,
    include_composite: bool = False
) -> str:
    """
    Export features to CSV format for external analysis.
    
    Creates a CSV string containing all requested feature data
    that can be saved to file or loaded into pandas/numpy.
    
    The CSV includes:
    - Timestamp column for time alignment
    - All features from selected feature types
    - Optional composite signals
    
    Usage with pandas:
    ```python
    import pandas as pd
    import io
    df = pd.read_csv(io.StringIO(csv_data))
    ```
    
    Save to file:
    ```python
    with open('features.csv', 'w') as f:
        f.write(csv_data)
    ```
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        feature_types: Comma-separated list of feature types
        lookback_minutes: Time range to export (max 1440 = 24h)
        include_composite: Include composite signals in export
    
    Returns:
        XML summary with CSV data embedded
    """
    import json
    from .tools.feature_query_tools import export_features_csv
    try:
        types = [t.strip() for t in feature_types.split(",") if t.strip()]
        result = await export_features_csv(symbol, exchange, types, lookback_minutes, include_composite)
        if result.get("success") and result.get("xml_summary"):
            # Return both summary and CSV data
            response = result["xml_summary"]
            if result.get("csv_data"):
                response += f"\n\n<!-- CSV_DATA_START -->\n{result['csv_data'][:5000]}\n<!-- CSV_DATA_END -->"
            return response
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="export_features_to_csv">{str(e)}</error>'


@mcp.tool()
async def get_feature_stats(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    feature_type: str = "prices",
    lookback_minutes: int = 60
) -> str:
    """
    Get statistical summary of feature distributions.
    
    Calculates comprehensive statistics for each feature:
    - Mean, standard deviation, min, max
    - Median and percentiles (25th, 75th)
    - Null/zero counts
    - Distribution insights and anomalies
    
    Useful for:
    - Understanding feature ranges and normalization needs
    - Detecting data quality issues
    - Identifying unusual market conditions
    
    Example statistics returned:
    - microprice: mean=50234.5, std=12.3, min=50200, max=50270
    - spread_zscore: mean=0.12, std=1.05, skewed left
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        feature_type: Feature type to analyze
        lookback_minutes: Time range for statistics (max 1440)
    
    Returns:
        XML with comprehensive feature statistics
    """
    import json
    from .tools.feature_query_tools import get_feature_statistics
    try:
        result = await get_feature_statistics(symbol, exchange, feature_type, lookback_minutes)
        if result.get("success") and result.get("xml_summary"):
            return result["xml_summary"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="get_feature_stats">{str(e)}</error>'


@mcp.tool()
async def analyze_feature_correlations(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    feature_types: str = "prices,orderbook,trades,funding,oi",
    lookback_minutes: int = 60
) -> str:
    """
    Analyze correlations between features across streams.
    
    Calculates pairwise correlations to identify:
    - Redundant features (high positive correlation)
    - Hedge signals (high negative correlation)
    - Cross-stream relationships
    - Independent signal sources
    
    Useful for:
    - Feature selection and dimensionality reduction
    - Understanding signal relationships
    - Building diversified feature sets
    
    Correlation interpretation:
    - > 0.8: Very strong positive (consider removing redundant feature)
    - 0.5-0.8: Strong positive
    - 0.3-0.5: Moderate positive
    - -0.3-0.3: Weak/no correlation (independent signals)
    - < -0.7: Strong negative (hedge/diversification signal)
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        feature_types: Comma-separated feature types to analyze
        lookback_minutes: Time range for correlation (max 1440)
    
    Returns:
        XML with correlation analysis and trading insights
    """
    import json
    from .tools.feature_query_tools import get_feature_correlation_analysis
    try:
        types = [t.strip() for t in feature_types.split(",") if t.strip()]
        result = await get_feature_correlation_analysis(symbol, exchange, types, lookback_minutes)
        if result.get("success") and result.get("xml_summary"):
            return result["xml_summary"]
        return json.dumps(result, indent=2)
    except Exception as e:
        return f'<error tool="analyze_feature_correlations">{str(e)}</error>'


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Start the MCP server."""
    # Determine which mode we're running in
    use_direct = os.environ.get("USE_DIRECT_EXCHANGES", "true").lower() in ("true", "1", "yes")
    mode = "DIRECT EXCHANGE" if use_direct else "GO BACKEND"
    
    logger.info("=" * 70)
    logger.info("  CRYPTO ARBITRAGE & ANALYTICS MCP SERVER")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"  MODE: {mode}")
    logger.info("")
    
    if use_direct:
        logger.info("  DIRECT EXCHANGE MODE:")
        logger.info("    Connecting directly to 9 exchanges:")
        logger.info("    Binance Futures/Spot, Bybit Futures/Spot, OKX,")
        logger.info("    Kraken Futures, Gate.io, Hyperliquid, Pyth Oracle")
        logger.info("    No Go backend required!")
        logger.info("")
    else:
        logger.info("  GO BACKEND MODE:")
        scanner_host = os.environ.get("ARBITRAGE_SCANNER_HOST", "localhost")
        scanner_port = os.environ.get("ARBITRAGE_SCANNER_PORT", "8082")
        logger.info(f"    Scanner WebSocket: ws://{scanner_host}:{scanner_port}/ws")
        logger.info("    Start Go scanner: cd crypto-futures-arbitrage-scanner && go run main.go")
        logger.info("")
    
    logger.info("  CORE MCP TOOLS:")
    logger.info("    • analyze_crypto_arbitrage_tool - Comprehensive arbitrage analysis")
    logger.info("    • get_crypto_prices            - Current prices from all exchanges")
    logger.info("    • get_crypto_spreads           - Spread matrix between exchanges")
    logger.info("    • get_arbitrage_opportunities  - Recent detected opportunities")
    logger.info("    • compare_exchange_prices      - Compare two specific exchanges")
    logger.info("    • crypto_scanner_health        - Check scanner connectivity")
    logger.info("")
    logger.info("  DATA STREAM TOOLS:")
    logger.info("    • get_orderbook_data           - Real-time orderbook depth")
    logger.info("    • get_trades            - Recent trades and volume stats")
    logger.info("    • get_funding_rate_data        - Funding rates and sentiment")
    logger.info("    • get_liquidation_data         - Recent liquidation events")
    logger.info("    • get_open_interest_data       - Open interest per exchange")
    logger.info("    • get_mark_price_data          - Mark prices and basis")
    logger.info("    • get_24h_ticker_data          - 24h volume and statistics")
    logger.info("    • get_market_overview          - Complete market summary")
    logger.info("")
    logger.info("  ANALYTICS TOOLS (Feature Intelligence Framework):")
    logger.info("    • get_market_intelligence      - Full 5-layer market analysis")
    logger.info("    • get_institutional_pressure   - Institutional buying/selling pressure")
    logger.info("    • get_squeeze_probability      - Short/long squeeze detection")
    logger.info("    • get_market_regime            - Regime & volatility classification")
    logger.info("    • get_liquidity_analysis       - Orderbook microstructure analysis")
    logger.info("    • get_leverage_analysis        - Leverage & risk flow analysis")
    logger.info("    • get_cross_exchange_analysis  - Cross-exchange intelligence")
    logger.info("    • get_analytics_summary        - Available features & stats")
    logger.info("")
    logger.info("  STREAMING ANALYSIS TOOLS:")
    logger.info("    • stream_and_analyze           - Stream for X seconds and analyze")
    logger.info("    • quick_analyze                - Quick 10-second market snapshot")
    logger.info("    • analyze_for_duration         - Analyze for X minutes with focus")
    logger.info("")
    logger.info("  BINANCE FUTURES REST API TOOLS:")
    logger.info("    • binance_ticker_tool          - 24h ticker statistics")
    logger.info("    • binance_orderbook_tool       - Full orderbook (up to 1000 levels)")
    logger.info("    • binance_trades_tool          - Recent aggregated trades")
    logger.info("    • binance_klines_tool          - OHLCV candlestick data")
    logger.info("    • binance_open_interest_tool   - Current open interest")
    logger.info("    • binance_open_interest_history_tool - Historical OI")
    logger.info("    • binance_funding_rate_tool    - Historical funding rates")
    logger.info("    • binance_premium_index_tool   - Mark/index prices & basis")
    logger.info("    • binance_long_short_ratio_tool - Long/short positioning")
    logger.info("    • binance_taker_volume_tool    - Taker buy/sell ratio")
    logger.info("    • binance_basis_tool           - Futures basis data")
    logger.info("    • binance_liquidations_tool    - Recent liquidation orders")
    logger.info("    • binance_snapshot_tool        - Comprehensive market snapshot")
    logger.info("    • binance_full_analysis_tool   - Full analysis with signals")
    logger.info("")
    logger.info("  BYBIT SPOT REST API TOOLS:")
    logger.info("    • bybit_spot_ticker_tool       - Spot ticker with 24h stats")
    logger.info("    • bybit_spot_orderbook_tool    - Spot orderbook (up to 200 levels)")
    logger.info("    • bybit_spot_trades_tool       - Recent spot trades")
    logger.info("    • bybit_spot_klines_tool       - Spot OHLCV candlesticks")
    logger.info("    • bybit_all_spot_tickers_tool  - All spot tickers summary")
    logger.info("")
    logger.info("  BYBIT FUTURES REST API TOOLS:")
    logger.info("    • bybit_futures_ticker_tool    - Perpetual ticker + funding/OI")
    logger.info("    • bybit_futures_orderbook_tool - Futures orderbook (up to 500 levels)")
    logger.info("    • bybit_open_interest_tool     - OI history with trend analysis")
    logger.info("    • bybit_funding_rate_tool      - Funding rate history")
    logger.info("    • bybit_long_short_ratio_tool  - Long/short positioning")
    logger.info("    • bybit_historical_volatility_tool - Historical volatility (options)")
    logger.info("    • bybit_insurance_fund_tool    - Insurance fund status")
    logger.info("    • bybit_all_perpetual_tickers_tool - All perpetuals summary")
    logger.info("")
    logger.info("  BYBIT ANALYSIS TOOLS:")
    logger.info("    • bybit_derivatives_analysis_tool - Full derivatives analysis")
    logger.info("    • bybit_market_snapshot_tool   - Complete market snapshot")
    logger.info("    • bybit_instruments_info_tool  - Contract specifications")
    logger.info("    • bybit_options_overview_tool  - Options market overview")
    logger.info("    • bybit_risk_limit_tool        - Risk limit tiers")
    logger.info("    • bybit_announcements_tool     - Platform announcements")
    logger.info("    • bybit_full_market_analysis_tool - Full analysis with signals")
    logger.info("")
    logger.info("  BINANCE SPOT REST API TOOLS:")
    logger.info("    • binance_spot_ticker_tool     - 24h ticker statistics")
    logger.info("    • binance_spot_price_tool      - Current price(s)")
    logger.info("    • binance_spot_orderbook_tool  - Orderbook depth (up to 5000 levels)")
    logger.info("    • binance_spot_trades_tool     - Recent trades with flow analysis")
    logger.info("    • binance_spot_klines_tool     - OHLCV candlesticks (1s to 1M)")
    logger.info("    • binance_spot_avg_price_tool  - 5-minute average price")
    logger.info("    • binance_spot_book_ticker_tool - Best bid/ask prices")
    logger.info("    • binance_spot_agg_trades_tool - Aggregate trades")
    logger.info("    • binance_spot_exchange_info_tool - Exchange info & rules")
    logger.info("    • binance_spot_rolling_ticker_tool - Rolling window stats")
    logger.info("    • binance_spot_all_tickers_tool - All tickers + top movers")
    logger.info("    • binance_spot_snapshot_tool   - Comprehensive market snapshot")
    logger.info("    • binance_spot_full_analysis_tool - Full analysis with signals")
    logger.info("")
    logger.info("  OKX REST API TOOLS:")
    logger.info("    • okx_ticker_tool              - Single ticker (spot/swap/futures)")
    logger.info("    • okx_all_tickers_tool         - All tickers by type")
    logger.info("    • okx_index_ticker_tool        - Index price ticker")
    logger.info("    • okx_orderbook_tool           - Orderbook depth (up to 400 levels)")
    logger.info("    • okx_trades_tool              - Recent trades with flow analysis")
    logger.info("    • okx_klines_tool              - OHLCV candlesticks")
    logger.info("    • okx_funding_rate_tool        - Current funding rate")
    logger.info("    • okx_funding_rate_history_tool - Historical funding rates")
    logger.info("    • okx_open_interest_tool       - Open interest data")
    logger.info("    • okx_oi_volume_tool           - OI and volume history")
    logger.info("    • okx_long_short_ratio_tool    - Long/short account ratio")
    logger.info("    • okx_taker_volume_tool        - Taker buy/sell volume")
    logger.info("    • okx_instruments_tool         - Instrument specifications")
    logger.info("    • okx_mark_price_tool          - Mark prices")
    logger.info("    • okx_insurance_fund_tool      - Insurance fund balance")
    logger.info("    • okx_platform_volume_tool     - 24h platform volume")
    logger.info("    • okx_options_summary_tool     - Options market summary")
    logger.info("    • okx_market_snapshot_tool     - Comprehensive snapshot")
    logger.info("    • okx_full_analysis_tool       - Full analysis with signals")
    logger.info("    • okx_top_movers_tool          - Top gainers/losers")
    logger.info("")
    logger.info("  KRAKEN REST API TOOLS (21 endpoints):")
    logger.info("    • kraken_spot_ticker_tool       - Spot ticker with analysis")
    logger.info("    • kraken_all_spot_tickers_tool  - Multiple spot tickers")
    logger.info("    • kraken_spot_orderbook_tool    - Spot orderbook depth")
    logger.info("    • kraken_spot_trades_tool       - Recent spot trades")
    logger.info("    • kraken_spot_klines_tool       - Spot OHLC candlesticks")
    logger.info("    • kraken_spread_tool            - Recent spread data")
    logger.info("    • kraken_assets_tool            - Supported assets")
    logger.info("    • kraken_spot_pairs_tool        - Trading pair specs")
    logger.info("    • kraken_futures_ticker_tool    - Futures/perpetual ticker")
    logger.info("    • kraken_all_futures_tickers_tool - All futures tickers")
    logger.info("    • kraken_futures_orderbook_tool - Futures orderbook")
    logger.info("    • kraken_futures_trades_tool    - Futures trades")
    logger.info("    • kraken_futures_klines_tool    - Futures candlesticks")
    logger.info("    • kraken_futures_instruments_tool - Instrument specs")
    logger.info("    • kraken_funding_rates_tool     - All funding rates")
    logger.info("    • kraken_open_interest_tool     - All open interest")
    logger.info("    • kraken_system_status_tool     - System status")
    logger.info("    • kraken_top_movers_tool        - Top gainers/losers")
    logger.info("    • kraken_market_snapshot_tool   - Comprehensive snapshot")
    logger.info("    • kraken_full_analysis_tool     - Full analysis + signals")
    logger.info("")
    logger.info("  GATE.IO FUTURES REST API TOOLS (27 endpoints):")
    logger.info("    • gateio_futures_contracts      - All futures contract specs")
    logger.info("    • gateio_futures_contract       - Single contract info")
    logger.info("    • gateio_futures_ticker         - Futures ticker")
    logger.info("    • gateio_all_futures_tickers    - All futures tickers")
    logger.info("    • gateio_futures_orderbook      - Futures orderbook (50 levels)")
    logger.info("    • gateio_futures_trades         - Recent futures trades")
    logger.info("    • gateio_futures_klines         - Futures candlesticks")
    logger.info("    • gateio_funding_rate           - Funding rate history")
    logger.info("    • gateio_all_funding_rates      - All perpetual funding rates")
    logger.info("    • gateio_contract_stats         - OI, liquidations, L/S ratio")
    logger.info("    • gateio_open_interest          - Top contracts OI")
    logger.info("    • gateio_liquidations           - Liquidation history")
    logger.info("    • gateio_insurance_fund         - Insurance fund balance")
    logger.info("    • gateio_risk_limit_tiers       - Risk limit tiers")
    logger.info("    • gateio_delivery_contracts     - Delivery futures contracts")
    logger.info("    • gateio_delivery_ticker        - Delivery futures ticker")
    logger.info("    • gateio_options_underlyings    - Options underlying assets")
    logger.info("    • gateio_options_expirations    - Options expiration dates")
    logger.info("    • gateio_options_contracts      - Options contracts (calls/puts)")
    logger.info("    • gateio_options_tickers        - Options tickers with Greeks")
    logger.info("    • gateio_options_underlying_ticker - Underlying asset ticker")
    logger.info("    • gateio_options_orderbook      - Options orderbook")
    logger.info("    • gateio_market_snapshot        - Comprehensive market snapshot")
    logger.info("    • gateio_top_movers             - Top gainers/losers")
    logger.info("    • gateio_full_analysis          - Full analysis + signals")
    logger.info("    • gateio_perpetuals             - All perpetual contracts")
    logger.info("")
    logger.info("  HYPERLIQUID REST API TOOLS (17 endpoints):")
    logger.info("    • hyperliquid_meta              - Exchange metadata & perpetuals")
    logger.info("    • hyperliquid_all_mids          - All mid prices")
    logger.info("    • hyperliquid_ticker            - Single coin ticker")
    logger.info("    • hyperliquid_all_tickers       - All perpetual tickers")
    logger.info("    • hyperliquid_orderbook         - L2 orderbook")
    logger.info("    • hyperliquid_klines            - OHLCV candlesticks")
    logger.info("    • hyperliquid_funding_rate      - Funding rate history (hourly)")
    logger.info("    • hyperliquid_all_funding_rates - All funding rates")
    logger.info("    • hyperliquid_open_interest     - All open interest")
    logger.info("    • hyperliquid_top_movers        - Top gainers/losers")
    logger.info("    • hyperliquid_exchange_stats    - Exchange-wide statistics")
    logger.info("    • hyperliquid_spot_meta         - Spot market metadata")
    logger.info("    • hyperliquid_spot_meta_and_ctxs - Spot meta + contexts")
    logger.info("    • hyperliquid_market_snapshot   - Comprehensive snapshot")
    logger.info("    • hyperliquid_full_analysis     - Full analysis + signals")
    logger.info("    • hyperliquid_perpetuals        - All perpetual contracts")
    logger.info("    • hyperliquid_recent_trades     - Trade activity proxy")
    logger.info("")
    logger.info("  DUCKDB HISTORICAL DATA TOOLS:")
    logger.info("    • get_historical_price_data     - Query stored price history")
    logger.info("    • get_historical_trade_data     - Query stored trade data")
    logger.info("    • get_historical_funding_data   - Query funding rate history")
    logger.info("    • get_historical_liquidation_data - Query liquidation history")
    logger.info("    • get_historical_oi_data        - Query open interest history")
    logger.info("    • get_database_statistics       - Get database stats and tables")
    logger.info("    • query_historical_analytics    - Custom OHLC/volatility queries")
    logger.info("")
    logger.info("  LIVE + HISTORICAL COMBINED TOOLS:")
    logger.info("    • get_full_market_snapshot      - Live + historical snapshot")
    logger.info("    • get_price_with_historical_context - Price with OHLC context")
    logger.info("    • analyze_funding_arbitrage     - Funding rate arbitrage detection")
    logger.info("    • get_liquidation_heatmap_analysis - Liquidation distribution")
    logger.info("    • detect_price_anomalies        - Z-score anomaly detection")
    logger.info("")
    logger.info("  ADVANCED FEATURE CALCULATORS (Plugin Framework):")
    if feature_registry and feature_registry.calculators:
        for name, calc in feature_registry.calculators.items():
            logger.info(f"    • calculate_{name:25} - {calc.description}")
    else:
        logger.info("    (No feature calculators registered)")
    logger.info("")
    logger.info("  SUPPORTED SYMBOLS: BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT")
    logger.info("=" * 70)
    
    # Run the server
    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()



