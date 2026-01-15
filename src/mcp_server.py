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

# Initialize MCP server
mcp = FastMCP("Crypto Arbitrage Analysis Server")


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



