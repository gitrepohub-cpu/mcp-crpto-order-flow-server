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
    arbitrage_scanner_health
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
        
        if not client.connected:
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
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Start the MCP server."""
    # Determine which mode we're running in
    use_direct = os.environ.get("USE_DIRECT_EXCHANGES", "true").lower() in ("true", "1", "yes")
    mode = "DIRECT EXCHANGE" if use_direct else "GO BACKEND"
    
    logger.info("=" * 70)
    logger.info("  CRYPTO ARBITRAGE MCP SERVER")
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
    
    logger.info("  AVAILABLE MCP TOOLS:")
    logger.info("    • analyze_crypto_arbitrage_tool - Comprehensive arbitrage analysis")
    logger.info("    • get_crypto_prices            - Current prices from all exchanges")
    logger.info("    • get_crypto_spreads           - Spread matrix between exchanges")
    logger.info("    • get_arbitrage_opportunities  - Recent detected opportunities")
    logger.info("    • compare_exchange_prices      - Compare two specific exchanges")
    logger.info("    • crypto_scanner_health        - Check scanner connectivity")
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
