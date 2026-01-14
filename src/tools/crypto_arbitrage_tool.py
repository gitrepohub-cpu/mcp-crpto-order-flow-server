"""
MCP tool implementations for crypto arbitrage analysis.
Supports both direct exchange connections and Go scanner backend.

Set USE_DIRECT_EXCHANGES=true (default) for cloud deployment without Go backend.
Set USE_DIRECT_EXCHANGES=false to use the Go arbitrage scanner.
"""

import asyncio
import logging
import os
from typing import Optional, List, Dict
from datetime import datetime

# Determine which client to use
USE_DIRECT_MODE = os.environ.get("USE_DIRECT_EXCHANGES", "true").lower() in ("true", "1", "yes")

if USE_DIRECT_MODE:
    from src.storage.direct_exchange_client import get_direct_client as get_client
    CLIENT_MODE = "direct_exchange"
else:
    from src.storage.websocket_client import get_arbitrage_client as get_client
    CLIENT_MODE = "go_backend"

from src.formatters.crypto_xml_formatter import CryptoArbitrageFormatter

logger = logging.getLogger(__name__)
logger.info(f"Crypto arbitrage tools using: {CLIENT_MODE} mode")


async def _ensure_client_connected(client):
    """Ensure the client is connected and has data."""
    # For direct client, check _started; for websocket client, check connected
    is_connected = getattr(client, '_started', False) or getattr(client, 'connected', False)
    
    if not is_connected:
        connected = await client.connect()
        if not connected:
            return False
        await asyncio.sleep(2.0)  # Wait for initial data
    return True


async def analyze_crypto_arbitrage(
    symbol: str = "BTCUSDT",
    min_profit_threshold: float = 0.05,
    include_spreads: bool = True,
    include_opportunities: bool = True,
    opportunity_limit: int = 10
) -> str:
    """
    Comprehensive crypto arbitrage analysis for a symbol.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT)
        min_profit_threshold: Minimum profit % to consider (default 0.05%)
        include_spreads: Include spread matrix in response
        include_opportunities: Include recent opportunities
        opportunity_limit: Max opportunities to return
    
    Returns:
        XML formatted analysis suitable for LLM consumption
    """
    client = get_client()
    formatter = CryptoArbitrageFormatter()
    
    try:
        # Ensure connection
        if not await _ensure_client_connected(client):
            suggestions = [
                "Ensure network connectivity to exchanges"
            ]
            if CLIENT_MODE == "go_backend":
                suggestions = [
                    "Ensure the Go scanner is running: go run main.go",
                    "Scanner should be at http://localhost:8082",
                    "Check ARBITRAGE_SCANNER_HOST and ARBITRAGE_SCANNER_PORT env vars"
                ]
            return _build_error_response(
                "CONNECTION_FAILED",
                "Unable to connect to data source",
                suggestions
            )
        
        # Gather data
        prices = await client.get_prices_snapshot(symbol)
        spreads = await client.get_spreads_snapshot(symbol) if include_spreads else {}
        opportunities = await client.get_arbitrage_opportunities(
            symbol=symbol,
            min_profit=min_profit_threshold,
            limit=opportunity_limit
        ) if include_opportunities else []
        
        # Build context
        context = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "prices": prices.get(symbol, {}),
            "spreads": spreads,
            "opportunities": opportunities,
            "analysis": _generate_analysis(prices.get(symbol, {}), spreads, opportunities)
        }
        
        return formatter.format_arbitrage_analysis(context)
        
    except asyncio.TimeoutError:
        return _build_error_response(
            "TIMEOUT",
            "Request timed out waiting for arbitrage data",
            [
                "The scanner may be under heavy load",
                "Try again in a few seconds"
            ]
        )
    except Exception as e:
        logger.error(f"Error in analyze_crypto_arbitrage: {e}", exc_info=True)
        return _build_error_response(
            "INTERNAL_ERROR",
            str(e),
            ["Check the MCP server logs for details"]
        )


async def get_exchange_prices(
    symbol: Optional[str] = None,
    sources: Optional[List[str]] = None
) -> str:
    """
    Get current prices from all exchanges.
    
    Args:
        symbol: Optional specific symbol (None = all symbols)
        sources: Optional list of sources to filter
    
    Returns:
        XML formatted price data
    """
    client = get_client()
    formatter = CryptoArbitrageFormatter()
    
    try:
        if not await _ensure_client_connected(client):
            return _build_error_response("CONNECTION_FAILED", "Unable to connect", [])
        
        prices = await client.get_prices_snapshot(symbol)
        
        # Filter sources if specified
        if sources and prices:
            for sym in list(prices.keys()):
                if sym in prices and isinstance(prices[sym], dict):
                    prices[sym] = {
                        src: data for src, data in prices[sym].items()
                        if src in sources
                    }
        
        if not prices:
            return _build_error_response(
                "NO_DATA",
                "No price data available yet",
                [
                    "The scanner may still be connecting to exchanges",
                    "Wait a few seconds and try again"
                ]
            )
        
        return formatter.format_prices(prices)
        
    except Exception as e:
        logger.error(f"Error in get_exchange_prices: {e}", exc_info=True)
        return _build_error_response("ERROR", str(e), [])


async def get_spread_matrix(symbol: str = "BTCUSDT") -> str:
    """
    Get the current spread matrix showing price differences between exchanges.
    
    Args:
        symbol: Trading pair to analyze
    
    Returns:
        XML formatted spread matrix
    """
    client = get_client()
    formatter = CryptoArbitrageFormatter()
    
    try:
        if not await _ensure_client_connected(client):
            return _build_error_response("CONNECTION_FAILED", "Unable to connect", [])
        
        spreads = await client.get_spreads_snapshot(symbol)
        prices = await client.get_prices_snapshot(symbol)
        
        if not spreads:
            return _build_error_response(
                "NO_SPREAD_DATA",
                f"No spread data available for {symbol}",
                [
                    "Ensure the symbol is being tracked (BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT)",
                    "The scanner needs a moment to calculate spreads"
                ]
            )
        
        return formatter.format_spread_matrix(symbol, spreads, prices.get(symbol, {}))
        
    except Exception as e:
        logger.error(f"Error in get_spread_matrix: {e}", exc_info=True)
        return _build_error_response("ERROR", str(e), [])


async def get_recent_opportunities(
    symbol: Optional[str] = None,
    min_profit: float = 0.0,
    limit: int = 20
) -> str:
    """
    Get recent arbitrage opportunities.
    
    Args:
        symbol: Optional filter by symbol
        min_profit: Minimum profit percentage
        limit: Maximum opportunities to return
    
    Returns:
        XML formatted opportunities list
    """
    client = get_client()
    formatter = CryptoArbitrageFormatter()
    
    try:
        if not await _ensure_client_connected(client):
            return _build_error_response("CONNECTION_FAILED", "Unable to connect", [])
        
        opportunities = await client.get_arbitrage_opportunities(
            symbol=symbol,
            min_profit=min_profit,
            limit=limit
        )
        
        return formatter.format_opportunities(opportunities)
        
    except Exception as e:
        logger.error(f"Error in get_recent_opportunities: {e}", exc_info=True)
        return _build_error_response("ERROR", str(e), [])


async def arbitrage_scanner_health() -> str:
    """
    Check the health of the arbitrage scanner connection.
    
    Returns:
        XML formatted health status
    """
    client = get_client()
    formatter = CryptoArbitrageFormatter()
    
    try:
        health = await client.health_check()
        # Add mode information
        health["client_mode"] = CLIENT_MODE
        return formatter.format_health_status(health)
    except Exception as e:
        logger.error(f"Error in health check: {e}", exc_info=True)
        suggestions = ["Check network connectivity"]
        if CLIENT_MODE == "go_backend":
            suggestions.insert(0, "Ensure the Go arbitrage scanner is running")
        return _build_error_response(
            "HEALTH_CHECK_FAILED",
            str(e),
            suggestions
        )


def _generate_analysis(
    prices: Dict,
    spreads: Dict,
    opportunities: List[Dict]
) -> Dict:
    """Generate analytical insights from the data."""
    analysis = {
        "market_status": "UNKNOWN",
        "spread_assessment": "Unable to assess",
        "recommendation": "",
        "key_insights": []
    }
    
    if not prices:
        analysis["market_status"] = "NO_DATA"
        analysis["recommendation"] = "Unable to analyze - no price data available. Ensure scanner is connected."
        return analysis
    
    # Calculate price range
    price_values = []
    for p in prices.values():
        if isinstance(p, dict):
            val = p.get("price", 0)
        else:
            val = p
        if val and val > 0:
            price_values.append(val)
    
    if price_values:
        min_price = min(price_values)
        max_price = max(price_values)
        spread_pct = ((max_price - min_price) / min_price) * 100 if min_price > 0 else 0
        
        analysis["price_range"] = {
            "min": min_price,
            "max": max_price,
            "spread_pct": round(spread_pct, 4)
        }
        
        # Assess market status based on spread
        if spread_pct < 0.01:
            analysis["market_status"] = "HIGHLY_EFFICIENT"
            analysis["spread_assessment"] = "Minimal price discrepancies across exchanges"
            analysis["key_insights"].append("Market is tightly arbitraged - professional traders are active")
        elif spread_pct < 0.05:
            analysis["market_status"] = "EFFICIENT"
            analysis["spread_assessment"] = "Low arbitrage potential after fees"
            analysis["key_insights"].append("Small spreads exist but likely consumed by trading fees")
        elif spread_pct < 0.15:
            analysis["market_status"] = "MODERATE_INEFFICIENCY"
            analysis["spread_assessment"] = "Potential arbitrage opportunities exist"
            analysis["key_insights"].append("Spreads may be exploitable depending on fee structure")
        else:
            analysis["market_status"] = "INEFFICIENT"
            analysis["spread_assessment"] = "Significant price discrepancies detected"
            analysis["key_insights"].append("Large spreads suggest possible arbitrage or data delays")
    
    # Analyze opportunities
    if opportunities:
        profits = [o.get("profit_pct", 0) for o in opportunities]
        avg_profit = sum(profits) / len(profits)
        best_profit = max(profits)
        
        analysis["opportunity_stats"] = {
            "count": len(opportunities),
            "avg_profit_pct": round(avg_profit, 4),
            "best_profit_pct": round(best_profit, 4)
        }
        
        # Find most common exchange pairs
        pairs: Dict[str, int] = {}
        for opp in opportunities:
            pair = f"{opp.get('buy_source')} → {opp.get('sell_source')}"
            pairs[pair] = pairs.get(pair, 0) + 1
        
        if pairs:
            best_pair = max(pairs.keys(), key=lambda k: pairs[k])
            analysis["key_insights"].append(
                f"Most frequent arbitrage route: {best_pair} ({pairs[best_pair]} occurrences)"
            )
        
        # Identify which exchanges tend to be cheaper/more expensive
        buy_sources: Dict[str, int] = {}
        sell_sources: Dict[str, int] = {}
        for opp in opportunities:
            buy_src = opp.get("buy_source", "")
            sell_src = opp.get("sell_source", "")
            buy_sources[buy_src] = buy_sources.get(buy_src, 0) + 1
            sell_sources[sell_src] = sell_sources.get(sell_src, 0) + 1
        
        if buy_sources:
            cheapest = max(buy_sources.keys(), key=lambda k: buy_sources[k])
            analysis["key_insights"].append(f"Cheapest exchange (buy): {cheapest}")
        
        if sell_sources:
            expensive = max(sell_sources.keys(), key=lambda k: sell_sources[k])
            analysis["key_insights"].append(f"Most expensive (sell): {expensive}")
    
    # Generate recommendation
    if analysis["market_status"] == "INEFFICIENT":
        analysis["recommendation"] = "Market shows significant inefficiencies. Monitor closely for executable opportunities. Consider latency and fees."
    elif analysis["market_status"] == "MODERATE_INEFFICIENCY":
        analysis["recommendation"] = "Occasional opportunities may arise. Automated monitoring recommended. Factor in exchange fees (~0.02-0.1%)."
    elif analysis["market_status"] == "EFFICIENT":
        analysis["recommendation"] = "Market is relatively efficient. Arbitrage unlikely to cover transaction costs for retail traders."
    else:
        analysis["recommendation"] = "Market is highly efficient. Focus on other strategies or wait for volatility events."
    
    return analysis


def _build_error_response(error_type: str, message: str, suggestions: List[str]) -> str:
    """Build XML error response."""
    suggestions_xml = "\n".join(f"    <suggestion>{s}</suggestion>" for s in suggestions)
    suggestions_section = f"\n  <suggestions>\n{suggestions_xml}\n  </suggestions>" if suggestions else ""
    
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<crypto_arbitrage_error type="{error_type}">
  <message>{message}</message>{suggestions_section}
  <timestamp>{datetime.utcnow().isoformat()}</timestamp>
</crypto_arbitrage_error>"""
