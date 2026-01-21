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
from xml.sax.saxutils import escape as xml_escape

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


# ============================================================================
# NEW DATA STREAM TOOLS
# ============================================================================

async def get_orderbooks(
    symbol: Optional[str] = None,
    exchange: Optional[str] = None
) -> str:
    """
    Get orderbook depth data from exchanges.
    Shows top 10 bid/ask levels, depth, and spread for each exchange.
    
    Args:
        symbol: Specific symbol (BTCUSDT, ETHUSDT, etc.) or None for all
        exchange: Specific exchange or None for all
    
    Returns:
        XML formatted orderbook data with depth analysis
    """
    client = get_client()
    
    try:
        if not await _ensure_client_connected(client):
            return _build_error_response("CONNECTION_FAILED", "Unable to connect", [])
        
        # Only direct client supports orderbooks
        if not hasattr(client, 'get_orderbooks'):
            return _build_error_response(
                "NOT_SUPPORTED",
                "Orderbook data requires direct exchange mode",
                ["Set USE_DIRECT_EXCHANGES=true"]
            )
        
        orderbooks = await client.get_orderbooks(symbol, exchange)
        
        if not orderbooks:
            return _build_error_response(
                "NO_DATA",
                "No orderbook data available",
                ["Wait a few seconds for data to arrive", "Check symbol/exchange parameters"]
            )
        
        return _format_orderbooks_xml(orderbooks)
        
    except Exception as e:
        logger.error(f"Error in get_orderbooks: {e}", exc_info=True)
        return _build_error_response("ERROR", str(e), [])


async def get_trades(
    symbol: Optional[str] = None,
    exchange: Optional[str] = None,
    limit: int = 50
) -> str:
    """
    Get recent trades from exchanges.
    Shows price, quantity, side (buy/sell), and value for each trade.
    
    Args:
        symbol: Specific symbol or None for all
        exchange: Specific exchange or None for all
        limit: Max trades per exchange (default 50)
    
    Returns:
        XML formatted recent trades with aggregated statistics
    """
    client = get_client()
    
    try:
        if not await _ensure_client_connected(client):
            return _build_error_response("CONNECTION_FAILED", "Unable to connect", [])
        
        if not hasattr(client, 'get_trades'):
            return _build_error_response(
                "NOT_SUPPORTED",
                "Trade data requires direct exchange mode",
                ["Set USE_DIRECT_EXCHANGES=true"]
            )
        
        trades = await client.get_trades(symbol, exchange, limit)
        
        if not trades:
            return _build_error_response("NO_DATA", "No trade data available", [])
        
        return _format_trades_xml(trades)
        
    except Exception as e:
        logger.error(f"Error in get_trades: {e}", exc_info=True)
        return _build_error_response("ERROR", str(e), [])


async def get_funding_rates(symbol: Optional[str] = None) -> str:
    """
    Get funding rates from all futures exchanges.
    Shows current rate, annualized rate, and next funding time.
    
    Args:
        symbol: Specific symbol or None for all
    
    Returns:
        XML formatted funding rates with analysis
    """
    client = get_client()
    
    try:
        if not await _ensure_client_connected(client):
            return _build_error_response("CONNECTION_FAILED", "Unable to connect", [])
        
        if not hasattr(client, 'get_funding_rates'):
            return _build_error_response(
                "NOT_SUPPORTED",
                "Funding rate data requires direct exchange mode",
                ["Set USE_DIRECT_EXCHANGES=true"]
            )
        
        funding = await client.get_funding_rates(symbol)
        
        if not funding:
            return _build_error_response("NO_DATA", "No funding rate data available", [])
        
        return _format_funding_rates_xml(funding)
        
    except Exception as e:
        logger.error(f"Error in get_funding_rates: {e}", exc_info=True)
        return _build_error_response("ERROR", str(e), [])


async def get_liquidations(symbol: Optional[str] = None, limit: int = 20) -> str:
    """
    Get recent liquidation events from exchanges.
    Shows forced liquidation side, price, quantity, and value.
    
    Args:
        symbol: Specific symbol or None for all
        limit: Max liquidations to return (default 20)
    
    Returns:
        XML formatted liquidation events
    """
    client = get_client()
    
    try:
        if not await _ensure_client_connected(client):
            return _build_error_response("CONNECTION_FAILED", "Unable to connect", [])
        
        if not hasattr(client, 'get_liquidations'):
            return _build_error_response(
                "NOT_SUPPORTED",
                "Liquidation data requires direct exchange mode",
                ["Set USE_DIRECT_EXCHANGES=true"]
            )
        
        liquidations = await client.get_liquidations(symbol, limit)
        
        if not liquidations or all(not v for v in liquidations.values()):
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<liquidations>
  <status>NO_RECENT_LIQUIDATIONS</status>
  <message>No recent liquidation events detected</message>
  <timestamp>{datetime.utcnow().isoformat()}</timestamp>
</liquidations>"""
        
        return _format_liquidations_xml(liquidations)
        
    except Exception as e:
        logger.error(f"Error in get_liquidations: {e}", exc_info=True)
        return _build_error_response("ERROR", str(e), [])


async def get_open_interest(symbol: Optional[str] = None) -> str:
    """
    Get open interest data from futures exchanges.
    Shows total open interest and value per exchange.
    
    Args:
        symbol: Specific symbol or None for all
    
    Returns:
        XML formatted open interest data
    """
    client = get_client()
    
    try:
        if not await _ensure_client_connected(client):
            return _build_error_response("CONNECTION_FAILED", "Unable to connect", [])
        
        if not hasattr(client, 'get_open_interest'):
            return _build_error_response(
                "NOT_SUPPORTED",
                "Open interest data requires direct exchange mode",
                ["Set USE_DIRECT_EXCHANGES=true"]
            )
        
        oi_data = await client.get_open_interest(symbol)
        
        if not oi_data:
            return _build_error_response("NO_DATA", "No open interest data available", [])
        
        return _format_open_interest_xml(oi_data)
        
    except Exception as e:
        logger.error(f"Error in get_open_interest: {e}", exc_info=True)
        return _build_error_response("ERROR", str(e), [])


async def get_mark_prices(symbol: Optional[str] = None) -> str:
    """
    Get mark prices and basis from futures exchanges.
    Shows mark price, index price, basis, and basis percentage.
    
    Args:
        symbol: Specific symbol or None for all
    
    Returns:
        XML formatted mark price and basis data
    """
    client = get_client()
    
    try:
        if not await _ensure_client_connected(client):
            return _build_error_response("CONNECTION_FAILED", "Unable to connect", [])
        
        if not hasattr(client, 'get_mark_prices'):
            return _build_error_response(
                "NOT_SUPPORTED",
                "Mark price data requires direct exchange mode",
                ["Set USE_DIRECT_EXCHANGES=true"]
            )
        
        mark_data = await client.get_mark_prices(symbol)
        
        if not mark_data:
            return _build_error_response("NO_DATA", "No mark price data available", [])
        
        return _format_mark_prices_xml(mark_data)
        
    except Exception as e:
        logger.error(f"Error in get_mark_prices: {e}", exc_info=True)
        return _build_error_response("ERROR", str(e), [])


async def get_ticker_24h(symbol: Optional[str] = None) -> str:
    """
    Get 24-hour ticker statistics from all exchanges.
    Shows volume, high, low, and price change percentage.
    
    Args:
        symbol: Specific symbol or None for all
    
    Returns:
        XML formatted 24h statistics
    """
    client = get_client()
    
    try:
        if not await _ensure_client_connected(client):
            return _build_error_response("CONNECTION_FAILED", "Unable to connect", [])
        
        if not hasattr(client, 'get_ticker_24h'):
            return _build_error_response(
                "NOT_SUPPORTED",
                "24h ticker data requires direct exchange mode",
                ["Set USE_DIRECT_EXCHANGES=true"]
            )
        
        ticker_data = await client.get_ticker_24h(symbol)
        
        if not ticker_data:
            return _build_error_response("NO_DATA", "No 24h ticker data available", [])
        
        return _format_ticker_24h_xml(ticker_data)
        
    except Exception as e:
        logger.error(f"Error in get_ticker_24h: {e}", exc_info=True)
        return _build_error_response("ERROR", str(e), [])


async def get_market_summary(symbol: str = "BTCUSDT") -> str:
    """
    Get comprehensive market summary for a symbol.
    Combines all data: prices, orderbooks, funding, OI, 24h stats, trades, liquidations.
    
    Args:
        symbol: Symbol to analyze (default BTCUSDT)
    
    Returns:
        XML formatted comprehensive market summary
    """
    client = get_client()
    
    try:
        if not await _ensure_client_connected(client):
            return _build_error_response("CONNECTION_FAILED", "Unable to connect", [])
        
        if not hasattr(client, 'get_market_summary'):
            return _build_error_response(
                "NOT_SUPPORTED",
                "Market summary requires direct exchange mode",
                ["Set USE_DIRECT_EXCHANGES=true"]
            )
        
        summary = await client.get_market_summary(symbol)
        return _format_market_summary_xml(summary)
        
    except Exception as e:
        logger.error(f"Error in get_market_summary: {e}", exc_info=True)
        return _build_error_response("ERROR", str(e), [])


# ============================================================================
# XML Formatting Helpers for New Tools
# ============================================================================

def _format_orderbooks_xml(orderbooks: Dict) -> str:
    """Format orderbook data as XML."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<orderbooks>']
    
    for symbol, exchanges in orderbooks.items():
        lines.append(f'  <symbol name="{xml_escape(str(symbol))}">')
        for exchange, data in exchanges.items():
            spread = data.get('spread', 0)
            spread_pct = data.get('spread_pct', 0)
            bid_depth = data.get('bid_depth', 0)
            ask_depth = data.get('ask_depth', 0)
            
            lines.append(f'    <exchange name="{xml_escape(str(exchange))}">')
            lines.append(f'      <spread>${spread:.2f}</spread>')
            lines.append(f'      <spread_pct>{spread_pct:.4f}%</spread_pct>')
            lines.append(f'      <bid_depth>{bid_depth:.4f}</bid_depth>')
            lines.append(f'      <ask_depth>{ask_depth:.4f}</ask_depth>')
            lines.append(f'      <imbalance>{((bid_depth - ask_depth) / (bid_depth + ask_depth) * 100) if bid_depth + ask_depth > 0 else 0:.2f}%</imbalance>')
            
            # Top 3 bids
            lines.append('      <top_bids>')
            for bid in data.get('bids', [])[:3]:
                lines.append(f'        <level price="{bid["price"]:.2f}" qty="{bid["quantity"]:.4f}"/>')
            lines.append('      </top_bids>')
            
            # Top 3 asks
            lines.append('      <top_asks>')
            for ask in data.get('asks', [])[:3]:
                lines.append(f'        <level price="{ask["price"]:.2f}" qty="{ask["quantity"]:.4f}"/>')
            lines.append('      </top_asks>')
            
            lines.append('    </exchange>')
        lines.append('  </symbol>')
    
    lines.append(f'  <timestamp>{datetime.utcnow().isoformat()}</timestamp>')
    lines.append('</orderbooks>')
    return '\n'.join(lines)


def _format_trades_xml(trades: Dict) -> str:
    """Format trade data as XML."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<trades>']
    
    for symbol, exchanges in trades.items():
        lines.append(f'  <symbol name="{symbol}">')
        
        for exchange, trade_list in exchanges.items():
            if not trade_list:
                continue
                
            # Calculate stats
            total_volume = sum(t.get('quantity', 0) for t in trade_list)
            total_value = sum(t.get('value', 0) for t in trade_list)
            buy_volume = sum(t.get('quantity', 0) for t in trade_list if t.get('side') == 'buy')
            sell_volume = sum(t.get('quantity', 0) for t in trade_list if t.get('side') == 'sell')
            
            lines.append(f'    <exchange name="{exchange}" trade_count="{len(trade_list)}">')
            lines.append(f'      <total_volume>{total_volume:.4f}</total_volume>')
            lines.append(f'      <total_value>${total_value:,.2f}</total_value>')
            lines.append(f'      <buy_volume>{buy_volume:.4f}</buy_volume>')
            lines.append(f'      <sell_volume>{sell_volume:.4f}</sell_volume>')
            lines.append(f'      <buy_sell_ratio>{(buy_volume/sell_volume if sell_volume > 0 else 0):.2f}</buy_sell_ratio>')
            
            # Recent trades
            lines.append('      <recent_trades>')
            for trade in trade_list[:5]:
                lines.append(f'        <trade side="{trade.get("side", "")}" price="{trade.get("price", 0):.2f}" qty="{trade.get("quantity", 0):.4f}" value="${trade.get("value", 0):,.2f}"/>')
            lines.append('      </recent_trades>')
            lines.append('    </exchange>')
        
        lines.append('  </symbol>')
    
    lines.append(f'  <timestamp>{datetime.utcnow().isoformat()}</timestamp>')
    lines.append('</trades>')
    return '\n'.join(lines)


def _format_funding_rates_xml(funding: Dict) -> str:
    """Format funding rate data as XML."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<funding_rates>']
    
    for symbol, exchanges in funding.items():
        if not exchanges:
            continue
        lines.append(f'  <symbol name="{symbol}">')
        
        rates = []
        for exchange, data in exchanges.items():
            rate_pct = data.get('rate_pct', 0)
            annualized = data.get('annualized_rate', 0)
            rates.append(rate_pct)
            
            sentiment = "NEUTRAL"
            if rate_pct > 0.01:
                sentiment = "BULLISH" if rate_pct > 0.05 else "SLIGHTLY_BULLISH"
            elif rate_pct < -0.01:
                sentiment = "BEARISH" if rate_pct < -0.05 else "SLIGHTLY_BEARISH"
            
            lines.append(f'    <exchange name="{exchange}">')
            lines.append(f'      <rate_pct>{rate_pct:.4f}%</rate_pct>')
            lines.append(f'      <annualized_rate>{annualized:.2f}%</annualized_rate>')
            lines.append(f'      <sentiment>{sentiment}</sentiment>')
            lines.append('    </exchange>')
        
        # Summary
        if rates:
            avg_rate = sum(rates) / len(rates)
            lines.append(f'    <summary avg_rate="{avg_rate:.4f}%" exchanges="{len(rates)}"/>')
        
        lines.append('  </symbol>')
    
    lines.append(f'  <timestamp>{datetime.utcnow().isoformat()}</timestamp>')
    lines.append('</funding_rates>')
    return '\n'.join(lines)


def _format_liquidations_xml(liquidations: Dict) -> str:
    """Format liquidation data as XML."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<liquidations>']
    
    total_long = 0
    total_short = 0
    
    for symbol, liq_list in liquidations.items():
        if not liq_list:
            continue
        lines.append(f'  <symbol name="{symbol}" count="{len(liq_list)}">')
        
        sym_long = sum(l.get('value', 0) for l in liq_list if l.get('side', '').lower() in ('buy', 'long'))
        sym_short = sum(l.get('value', 0) for l in liq_list if l.get('side', '').lower() in ('sell', 'short'))
        total_long += sym_long
        total_short += sym_short
        
        lines.append(f'    <long_liquidations>${sym_long:,.2f}</long_liquidations>')
        lines.append(f'    <short_liquidations>${sym_short:,.2f}</short_liquidations>')
        
        for liq in liq_list[:10]:
            lines.append(f'    <liquidation side="{liq.get("side", "")}" exchange="{liq.get("exchange", "")}" price="{liq.get("price", 0):.2f}" qty="{liq.get("quantity", 0):.4f}" value="${liq.get("value", 0):,.2f}"/>')
        
        lines.append('  </symbol>')
    
    lines.append(f'  <total_long_liquidations>${total_long:,.2f}</total_long_liquidations>')
    lines.append(f'  <total_short_liquidations>${total_short:,.2f}</total_short_liquidations>')
    lines.append(f'  <timestamp>{datetime.utcnow().isoformat()}</timestamp>')
    lines.append('</liquidations>')
    return '\n'.join(lines)


def _format_open_interest_xml(oi_data: Dict) -> str:
    """Format open interest data as XML."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<open_interest>']
    
    for symbol, exchanges in oi_data.items():
        if not exchanges:
            continue
        lines.append(f'  <symbol name="{symbol}">')
        
        total_oi = 0
        for exchange, data in exchanges.items():
            oi = data.get('open_interest', 0)
            oi_value = data.get('open_interest_value', 0)
            total_oi += oi
            
            lines.append(f'    <exchange name="{exchange}">')
            lines.append(f'      <open_interest>{oi:,.0f}</open_interest>')
            if oi_value > 0:
                lines.append(f'      <open_interest_value>${oi_value:,.2f}</open_interest_value>')
            lines.append('    </exchange>')
        
        lines.append(f'    <total_oi>{total_oi:,.0f}</total_oi>')
        lines.append('  </symbol>')
    
    lines.append(f'  <timestamp>{datetime.utcnow().isoformat()}</timestamp>')
    lines.append('</open_interest>')
    return '\n'.join(lines)


def _format_mark_prices_xml(mark_data: Dict) -> str:
    """Format mark price data as XML."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<mark_prices>']
    
    for symbol, exchanges in mark_data.items():
        if not exchanges:
            continue
        lines.append(f'  <symbol name="{symbol}">')
        
        for exchange, data in exchanges.items():
            mark = data.get('mark_price', 0)
            index = data.get('index_price', 0)
            basis = data.get('basis', 0)
            basis_pct = data.get('basis_pct', 0)
            
            # Basis interpretation
            if basis_pct > 0.05:
                basis_status = "CONTANGO (futures premium)"
            elif basis_pct < -0.05:
                basis_status = "BACKWARDATION (futures discount)"
            else:
                basis_status = "NEUTRAL"
            
            lines.append(f'    <exchange name="{exchange}">')
            lines.append(f'      <mark_price>${mark:,.2f}</mark_price>')
            if index > 0:
                lines.append(f'      <index_price>${index:,.2f}</index_price>')
                lines.append(f'      <basis>${basis:.2f}</basis>')
                lines.append(f'      <basis_pct>{basis_pct:.4f}%</basis_pct>')
                lines.append(f'      <basis_status>{basis_status}</basis_status>')
            lines.append('    </exchange>')
        
        lines.append('  </symbol>')
    
    lines.append(f'  <timestamp>{datetime.utcnow().isoformat()}</timestamp>')
    lines.append('</mark_prices>')
    return '\n'.join(lines)


def _format_ticker_24h_xml(ticker_data: Dict) -> str:
    """Format 24h ticker data as XML."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<ticker_24h>']
    
    for symbol, exchanges in ticker_data.items():
        if not exchanges:
            continue
        lines.append(f'  <symbol name="{symbol}">')
        
        for exchange, data in exchanges.items():
            volume = data.get('volume', 0)
            quote_vol = data.get('quote_volume', 0)
            high = data.get('high_24h', 0)
            low = data.get('low_24h', 0)
            change_pct = data.get('price_change_pct', 0)
            trades = data.get('trades_count', 0)
            
            lines.append(f'    <exchange name="{exchange}">')
            lines.append(f'      <volume>{volume:,.2f}</volume>')
            if quote_vol > 0:
                lines.append(f'      <quote_volume>${quote_vol:,.2f}</quote_volume>')
            if high > 0:
                lines.append(f'      <high_24h>${high:,.2f}</high_24h>')
            if low > 0:
                lines.append(f'      <low_24h>${low:,.2f}</low_24h>')
            lines.append(f'      <price_change_pct>{change_pct:.2f}%</price_change_pct>')
            if trades > 0:
                lines.append(f'      <trades_count>{trades:,}</trades_count>')
            lines.append('    </exchange>')
        
        lines.append('  </symbol>')
    
    lines.append(f'  <timestamp>{datetime.utcnow().isoformat()}</timestamp>')
    lines.append('</ticker_24h>')
    return '\n'.join(lines)


def _format_market_summary_xml(summary: Dict) -> str:
    """Format comprehensive market summary as XML."""
    symbol = summary.get('symbol', 'UNKNOWN')
    
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', f'<market_summary symbol="{symbol}">']
    
    # Prices section
    prices = summary.get('prices', {})
    if prices:
        lines.append('  <prices>')
        for ex, data in prices.items():
            price = data.get('price', 0)
            if price > 0:
                lines.append(f'    <exchange name="{ex}">${price:,.2f}</exchange>')
        lines.append('  </prices>')
    
    # Funding rates
    funding = summary.get('funding_rates', {})
    if funding:
        lines.append('  <funding_rates>')
        for ex, data in funding.items():
            rate = data.get('rate_pct', 0)
            lines.append(f'    <exchange name="{ex}">{rate:.4f}%</exchange>')
        lines.append('  </funding_rates>')
    
    # Open interest
    oi = summary.get('open_interest', {})
    if oi:
        lines.append('  <open_interest>')
        for ex, data in oi.items():
            oi_val = data.get('open_interest', 0)
            if oi_val > 0:
                lines.append(f'    <exchange name="{ex}">{oi_val:,.0f}</exchange>')
        lines.append('  </open_interest>')
    
    # 24h stats
    ticker = summary.get('ticker_24h', {})
    if ticker:
        lines.append('  <volume_24h>')
        for ex, data in ticker.items():
            vol = data.get('volume', 0)
            if vol > 0:
                lines.append(f'    <exchange name="{ex}">{vol:,.2f}</exchange>')
        lines.append('  </volume_24h>')
    
    # Trade counts
    trade_counts = summary.get('recent_trades_count', {})
    if trade_counts:
        total = sum(trade_counts.values())
        lines.append(f'  <recent_trades_total>{total}</recent_trades_total>')
    
    # Recent liquidations
    liqs = summary.get('recent_liquidations', [])
    if liqs:
        lines.append(f'  <recent_liquidations count="{len(liqs)}">')
        for liq in liqs[:3]:
            lines.append(f'    <liquidation side="{liq.get("side", "")}" value="${liq.get("value", 0):,.2f}"/>')
        lines.append('  </recent_liquidations>')
    
    lines.append(f'  <timestamp>{summary.get("timestamp", datetime.utcnow().isoformat())}</timestamp>')
    lines.append('</market_summary>')
    return '\n'.join(lines)


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
