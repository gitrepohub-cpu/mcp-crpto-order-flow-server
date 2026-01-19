"""
Gate.io MCP Tool Wrappers
Provides tool functions for all Gate.io REST API endpoints
"""

import asyncio
from typing import Optional, Dict, Any, List, Union
import logging

from ..storage.gateio_rest_client import (
    GateioRESTClient,
    get_gateio_rest_client,
    GateioSettle,
    GateioInterval,
    GateioContractStatInterval
)

logger = logging.getLogger(__name__)


# ==================== PERPETUAL FUTURES TOOLS ====================

async def gateio_futures_contracts_tool(
    settle: str = "usdt"
) -> Dict[str, Any]:
    """
    Get all Gate.io futures contracts
    
    Args:
        settle: Settlement currency - 'usdt' or 'btc'
    
    Returns:
        List of all futures contracts with specifications
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    contracts = await client.get_futures_contracts(settle_enum)
    
    if isinstance(contracts, list):
        # Format contracts
        formatted = []
        for c in contracts:
            formatted.append({
                "contract": c.get("name"),
                "type": c.get("type"),
                "quanto_multiplier": float(c.get("quanto_multiplier", 1)),
                "leverage_min": c.get("leverage_min"),
                "leverage_max": c.get("leverage_max"),
                "maintenance_rate": float(c.get("maintenance_rate", 0)),
                "mark_type": c.get("mark_type"),
                "mark_price": float(c.get("mark_price", 0)),
                "index_price": float(c.get("index_price", 0)),
                "last_price": float(c.get("last_price", 0)),
                "funding_rate": float(c.get("funding_rate", 0)),
                "funding_interval": c.get("funding_interval"),
                "order_size_min": c.get("order_size_min"),
                "order_size_max": c.get("order_size_max"),
                "orders_limit": c.get("orders_limit")
            })
        
        return {
            "exchange": "gateio",
            "settle": settle,
            "contract_count": len(formatted),
            "contracts": formatted
        }
    
    return {"error": "Failed to fetch contracts"}


async def gateio_futures_contract_tool(
    contract: str,
    settle: str = "usdt"
) -> Dict[str, Any]:
    """
    Get single Gate.io futures contract info
    
    Args:
        contract: Contract name (e.g., 'BTC_USDT')
        settle: Settlement currency
    
    Returns:
        Contract specifications and current state
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    info = await client.get_futures_contract(contract, settle_enum)
    
    if isinstance(info, dict) and "error" not in info:
        return {
            "exchange": "gateio",
            "contract": contract,
            "settle": settle,
            "specifications": {
                "type": info.get("type"),
                "quanto_multiplier": float(info.get("quanto_multiplier", 1)),
                "leverage_min": info.get("leverage_min"),
                "leverage_max": info.get("leverage_max"),
                "maintenance_rate": float(info.get("maintenance_rate", 0)),
                "funding_rate": float(info.get("funding_rate", 0)),
                "funding_interval": info.get("funding_interval"),
                "order_size_min": info.get("order_size_min"),
                "order_size_max": info.get("order_size_max"),
                "mark_price": float(info.get("mark_price", 0)),
                "index_price": float(info.get("index_price", 0)),
                "last_price": float(info.get("last_price", 0))
            }
        }
    
    return info


async def gateio_futures_ticker_tool(
    contract: str,
    settle: str = "usdt"
) -> Dict[str, Any]:
    """
    Get Gate.io futures ticker
    
    Args:
        contract: Contract name (e.g., 'BTC_USDT')
        settle: Settlement currency
    
    Returns:
        Current ticker data with price and volume info
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    ticker = await client.get_futures_ticker(contract, settle_enum)
    
    if isinstance(ticker, dict) and "error" not in ticker:
        return {
            "exchange": "gateio",
            "contract": contract,
            "settle": settle,
            "ticker": {
                "last_price": float(ticker.get("last", 0)),
                "mark_price": float(ticker.get("mark_price", 0)),
                "index_price": float(ticker.get("index_price", 0)),
                "funding_rate": float(ticker.get("funding_rate", 0)),
                "funding_rate_pct": float(ticker.get("funding_rate", 0)) * 100,
                "volume_24h": float(ticker.get("volume_24h", 0)),
                "volume_24h_usd": float(ticker.get("volume_24h_usd", 0)),
                "change_pct": float(ticker.get("change_percentage", 0)),
                "high_24h": float(ticker.get("high_24h", 0)),
                "low_24h": float(ticker.get("low_24h", 0)),
                "total_size": float(ticker.get("total_size", 0))
            }
        }
    
    return ticker


async def gateio_all_futures_tickers_tool(
    settle: str = "usdt"
) -> Dict[str, Any]:
    """
    Get all Gate.io futures tickers
    
    Args:
        settle: Settlement currency - 'usdt' or 'btc'
    
    Returns:
        All futures tickers sorted by volume
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    tickers = await client.get_futures_tickers(settle_enum)
    
    if isinstance(tickers, list):
        formatted = []
        for t in tickers:
            formatted.append({
                "contract": t.get("contract"),
                "last_price": float(t.get("last", 0)),
                "mark_price": float(t.get("mark_price", 0)),
                "index_price": float(t.get("index_price", 0)),
                "funding_rate": float(t.get("funding_rate", 0)),
                "funding_rate_pct": float(t.get("funding_rate", 0)) * 100,
                "volume_24h": float(t.get("volume_24h", 0)),
                "volume_24h_usd": float(t.get("volume_24h_usd", 0)),
                "change_pct": float(t.get("change_percentage", 0)),
                "high_24h": float(t.get("high_24h", 0)),
                "low_24h": float(t.get("low_24h", 0))
            })
        
        # Sort by volume
        formatted.sort(key=lambda x: x["volume_24h_usd"], reverse=True)
        
        return {
            "exchange": "gateio",
            "settle": settle,
            "ticker_count": len(formatted),
            "tickers": formatted
        }
    
    return {"error": "Failed to fetch tickers"}


async def gateio_futures_orderbook_tool(
    contract: str,
    settle: str = "usdt",
    limit: int = 50
) -> Dict[str, Any]:
    """
    Get Gate.io futures orderbook
    
    Args:
        contract: Contract name (e.g., 'BTC_USDT')
        settle: Settlement currency
        limit: Depth limit (max 50)
    
    Returns:
        Order book with bids and asks
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    orderbook = await client.get_futures_orderbook(contract, settle_enum, limit=min(limit, 50))
    
    if isinstance(orderbook, dict) and "asks" in orderbook:
        asks = orderbook.get("asks", [])
        bids = orderbook.get("bids", [])
        
        # Calculate metrics
        bid_vol = sum(float(b.get("s", 0)) for b in bids[:10]) if bids else 0
        ask_vol = sum(float(a.get("s", 0)) for a in asks[:10]) if asks else 0
        
        best_bid = float(bids[0].get("p", 0)) if bids else 0
        best_ask = float(asks[0].get("p", 0)) if asks else 0
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
        
        return {
            "exchange": "gateio",
            "contract": contract,
            "settle": settle,
            "orderbook": {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "spread_pct": round(spread_pct, 4),
                "bid_depth_10": bid_vol,
                "ask_depth_10": ask_vol,
                "imbalance_pct": round((bid_vol - ask_vol) / (bid_vol + ask_vol) * 100, 2) if (bid_vol + ask_vol) > 0 else 0,
                "bid_count": len(bids),
                "ask_count": len(asks)
            },
            "bids": [{"price": float(b.get("p", 0)), "size": float(b.get("s", 0))} for b in bids[:20]],
            "asks": [{"price": float(a.get("p", 0)), "size": float(a.get("s", 0))} for a in asks[:20]]
        }
    
    return {"error": "Failed to fetch orderbook"}


async def gateio_futures_trades_tool(
    contract: str,
    settle: str = "usdt",
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get Gate.io futures recent trades
    
    Args:
        contract: Contract name (e.g., 'BTC_USDT')
        settle: Settlement currency
        limit: Number of trades (max 1000)
    
    Returns:
        Recent trades with side and size
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    trades = await client.get_futures_trades(contract, settle_enum, limit=min(limit, 1000))
    
    if isinstance(trades, list):
        formatted = []
        buy_vol = 0
        sell_vol = 0
        
        for t in trades:
            size = float(t.get("size", 0))
            price = float(t.get("price", 0))
            
            # Gate.io: size > 0 = buy, size < 0 = sell
            if size >= 0:
                buy_vol += abs(size)
                side = "buy"
            else:
                sell_vol += abs(size)
                side = "sell"
            
            formatted.append({
                "price": price,
                "size": abs(size),
                "side": side,
                "timestamp": t.get("create_time"),
                "id": t.get("id")
            })
        
        total_vol = buy_vol + sell_vol
        
        return {
            "exchange": "gateio",
            "contract": contract,
            "settle": settle,
            "trade_count": len(formatted),
            "statistics": {
                "buy_volume": buy_vol,
                "sell_volume": sell_vol,
                "total_volume": total_vol,
                "buy_pct": round(buy_vol / total_vol * 100, 2) if total_vol > 0 else 0,
                "sell_pct": round(sell_vol / total_vol * 100, 2) if total_vol > 0 else 0
            },
            "trades": formatted[:50]  # Return first 50
        }
    
    return {"error": "Failed to fetch trades"}


async def gateio_futures_klines_tool(
    contract: str,
    interval: str = "1h",
    settle: str = "usdt",
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get Gate.io futures candlesticks/OHLCV
    
    Args:
        contract: Contract name (e.g., 'BTC_USDT')
        interval: Kline interval (10s, 1m, 5m, 15m, 30m, 1h, 4h, 8h, 1d, 7d, 30d)
        settle: Settlement currency
        limit: Number of candles (max 2000)
    
    Returns:
        Candlestick data with OHLCV
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    interval_map = {
        "10s": GateioInterval.SEC_10,
        "1m": GateioInterval.MIN_1,
        "5m": GateioInterval.MIN_5,
        "15m": GateioInterval.MIN_15,
        "30m": GateioInterval.MIN_30,
        "1h": GateioInterval.HOUR_1,
        "4h": GateioInterval.HOUR_4,
        "8h": GateioInterval.HOUR_8,
        "1d": GateioInterval.DAY_1,
        "7d": GateioInterval.WEEK_1,
        "30d": GateioInterval.MONTH_1
    }
    
    interval_enum = interval_map.get(interval, GateioInterval.HOUR_1)
    klines = await client.get_futures_candlesticks(contract, settle_enum, interval_enum, min(limit, 2000))
    
    if isinstance(klines, list):
        formatted = []
        for k in klines:
            formatted.append({
                "timestamp": k.get("t"),
                "open": float(k.get("o", 0)),
                "high": float(k.get("h", 0)),
                "low": float(k.get("l", 0)),
                "close": float(k.get("c", 0)),
                "volume": float(k.get("v", 0)),
                "sum": float(k.get("sum", 0))  # Quote volume
            })
        
        # Calculate price change
        if len(formatted) >= 2:
            first_close = formatted[0]["close"]
            last_close = formatted[-1]["close"]
            change_pct = ((last_close - first_close) / first_close * 100) if first_close > 0 else 0
        else:
            change_pct = 0
        
        return {
            "exchange": "gateio",
            "contract": contract,
            "settle": settle,
            "interval": interval,
            "candle_count": len(formatted),
            "price_change_pct": round(change_pct, 2),
            "klines": formatted
        }
    
    return {"error": "Failed to fetch klines"}


async def gateio_funding_rate_tool(
    contract: str,
    settle: str = "usdt",
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get Gate.io funding rate history
    
    Args:
        contract: Contract name (e.g., 'BTC_USDT')
        settle: Settlement currency
        limit: Number of records (max 1000)
    
    Returns:
        Funding rate history
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    funding = await client.get_funding_rate(contract, settle_enum, min(limit, 1000))
    
    if isinstance(funding, list):
        formatted = []
        total_rate = 0
        
        for f in funding:
            rate = float(f.get("r", 0))
            total_rate += rate
            formatted.append({
                "rate": rate,
                "rate_pct": rate * 100,
                "timestamp": f.get("t")
            })
        
        avg_rate = total_rate / len(funding) if funding else 0
        latest_rate = float(funding[0].get("r", 0)) if funding else 0
        
        # Annualized rate (assuming 8h funding)
        annual_rate = latest_rate * 3 * 365 * 100
        
        return {
            "exchange": "gateio",
            "contract": contract,
            "settle": settle,
            "current_rate": latest_rate,
            "current_rate_pct": latest_rate * 100,
            "average_rate_pct": avg_rate * 100,
            "annualized_rate_pct": round(annual_rate, 2),
            "funding_history_count": len(formatted),
            "funding_history": formatted[:24]  # Last 24 funding periods
        }
    
    return {"error": "Failed to fetch funding rate"}


async def gateio_all_funding_rates_tool(
    settle: str = "usdt"
) -> Dict[str, Any]:
    """
    Get funding rates for all Gate.io perpetuals
    
    Args:
        settle: Settlement currency
    
    Returns:
        All funding rates sorted by absolute value
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    funding_data = await client.get_funding_rates_all(settle_enum)
    
    if isinstance(funding_data, list):
        # Calculate stats
        positive = [f for f in funding_data if f["funding_rate"] > 0]
        negative = [f for f in funding_data if f["funding_rate"] < 0]
        
        return {
            "exchange": "gateio",
            "settle": settle,
            "total_contracts": len(funding_data),
            "positive_funding_count": len(positive),
            "negative_funding_count": len(negative),
            "highest_funding": funding_data[:10] if funding_data else [],
            "lowest_funding": sorted(funding_data, key=lambda x: x["funding_rate"])[:10] if funding_data else []
        }
    
    return {"error": "Failed to fetch funding rates"}


async def gateio_contract_stats_tool(
    contract: str,
    settle: str = "usdt",
    interval: str = "1h",
    limit: int = 24
) -> Dict[str, Any]:
    """
    Get Gate.io contract statistics (OI, liquidations, L/S ratio)
    
    Args:
        contract: Contract name (e.g., 'BTC_USDT')
        settle: Settlement currency
        interval: Stats interval (5m, 1h, 1d)
        limit: Number of records (max 100)
    
    Returns:
        Contract statistics with open interest and liquidations
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    interval_map = {
        "5m": GateioContractStatInterval.MIN_5,
        "1h": GateioContractStatInterval.HOUR_1,
        "1d": GateioContractStatInterval.DAY_1
    }
    interval_enum = interval_map.get(interval, GateioContractStatInterval.HOUR_1)
    
    stats = await client.get_contract_stats(contract, settle_enum, interval_enum, min(limit, 100))
    
    if isinstance(stats, list) and stats:
        latest = stats[0]
        
        formatted = []
        for s in stats:
            formatted.append({
                "timestamp": s.get("time"),
                "open_interest": float(s.get("open_interest", 0)),
                "open_interest_usd": float(s.get("open_interest_usd", 0)),
                "lsr_taker": float(s.get("lsr_taker", 0)),
                "lsr_account": float(s.get("lsr_account", 0)),
                "top_lsr_account": float(s.get("top_lsr_account", 0)),
                "top_lsr_size": float(s.get("top_lsr_size", 0)),
                "long_liq_size": float(s.get("long_liq_size", 0)),
                "short_liq_size": float(s.get("short_liq_size", 0)),
                "long_liq_usd": float(s.get("long_liq_usd", 0)),
                "short_liq_usd": float(s.get("short_liq_usd", 0))
            })
        
        return {
            "exchange": "gateio",
            "contract": contract,
            "settle": settle,
            "interval": interval,
            "current_stats": {
                "open_interest": float(latest.get("open_interest", 0)),
                "open_interest_usd": float(latest.get("open_interest_usd", 0)),
                "long_short_ratio_taker": float(latest.get("lsr_taker", 0)),
                "long_short_ratio_account": float(latest.get("lsr_account", 0)),
                "top_trader_lsr_account": float(latest.get("top_lsr_account", 0)),
                "top_trader_lsr_size": float(latest.get("top_lsr_size", 0)),
                "long_liquidations_usd": float(latest.get("long_liq_usd", 0)),
                "short_liquidations_usd": float(latest.get("short_liq_usd", 0))
            },
            "history_count": len(formatted),
            "history": formatted
        }
    
    return {"error": "Failed to fetch contract stats"}


async def gateio_open_interest_tool(
    settle: str = "usdt"
) -> Dict[str, Any]:
    """
    Get open interest for top Gate.io contracts
    
    Args:
        settle: Settlement currency
    
    Returns:
        Open interest for top contracts by volume
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    oi_data = await client.get_open_interest_all(settle_enum)
    
    if isinstance(oi_data, list):
        total_oi = sum(o["open_interest_usd"] for o in oi_data)
        
        return {
            "exchange": "gateio",
            "settle": settle,
            "total_open_interest_usd": total_oi,
            "contract_count": len(oi_data),
            "top_contracts": oi_data
        }
    
    return {"error": "Failed to fetch open interest"}


async def gateio_liquidations_tool(
    settle: str = "usdt",
    contract: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get Gate.io liquidation history
    
    Args:
        settle: Settlement currency
        contract: Optional contract filter
        limit: Number of records (max 1000)
    
    Returns:
        Recent liquidations
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    liquidations = await client.get_liquidation_history(settle_enum, contract, min(limit, 1000))
    
    if isinstance(liquidations, list):
        long_liq = 0
        short_liq = 0
        
        formatted = []
        for l in liquidations:
            size = float(l.get("size", 0))
            price = float(l.get("price", 0))
            
            # Determine if long or short liquidation
            if l.get("order_type") == "close_long":
                long_liq += size * price
                side = "long"
            else:
                short_liq += size * price
                side = "short"
            
            formatted.append({
                "contract": l.get("contract"),
                "price": price,
                "size": size,
                "side": side,
                "timestamp": l.get("time")
            })
        
        return {
            "exchange": "gateio",
            "settle": settle,
            "contract_filter": contract,
            "liquidation_count": len(formatted),
            "statistics": {
                "long_liquidations_value": long_liq,
                "short_liquidations_value": short_liq,
                "total_liquidations_value": long_liq + short_liq,
                "long_pct": round(long_liq / (long_liq + short_liq) * 100, 2) if (long_liq + short_liq) > 0 else 0
            },
            "liquidations": formatted[:50]
        }
    
    return {"error": "Failed to fetch liquidations"}


async def gateio_insurance_fund_tool(
    settle: str = "usdt"
) -> Dict[str, Any]:
    """
    Get Gate.io insurance fund balance
    
    Args:
        settle: Settlement currency
    
    Returns:
        Insurance fund balance history
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    insurance = await client.get_insurance_fund(settle_enum)
    
    if isinstance(insurance, list) and insurance:
        latest = insurance[0]
        
        return {
            "exchange": "gateio",
            "settle": settle,
            "current_balance": float(latest.get("b", 0)),
            "timestamp": latest.get("t"),
            "history_count": len(insurance),
            "history": [
                {"balance": float(i.get("b", 0)), "timestamp": i.get("t")}
                for i in insurance[:24]
            ]
        }
    
    return {"error": "Failed to fetch insurance fund"}


async def gateio_risk_limit_tiers_tool(
    contract: str,
    settle: str = "usdt"
) -> Dict[str, Any]:
    """
    Get Gate.io risk limit tiers for a contract
    
    Args:
        contract: Contract name (e.g., 'BTC_USDT')
        settle: Settlement currency
    
    Returns:
        Risk limit tiers with maintenance margin requirements
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    tiers = await client.get_risk_limit_tiers(contract, settle_enum)
    
    if isinstance(tiers, list):
        return {
            "exchange": "gateio",
            "contract": contract,
            "settle": settle,
            "tier_count": len(tiers),
            "tiers": tiers
        }
    
    return {"error": "Failed to fetch risk limit tiers"}


# ==================== DELIVERY FUTURES TOOLS ====================

async def gateio_delivery_contracts_tool(
    settle: str = "usdt"
) -> Dict[str, Any]:
    """
    Get all Gate.io delivery futures contracts
    
    Args:
        settle: Settlement currency
    
    Returns:
        List of delivery futures contracts
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    contracts = await client.get_delivery_contracts(settle_enum)
    
    if isinstance(contracts, list):
        return {
            "exchange": "gateio",
            "product": "delivery",
            "settle": settle,
            "contract_count": len(contracts),
            "contracts": contracts
        }
    
    return {"error": "Failed to fetch delivery contracts"}


async def gateio_delivery_ticker_tool(
    contract: str,
    settle: str = "usdt"
) -> Dict[str, Any]:
    """
    Get Gate.io delivery futures ticker
    
    Args:
        contract: Contract name
        settle: Settlement currency
    
    Returns:
        Delivery futures ticker
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    tickers = await client.get_delivery_tickers(settle_enum, contract)
    
    if isinstance(tickers, list) and tickers:
        t = tickers[0]
        return {
            "exchange": "gateio",
            "product": "delivery",
            "contract": contract,
            "settle": settle,
            "ticker": {
                "last_price": float(t.get("last", 0)),
                "mark_price": float(t.get("mark_price", 0)),
                "index_price": float(t.get("index_price", 0)),
                "volume_24h": float(t.get("volume_24h", 0)),
                "change_pct": float(t.get("change_percentage", 0)),
                "basis": float(t.get("basis_value", 0)),
                "basis_rate": float(t.get("basis_rate", 0))
            }
        }
    
    return {"error": "Failed to fetch delivery ticker"}


# ==================== OPTIONS TOOLS ====================

async def gateio_options_underlyings_tool() -> Dict[str, Any]:
    """
    Get Gate.io options underlying assets
    
    Returns:
        List of available underlying assets for options
    """
    client = get_gateio_rest_client()
    
    underlyings = await client.get_options_underlyings()
    
    if isinstance(underlyings, list):
        return {
            "exchange": "gateio",
            "product": "options",
            "underlying_count": len(underlyings),
            "underlyings": underlyings
        }
    
    return {"error": "Failed to fetch options underlyings"}


async def gateio_options_expirations_tool(
    underlying: str
) -> Dict[str, Any]:
    """
    Get Gate.io options expiration dates
    
    Args:
        underlying: Underlying asset (e.g., 'BTC_USDT')
    
    Returns:
        List of expiration timestamps
    """
    client = get_gateio_rest_client()
    
    expirations = await client.get_options_expirations(underlying)
    
    if isinstance(expirations, list):
        return {
            "exchange": "gateio",
            "product": "options",
            "underlying": underlying,
            "expiration_count": len(expirations),
            "expirations": expirations
        }
    
    return {"error": "Failed to fetch options expirations"}


async def gateio_options_contracts_tool(
    underlying: str,
    expiration: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get Gate.io options contracts
    
    Args:
        underlying: Underlying asset (e.g., 'BTC_USDT')
        expiration: Optional expiration timestamp filter
    
    Returns:
        List of options contracts
    """
    client = get_gateio_rest_client()
    
    contracts = await client.get_options_contracts(underlying, expiration)
    
    if isinstance(contracts, list):
        calls = [c for c in contracts if c.get("is_call")]
        puts = [c for c in contracts if not c.get("is_call")]
        
        return {
            "exchange": "gateio",
            "product": "options",
            "underlying": underlying,
            "expiration_filter": expiration,
            "total_contracts": len(contracts),
            "call_count": len(calls),
            "put_count": len(puts),
            "contracts": contracts[:100]  # Limit to first 100
        }
    
    return {"error": "Failed to fetch options contracts"}


async def gateio_options_tickers_tool(
    underlying: str
) -> Dict[str, Any]:
    """
    Get Gate.io options tickers
    
    Args:
        underlying: Underlying asset (e.g., 'BTC_USDT')
    
    Returns:
        Options tickers with Greeks
    """
    client = get_gateio_rest_client()
    
    tickers = await client.get_options_tickers(underlying)
    
    if isinstance(tickers, list):
        formatted = []
        for t in tickers:
            formatted.append({
                "contract": t.get("name"),
                "last_price": float(t.get("last_price", 0)),
                "mark_price": float(t.get("mark_price", 0)),
                "bid": float(t.get("bid1_price", 0)),
                "ask": float(t.get("ask1_price", 0)),
                "delta": float(t.get("delta", 0)),
                "gamma": float(t.get("gamma", 0)),
                "vega": float(t.get("vega", 0)),
                "theta": float(t.get("theta", 0)),
                "mark_iv": float(t.get("mark_iv", 0)),
                "bid_iv": float(t.get("bid_iv", 0)),
                "ask_iv": float(t.get("ask_iv", 0))
            })
        
        return {
            "exchange": "gateio",
            "product": "options",
            "underlying": underlying,
            "ticker_count": len(formatted),
            "tickers": formatted[:50]
        }
    
    return {"error": "Failed to fetch options tickers"}


async def gateio_options_underlying_ticker_tool(
    underlying: str
) -> Dict[str, Any]:
    """
    Get Gate.io underlying ticker for options
    
    Args:
        underlying: Underlying asset (e.g., 'BTC_USDT')
    
    Returns:
        Underlying asset ticker
    """
    client = get_gateio_rest_client()
    
    ticker = await client.get_options_underlying_ticker(underlying)
    
    if isinstance(ticker, dict) and "error" not in ticker:
        return {
            "exchange": "gateio",
            "product": "options",
            "underlying": underlying,
            "ticker": {
                "index_price": float(ticker.get("index_price", 0)),
                "mark_price": float(ticker.get("mark_price", 0)),
                "mark_iv": float(ticker.get("mark_iv", 0))
            }
        }
    
    return ticker


async def gateio_options_orderbook_tool(
    contract: str,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Get Gate.io options orderbook
    
    Args:
        contract: Options contract name
        limit: Depth limit (max 50)
    
    Returns:
        Options order book
    """
    client = get_gateio_rest_client()
    
    orderbook = await client.get_options_orderbook(contract, limit=min(limit, 50))
    
    if isinstance(orderbook, dict) and "asks" in orderbook:
        asks = orderbook.get("asks", [])
        bids = orderbook.get("bids", [])
        
        return {
            "exchange": "gateio",
            "product": "options",
            "contract": contract,
            "bid_count": len(bids),
            "ask_count": len(asks),
            "bids": [{"price": float(b.get("p", 0)), "size": float(b.get("s", 0))} for b in bids],
            "asks": [{"price": float(a.get("p", 0)), "size": float(a.get("s", 0))} for a in asks]
        }
    
    return {"error": "Failed to fetch options orderbook"}


# ==================== COMPOSITE/ANALYSIS TOOLS ====================

async def gateio_market_snapshot_tool(
    symbol: str = "BTC",
    settle: str = "usdt"
) -> Dict[str, Any]:
    """
    Get comprehensive Gate.io market snapshot
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC')
        settle: Settlement currency
    
    Returns:
        Comprehensive market data including ticker, funding, OI, orderbook
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    return await client.get_market_snapshot(symbol, settle_enum)


async def gateio_top_movers_tool(
    settle: str = "usdt",
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get top gainers and losers on Gate.io
    
    Args:
        settle: Settlement currency
        limit: Number of movers to return
    
    Returns:
        Top gainers and losers by price change
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    return await client.get_top_movers(settle_enum, limit)


async def gateio_full_analysis_tool(
    symbol: str = "BTC",
    settle: str = "usdt"
) -> Dict[str, Any]:
    """
    Get full Gate.io analysis with trading signals
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC')
        settle: Settlement currency
    
    Returns:
        Comprehensive analysis with signals for funding, basis, L/S ratio, etc.
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    return await client.get_full_analysis(symbol, settle_enum)


async def gateio_perpetuals_tool(
    settle: str = "usdt"
) -> Dict[str, Any]:
    """
    Get all Gate.io perpetual futures
    
    Args:
        settle: Settlement currency
    
    Returns:
        All perpetual contracts with ticker data
    """
    client = get_gateio_rest_client()
    settle_enum = GateioSettle(settle.lower())
    
    perpetuals = await client.get_all_perpetuals(settle_enum)
    
    if isinstance(perpetuals, list):
        return {
            "exchange": "gateio",
            "settle": settle,
            "perpetual_count": len(perpetuals),
            "perpetuals": perpetuals[:50]  # Top 50 by volume
        }
    
    return {"error": "Failed to fetch perpetuals"}
