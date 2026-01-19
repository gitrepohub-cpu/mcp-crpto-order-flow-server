"""
Kraken MCP Tools
Tool wrappers for Kraken REST API with analysis
"""

from typing import Dict, Any, List, Optional
from src.storage.kraken_rest_client import (
    get_kraken_rest_client,
    KrakenInterval,
    KrakenFuturesInterval
)
import logging

logger = logging.getLogger(__name__)


# ==================== SPOT TICKER TOOLS ====================

async def kraken_spot_ticker(
    pair: str = "XBTUSD"
) -> Dict[str, Any]:
    """
    Get Kraken spot ticker
    
    Args:
        pair: Trading pair (e.g., XBTUSD, ETHUSD, XXBTZUSD)
    
    Returns:
        Spot ticker data with analysis
    """
    client = get_kraken_rest_client()
    
    try:
        result = await client.get_spot_ticker_formatted(pair)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        # Add analysis
        if isinstance(result, dict):
            price = result.get("last_price", 0)
            high = result.get("high_24h", 0)
            low = result.get("low_24h", 0)
            open_price = result.get("open_today", 0)
            
            change_pct = ((price - open_price) / open_price * 100) if open_price > 0 else 0
            range_pct = ((high - low) / low * 100) if low > 0 else 0
            position_in_range = ((price - low) / (high - low) * 100) if (high - low) > 0 else 50
            
            result["analysis"] = {
                "change_today_pct": round(change_pct, 2),
                "range_24h_pct": round(range_pct, 2),
                "position_in_range_pct": round(position_in_range, 1),
                "trend": "🟢 Bullish" if change_pct > 1 else "🔴 Bearish" if change_pct < -1 else "⚪ Neutral"
            }
        
        return {
            "exchange": "kraken",
            "market_type": "spot",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error getting Kraken spot ticker: {e}")
        return {"error": str(e)}


async def kraken_all_spot_tickers(
    pairs: List[str] = None
) -> Dict[str, Any]:
    """
    Get multiple Kraken spot tickers
    
    Args:
        pairs: List of trading pairs (default: major pairs)
    
    Returns:
        Multiple ticker data
    """
    client = get_kraken_rest_client()
    
    try:
        if not pairs:
            pairs = ["XBTUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD"]
        
        result = await client.get_ticker(pairs)
        
        if isinstance(result, dict) and "error" not in result:
            formatted = []
            for pair_name, data in result.items():
                if isinstance(data, dict):
                    price = float(data.get("c", [0])[0])
                    vol_24h = float(data.get("v", [0, 0])[1])
                    high_24h = float(data.get("h", [0, 0])[1])
                    low_24h = float(data.get("l", [0, 0])[1])
                    open_price = float(data.get("o", 0))
                    
                    change_pct = ((price - open_price) / open_price * 100) if open_price > 0 else 0
                    
                    formatted.append({
                        "pair": pair_name,
                        "last_price": price,
                        "volume_24h": vol_24h,
                        "high_24h": high_24h,
                        "low_24h": low_24h,
                        "open_today": open_price,
                        "change_pct": round(change_pct, 2)
                    })
            
            # Sort by volume
            formatted.sort(key=lambda x: x["volume_24h"], reverse=True)
            
            return {
                "exchange": "kraken",
                "market_type": "spot",
                "count": len(formatted),
                "tickers": formatted
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting Kraken spot tickers: {e}")
        return {"error": str(e)}


# ==================== FUTURES TICKER TOOLS ====================

async def kraken_futures_ticker(
    symbol: str = "PF_XBTUSD"
) -> Dict[str, Any]:
    """
    Get Kraken futures ticker
    
    Args:
        symbol: Futures symbol (e.g., PF_XBTUSD for perpetual BTC)
    
    Returns:
        Futures ticker data with analysis
    """
    client = get_kraken_rest_client()
    
    try:
        result = await client.get_futures_ticker_formatted(symbol)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        # Add analysis
        if isinstance(result, dict):
            funding = result.get("funding_rate", 0)
            mark = result.get("mark_price", 0)
            index = result.get("index_price", 0)
            
            premium = ((mark - index) / index * 100) if index > 0 else 0
            annualized_funding = funding * 3 * 365 * 100  # 8h funding, annualized
            
            result["analysis"] = {
                "premium_pct": round(premium, 4),
                "funding_rate_pct": round(funding * 100, 4),
                "annualized_funding_pct": round(annualized_funding, 2),
                "funding_signal": "🔴 Shorts paid" if funding > 0.0001 else "🟢 Longs paid" if funding < -0.0001 else "⚪ Neutral"
            }
        
        return {
            "exchange": "kraken",
            "market_type": "perpetual",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error getting Kraken futures ticker: {e}")
        return {"error": str(e)}


async def kraken_all_futures_tickers() -> Dict[str, Any]:
    """
    Get all Kraken futures tickers
    
    Returns:
        All futures tickers with analysis
    """
    client = get_kraken_rest_client()
    
    try:
        perpetuals = await client.get_all_perpetuals()
        
        if not perpetuals:
            return {"error": "No perpetual data available"}
        
        # Calculate aggregates
        total_oi = sum(p.get("open_interest", 0) * p.get("mark_price", 0) for p in perpetuals)
        total_vol = sum(p.get("volume_24h", 0) for p in perpetuals)
        avg_funding = sum(p.get("funding_rate", 0) for p in perpetuals) / len(perpetuals) if perpetuals else 0
        
        # Get extremes
        highest_funding = max(perpetuals, key=lambda x: x.get("funding_rate", 0))
        lowest_funding = min(perpetuals, key=lambda x: x.get("funding_rate", 0))
        
        return {
            "exchange": "kraken",
            "market_type": "perpetual",
            "count": len(perpetuals),
            "aggregates": {
                "total_open_interest_usd": round(total_oi, 2),
                "total_volume_24h": round(total_vol, 2),
                "avg_funding_rate_pct": round(avg_funding * 100, 4)
            },
            "extremes": {
                "highest_funding": {
                    "symbol": highest_funding["symbol"],
                    "rate_pct": round(highest_funding["funding_rate_pct"], 4)
                },
                "lowest_funding": {
                    "symbol": lowest_funding["symbol"],
                    "rate_pct": round(lowest_funding["funding_rate_pct"], 4)
                }
            },
            "tickers": perpetuals[:20]  # Top 20 by volume
        }
        
    except Exception as e:
        logger.error(f"Error getting Kraken futures tickers: {e}")
        return {"error": str(e)}


# ==================== ORDERBOOK TOOLS ====================

async def kraken_spot_orderbook(
    pair: str = "XBTUSD",
    depth: int = 100
) -> Dict[str, Any]:
    """
    Get Kraken spot order book
    
    Args:
        pair: Trading pair
        depth: Number of levels (max 500)
    
    Returns:
        Order book with analysis
    """
    client = get_kraken_rest_client()
    
    try:
        result = await client.get_orderbook(pair, depth)
        
        if isinstance(result, dict) and "error" not in result:
            # Find the orderbook in result (key is pair name)
            for pair_name, data in result.items():
                if isinstance(data, dict) and "bids" in data and "asks" in data:
                    bids = data["bids"]
                    asks = data["asks"]
                    
                    # Calculate metrics
                    bid_volume = sum(float(b[1]) for b in bids[:20])
                    ask_volume = sum(float(a[1]) for a in asks[:20])
                    
                    best_bid = float(bids[0][0]) if bids else 0
                    best_ask = float(asks[0][0]) if asks else 0
                    spread = best_ask - best_bid
                    spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
                    
                    imbalance = ((bid_volume - ask_volume) / (bid_volume + ask_volume) * 100) if (bid_volume + ask_volume) > 0 else 0
                    
                    return {
                        "exchange": "kraken",
                        "market_type": "spot",
                        "pair": pair_name,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "spread": round(spread, 2),
                        "spread_pct": round(spread_pct, 4),
                        "bid_levels": len(bids),
                        "ask_levels": len(asks),
                        "analysis": {
                            "bid_volume_top20": round(bid_volume, 4),
                            "ask_volume_top20": round(ask_volume, 4),
                            "imbalance_pct": round(imbalance, 2),
                            "pressure": "🟢 Buy pressure" if imbalance > 10 else "🔴 Sell pressure" if imbalance < -10 else "⚪ Balanced"
                        },
                        "bids": [[float(b[0]), float(b[1])] for b in bids[:10]],
                        "asks": [[float(a[0]), float(a[1])] for a in asks[:10]]
                    }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting Kraken spot orderbook: {e}")
        return {"error": str(e)}


async def kraken_futures_orderbook(
    symbol: str = "PF_XBTUSD"
) -> Dict[str, Any]:
    """
    Get Kraken futures order book
    
    Args:
        symbol: Futures symbol
    
    Returns:
        Futures order book with analysis
    """
    client = get_kraken_rest_client()
    
    try:
        result = await client.get_futures_orderbook(symbol)
        
        if isinstance(result, dict) and "error" not in result:
            bids = result.get("bids", [])
            asks = result.get("asks", [])
            
            # Calculate metrics
            bid_volume = sum(float(b[1]) for b in bids[:20]) if bids else 0
            ask_volume = sum(float(a[1]) for a in asks[:20]) if asks else 0
            
            best_bid = float(bids[0][0]) if bids else 0
            best_ask = float(asks[0][0]) if asks else 0
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
            
            imbalance = ((bid_volume - ask_volume) / (bid_volume + ask_volume) * 100) if (bid_volume + ask_volume) > 0 else 0
            
            return {
                "exchange": "kraken",
                "market_type": "perpetual",
                "symbol": symbol,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": round(spread, 2),
                "spread_pct": round(spread_pct, 4),
                "bid_levels": len(bids),
                "ask_levels": len(asks),
                "analysis": {
                    "bid_volume_top20": round(bid_volume, 4),
                    "ask_volume_top20": round(ask_volume, 4),
                    "imbalance_pct": round(imbalance, 2),
                    "pressure": "🟢 Buy pressure" if imbalance > 10 else "🔴 Sell pressure" if imbalance < -10 else "⚪ Balanced"
                },
                "bids": [[float(b[0]), float(b[1])] for b in bids[:10]] if bids else [],
                "asks": [[float(a[0]), float(a[1])] for a in asks[:10]] if asks else []
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting Kraken futures orderbook: {e}")
        return {"error": str(e)}


# ==================== TRADES TOOLS ====================

async def kraken_spot_trades(
    pair: str = "XBTUSD",
    count: int = 100
) -> Dict[str, Any]:
    """
    Get Kraken spot recent trades
    
    Args:
        pair: Trading pair
        count: Number of trades
    
    Returns:
        Recent trades with analysis
    """
    client = get_kraken_rest_client()
    
    try:
        result = await client.get_trades(pair, count=count)
        
        if isinstance(result, dict) and "error" not in result:
            # Find trades in result
            for pair_name, data in result.items():
                if pair_name == "last":
                    continue
                if isinstance(data, list):
                    trades = []
                    buy_volume = 0
                    sell_volume = 0
                    
                    for t in data[-100:]:  # Last 100
                        if len(t) >= 4:
                            price = float(t[0])
                            volume = float(t[1])
                            side = "buy" if t[3] == "b" else "sell"
                            
                            if side == "buy":
                                buy_volume += volume
                            else:
                                sell_volume += volume
                            
                            trades.append({
                                "price": price,
                                "volume": volume,
                                "timestamp": float(t[2]),
                                "side": side,
                                "order_type": "market" if t[4] == "m" else "limit"
                            })
                    
                    total_volume = buy_volume + sell_volume
                    
                    return {
                        "exchange": "kraken",
                        "market_type": "spot",
                        "pair": pair_name,
                        "trade_count": len(trades),
                        "analysis": {
                            "buy_volume": round(buy_volume, 4),
                            "sell_volume": round(sell_volume, 4),
                            "total_volume": round(total_volume, 4),
                            "buy_ratio_pct": round(buy_volume / total_volume * 100, 1) if total_volume > 0 else 50,
                            "flow": "🟢 Net buying" if buy_volume > sell_volume * 1.1 else "🔴 Net selling" if sell_volume > buy_volume * 1.1 else "⚪ Balanced"
                        },
                        "recent_trades": trades[-10:]
                    }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting Kraken spot trades: {e}")
        return {"error": str(e)}


async def kraken_futures_trades(
    symbol: str = "PF_XBTUSD"
) -> Dict[str, Any]:
    """
    Get Kraken futures recent trades
    
    Args:
        symbol: Futures symbol
    
    Returns:
        Recent trades with analysis
    """
    client = get_kraken_rest_client()
    
    try:
        result = await client.get_futures_trades(symbol)
        
        if isinstance(result, list):
            trades = []
            buy_volume = 0
            sell_volume = 0
            
            for t in result[-100:]:
                if isinstance(t, dict):
                    price = float(t.get("price", 0))
                    size = float(t.get("size", 0))
                    side = t.get("side", "").lower()
                    
                    if side == "buy":
                        buy_volume += size
                    else:
                        sell_volume += size
                    
                    trades.append({
                        "price": price,
                        "size": size,
                        "side": side,
                        "timestamp": t.get("time"),
                        "uid": t.get("uid")
                    })
            
            total_volume = buy_volume + sell_volume
            
            return {
                "exchange": "kraken",
                "market_type": "perpetual",
                "symbol": symbol,
                "trade_count": len(trades),
                "analysis": {
                    "buy_volume": round(buy_volume, 4),
                    "sell_volume": round(sell_volume, 4),
                    "total_volume": round(total_volume, 4),
                    "buy_ratio_pct": round(buy_volume / total_volume * 100, 1) if total_volume > 0 else 50,
                    "flow": "🟢 Net buying" if buy_volume > sell_volume * 1.1 else "🔴 Net selling" if sell_volume > buy_volume * 1.1 else "⚪ Balanced"
                },
                "recent_trades": trades[-10:]
            }
        
        return {"error": "Invalid response format", "data": result}
        
    except Exception as e:
        logger.error(f"Error getting Kraken futures trades: {e}")
        return {"error": str(e)}


# ==================== KLINES/OHLC TOOLS ====================

async def kraken_spot_klines(
    pair: str = "XBTUSD",
    interval: str = "1H",
    since: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get Kraken spot OHLC/candlestick data
    
    Args:
        pair: Trading pair
        interval: 1m, 5m, 15m, 30m, 1H, 4H, 1D, 1W
        since: Unix timestamp to start from
    
    Returns:
        Candlestick data with analysis
    """
    client = get_kraken_rest_client()
    
    try:
        interval_map = {
            "1m": KrakenInterval.MIN_1,
            "5m": KrakenInterval.MIN_5,
            "15m": KrakenInterval.MIN_15,
            "30m": KrakenInterval.MIN_30,
            "1H": KrakenInterval.HOUR_1,
            "4H": KrakenInterval.HOUR_4,
            "1D": KrakenInterval.DAY_1,
            "1W": KrakenInterval.WEEK_1
        }
        
        kraken_interval = interval_map.get(interval, KrakenInterval.HOUR_1)
        result = await client.get_ohlc(pair, kraken_interval, since)
        
        if isinstance(result, dict) and "error" not in result:
            # Find OHLC data in result
            for pair_name, data in result.items():
                if pair_name == "last":
                    continue
                if isinstance(data, list):
                    candles = []
                    for c in data[-100:]:
                        if len(c) >= 7:
                            candles.append({
                                "timestamp": int(c[0]),
                                "open": float(c[1]),
                                "high": float(c[2]),
                                "low": float(c[3]),
                                "close": float(c[4]),
                                "vwap": float(c[5]),
                                "volume": float(c[6]),
                                "count": int(c[7]) if len(c) > 7 else 0
                            })
                    
                    if candles:
                        # Calculate technical analysis
                        closes = [c["close"] for c in candles]
                        latest = candles[-1]
                        
                        # Simple moving averages
                        sma_20 = sum(closes[-20:]) / min(20, len(closes)) if closes else 0
                        sma_50 = sum(closes[-50:]) / min(50, len(closes)) if len(closes) >= 20 else 0
                        
                        # Trend
                        price_vs_sma20 = ((latest["close"] - sma_20) / sma_20 * 100) if sma_20 > 0 else 0
                        
                        return {
                            "exchange": "kraken",
                            "market_type": "spot",
                            "pair": pair_name,
                            "interval": interval,
                            "candle_count": len(candles),
                            "latest": latest,
                            "analysis": {
                                "sma_20": round(sma_20, 2),
                                "sma_50": round(sma_50, 2),
                                "price_vs_sma20_pct": round(price_vs_sma20, 2),
                                "trend": "🟢 Above SMA20" if price_vs_sma20 > 1 else "🔴 Below SMA20" if price_vs_sma20 < -1 else "⚪ At SMA20"
                            },
                            "candles": candles[-20:]  # Last 20 candles
                        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting Kraken spot klines: {e}")
        return {"error": str(e)}


async def kraken_futures_klines(
    symbol: str = "PF_XBTUSD",
    interval: str = "1h"
) -> Dict[str, Any]:
    """
    Get Kraken futures candlestick data
    
    Args:
        symbol: Futures symbol
        interval: 1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d, 1w
    
    Returns:
        Candlestick data with analysis
    """
    client = get_kraken_rest_client()
    
    try:
        interval_map = {
            "1m": KrakenFuturesInterval.MIN_1,
            "5m": KrakenFuturesInterval.MIN_5,
            "15m": KrakenFuturesInterval.MIN_15,
            "30m": KrakenFuturesInterval.MIN_30,
            "1h": KrakenFuturesInterval.HOUR_1,
            "4h": KrakenFuturesInterval.HOUR_4,
            "12h": KrakenFuturesInterval.HOUR_12,
            "1d": KrakenFuturesInterval.DAY_1,
            "1w": KrakenFuturesInterval.WEEK_1
        }
        
        kraken_interval = interval_map.get(interval.lower(), KrakenFuturesInterval.HOUR_1)
        result = await client.get_futures_candles(symbol, kraken_interval)
        
        if isinstance(result, list) and result:
            candles = []
            for c in result[-100:]:
                if isinstance(c, dict):
                    candles.append({
                        "timestamp": c.get("time"),
                        "open": float(c.get("open", 0)),
                        "high": float(c.get("high", 0)),
                        "low": float(c.get("low", 0)),
                        "close": float(c.get("close", 0)),
                        "volume": float(c.get("volume", 0))
                    })
            
            if candles:
                closes = [c["close"] for c in candles]
                latest = candles[-1]
                
                sma_20 = sum(closes[-20:]) / min(20, len(closes)) if closes else 0
                price_vs_sma20 = ((latest["close"] - sma_20) / sma_20 * 100) if sma_20 > 0 else 0
                
                return {
                    "exchange": "kraken",
                    "market_type": "perpetual",
                    "symbol": symbol,
                    "interval": interval,
                    "candle_count": len(candles),
                    "latest": latest,
                    "analysis": {
                        "sma_20": round(sma_20, 2),
                        "price_vs_sma20_pct": round(price_vs_sma20, 2),
                        "trend": "🟢 Above SMA20" if price_vs_sma20 > 1 else "🔴 Below SMA20" if price_vs_sma20 < -1 else "⚪ At SMA20"
                    },
                    "candles": candles[-20:]
                }
        
        return {"error": "No candle data available", "raw": result}
        
    except Exception as e:
        logger.error(f"Error getting Kraken futures klines: {e}")
        return {"error": str(e)}


# ==================== FUNDING RATE TOOLS ====================

async def kraken_funding_rates() -> Dict[str, Any]:
    """
    Get all Kraken perpetual funding rates
    
    Returns:
        Funding rates for all perpetuals
    """
    client = get_kraken_rest_client()
    
    try:
        result = await client.get_funding_rates()
        
        if isinstance(result, list) and result:
            # Calculate statistics
            positive = [r for r in result if r["funding_rate"] > 0]
            negative = [r for r in result if r["funding_rate"] < 0]
            
            avg_rate = sum(r["funding_rate"] for r in result) / len(result) if result else 0
            
            return {
                "exchange": "kraken",
                "count": len(result),
                "summary": {
                    "positive_count": len(positive),
                    "negative_count": len(negative),
                    "avg_funding_rate_pct": round(avg_rate * 100, 4),
                    "market_sentiment": "🔴 Bearish (more paying longs)" if len(positive) > len(negative) * 1.5 else "🟢 Bullish (more paying shorts)" if len(negative) > len(positive) * 1.5 else "⚪ Mixed"
                },
                "highest_rates": result[:10],
                "lowest_rates": result[-10:][::-1]
            }
        
        return {"error": "No funding rate data", "data": result}
        
    except Exception as e:
        logger.error(f"Error getting Kraken funding rates: {e}")
        return {"error": str(e)}


# ==================== OPEN INTEREST TOOLS ====================

async def kraken_open_interest() -> Dict[str, Any]:
    """
    Get all Kraken perpetual open interest
    
    Returns:
        Open interest for all perpetuals
    """
    client = get_kraken_rest_client()
    
    try:
        result = await client.get_open_interest_all()
        
        if isinstance(result, list) and result:
            total_oi_usd = sum(r["open_interest_usd"] for r in result)
            total_volume = sum(r["volume_24h"] for r in result)
            
            return {
                "exchange": "kraken",
                "count": len(result),
                "totals": {
                    "total_open_interest_usd": round(total_oi_usd, 2),
                    "total_volume_24h": round(total_volume, 2)
                },
                "by_symbol": result[:15]
            }
        
        return {"error": "No open interest data", "data": result}
        
    except Exception as e:
        logger.error(f"Error getting Kraken open interest: {e}")
        return {"error": str(e)}


# ==================== INSTRUMENTS TOOLS ====================

async def kraken_spot_pairs(
    info: str = "info"
) -> Dict[str, Any]:
    """
    Get Kraken spot trading pairs
    
    Args:
        info: info, leverage, fees, margin
    
    Returns:
        Trading pairs information
    """
    client = get_kraken_rest_client()
    
    try:
        result = await client.get_asset_pairs(info=info)
        
        if isinstance(result, dict) and "error" not in result:
            pairs = []
            for name, data in result.items():
                if isinstance(data, dict):
                    pairs.append({
                        "pair": name,
                        "altname": data.get("altname"),
                        "base": data.get("base"),
                        "quote": data.get("quote"),
                        "status": data.get("status"),
                        "lot_decimals": data.get("lot_decimals"),
                        "pair_decimals": data.get("pair_decimals"),
                        "ordermin": data.get("ordermin"),
                        "costmin": data.get("costmin")
                    })
            
            # Filter active pairs
            active_pairs = [p for p in pairs if p.get("status") == "online"]
            
            return {
                "exchange": "kraken",
                "market_type": "spot",
                "total_pairs": len(pairs),
                "active_pairs": len(active_pairs),
                "pairs": active_pairs[:50]  # First 50
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting Kraken spot pairs: {e}")
        return {"error": str(e)}


async def kraken_futures_instruments() -> Dict[str, Any]:
    """
    Get Kraken futures instruments
    
    Returns:
        All futures instruments information
    """
    client = get_kraken_rest_client()
    
    try:
        result = await client.get_futures_instruments()
        
        if isinstance(result, list) and result:
            perpetuals = [i for i in result if i.get("type") == "flexible_futures"]
            futures = [i for i in result if i.get("type") == "futures"]
            
            formatted_perps = []
            for p in perpetuals:
                formatted_perps.append({
                    "symbol": p.get("symbol"),
                    "underlying": p.get("underlying"),
                    "tick_size": p.get("tickSize"),
                    "contract_size": p.get("contractSize"),
                    "tradeable": p.get("tradeable"),
                    "margin_levels": p.get("marginLevels", [])
                })
            
            return {
                "exchange": "kraken",
                "perpetual_count": len(perpetuals),
                "futures_count": len(futures),
                "total_count": len(result),
                "perpetuals": formatted_perps[:20],
                "instruments_raw": result[:10]
            }
        
        return {"error": "No instruments data", "data": result}
        
    except Exception as e:
        logger.error(f"Error getting Kraken futures instruments: {e}")
        return {"error": str(e)}


# ==================== SYSTEM STATUS TOOLS ====================

async def kraken_system_status() -> Dict[str, Any]:
    """
    Get Kraken system status
    
    Returns:
        System status and server time
    """
    client = get_kraken_rest_client()
    
    try:
        status_task = client.get_system_status()
        time_task = client.get_server_time()
        
        import asyncio
        status, server_time = await asyncio.gather(
            status_task, time_task,
            return_exceptions=True
        )
        
        result = {
            "exchange": "kraken"
        }
        
        if isinstance(status, dict) and "error" not in status:
            result["status"] = status.get("status")
            result["timestamp"] = status.get("timestamp")
        
        if isinstance(server_time, dict) and "error" not in server_time:
            result["server_time_unix"] = server_time.get("unixtime")
            result["server_time_rfc"] = server_time.get("rfc1123")
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting Kraken system status: {e}")
        return {"error": str(e)}


# ==================== SPREAD TOOLS ====================

async def kraken_spread(
    pair: str = "XBTUSD"
) -> Dict[str, Any]:
    """
    Get Kraken recent spread data
    
    Args:
        pair: Trading pair
    
    Returns:
        Recent spread history
    """
    client = get_kraken_rest_client()
    
    try:
        result = await client.get_spread(pair)
        
        if isinstance(result, dict) and "error" not in result:
            for pair_name, data in result.items():
                if pair_name == "last":
                    continue
                if isinstance(data, list):
                    spreads = []
                    for s in data[-50:]:
                        if len(s) >= 3:
                            bid = float(s[1])
                            ask = float(s[2])
                            spread = ask - bid
                            spread_pct = (spread / bid * 100) if bid > 0 else 0
                            
                            spreads.append({
                                "timestamp": int(s[0]),
                                "bid": bid,
                                "ask": ask,
                                "spread": round(spread, 4),
                                "spread_pct": round(spread_pct, 4)
                            })
                    
                    if spreads:
                        avg_spread_pct = sum(s["spread_pct"] for s in spreads) / len(spreads)
                        
                        return {
                            "exchange": "kraken",
                            "market_type": "spot",
                            "pair": pair_name,
                            "spread_count": len(spreads),
                            "current": spreads[-1],
                            "analysis": {
                                "avg_spread_pct": round(avg_spread_pct, 4),
                                "min_spread_pct": round(min(s["spread_pct"] for s in spreads), 4),
                                "max_spread_pct": round(max(s["spread_pct"] for s in spreads), 4),
                                "liquidity": "🟢 Tight" if avg_spread_pct < 0.05 else "🟡 Normal" if avg_spread_pct < 0.2 else "🔴 Wide"
                            },
                            "recent_spreads": spreads[-10:]
                        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting Kraken spread: {e}")
        return {"error": str(e)}


# ==================== ASSETS TOOLS ====================

async def kraken_assets() -> Dict[str, Any]:
    """
    Get Kraken asset information
    
    Returns:
        All supported assets
    """
    client = get_kraken_rest_client()
    
    try:
        result = await client.get_assets()
        
        if isinstance(result, dict) and "error" not in result:
            assets = []
            for name, data in result.items():
                if isinstance(data, dict):
                    assets.append({
                        "asset": name,
                        "altname": data.get("altname"),
                        "aclass": data.get("aclass"),
                        "decimals": data.get("decimals"),
                        "display_decimals": data.get("display_decimals"),
                        "status": data.get("status")
                    })
            
            return {
                "exchange": "kraken",
                "asset_count": len(assets),
                "assets": assets[:50]
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting Kraken assets: {e}")
        return {"error": str(e)}


# ==================== TOP MOVERS TOOLS ====================

async def kraken_top_movers(
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get Kraken top gainers and losers
    
    Args:
        limit: Number of results per category
    
    Returns:
        Top movers from futures
    """
    client = get_kraken_rest_client()
    
    try:
        result = await client.get_top_movers(limit)
        
        if isinstance(result, dict) and "error" not in result:
            return {
                "exchange": "kraken",
                "market_type": "perpetual",
                **result
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting Kraken top movers: {e}")
        return {"error": str(e)}


# ==================== COMPREHENSIVE ANALYSIS TOOLS ====================

async def kraken_market_snapshot(
    symbol: str = "BTC"
) -> Dict[str, Any]:
    """
    Get comprehensive Kraken market snapshot
    
    Args:
        symbol: Base symbol (BTC, ETH, etc.)
    
    Returns:
        Combined spot and futures data
    """
    client = get_kraken_rest_client()
    
    try:
        result = await client.get_market_snapshot(symbol)
        
        if isinstance(result, dict):
            return {
                "exchange": "kraken",
                **result
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting Kraken market snapshot: {e}")
        return {"error": str(e)}


async def kraken_full_analysis(
    symbol: str = "BTC"
) -> Dict[str, Any]:
    """
    Get full Kraken analysis with trading signals
    
    Args:
        symbol: Base symbol (BTC, ETH, etc.)
    
    Returns:
        Comprehensive analysis with signals
    """
    client = get_kraken_rest_client()
    
    try:
        result = await client.get_full_analysis(symbol)
        
        if isinstance(result, dict):
            return {
                "exchange": "kraken",
                **result
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting Kraken full analysis: {e}")
        return {"error": str(e)}
