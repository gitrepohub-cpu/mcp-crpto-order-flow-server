"""
Binance Spot MCP Tools
Provides comprehensive spot market data analysis for Binance
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from ..storage.binance_spot_rest_client import (
    get_binance_spot_client,
    BinanceSpotInterval
)

logger = logging.getLogger(__name__)


# ==================== MARKET DATA TOOLS ====================

async def binance_spot_ticker(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Get Binance spot 24hr ticker data
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT, ETHUSDT)
    
    Returns:
        24hr ticker with price, volume, and change stats
    """
    client = get_binance_spot_client()
    
    try:
        result = await client.get_ticker_24hr(symbol)
        
        if "error" in result:
            return {"error": result["error"]}
        
        price = float(result.get("lastPrice", 0))
        change = float(result.get("priceChangePercent", 0))
        volume = float(result.get("volume", 0))
        quote_volume = float(result.get("quoteVolume", 0))
        high = float(result.get("highPrice", 0))
        low = float(result.get("lowPrice", 0))
        trades = int(result.get("count", 0))
        
        return {
            "exchange": "binance",
            "market": "spot",
            "symbol": symbol,
            "price": price,
            "price_formatted": f"${price:,.2f}",
            "change_24h_pct": round(change, 2),
            "change_direction": "📈" if change > 0 else "📉" if change < 0 else "➡️",
            "price_change": float(result.get("priceChange", 0)),
            "volume_24h": volume,
            "volume_formatted": f"{volume:,.2f}",
            "quote_volume_24h": quote_volume,
            "quote_volume_formatted": f"${quote_volume:,.0f}",
            "high_24h": high,
            "low_24h": low,
            "range_24h": f"${low:,.2f} - ${high:,.2f}",
            "range_pct": round((high - low) / low * 100, 2) if low > 0 else 0,
            "weighted_avg_price": float(result.get("weightedAvgPrice", 0)),
            "trade_count_24h": trades,
            "open_price": float(result.get("openPrice", 0)),
            "bid_price": float(result.get("bidPrice", 0)),
            "ask_price": float(result.get("askPrice", 0)),
            "spread": float(result.get("askPrice", 0)) - float(result.get("bidPrice", 0)),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Binance spot ticker: {e}")
        return {"error": str(e)}


async def binance_spot_price(symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Get current Binance spot price(s)
    
    Args:
        symbol: Trading pair or None for major pairs
    
    Returns:
        Current price data
    """
    client = get_binance_spot_client()
    
    try:
        if symbol:
            result = await client.get_ticker_price(symbol)
            
            if "error" in result:
                return {"error": result["error"]}
            
            price = float(result.get("price", 0))
            return {
                "exchange": "binance",
                "market": "spot",
                "symbol": result.get("symbol"),
                "price": price,
                "price_formatted": f"${price:,.2f}",
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Get supported system symbols only
            major_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ARUSDT"]
            result = await client.get_ticker_price(symbols=major_symbols)
            
            if "error" in result:
                return {"error": result["error"]}
            
            prices = []
            for p in result if isinstance(result, list) else [result]:
                price = float(p.get("price", 0))
                prices.append({
                    "symbol": p.get("symbol"),
                    "price": price,
                    "price_formatted": f"${price:,.2f}"
                })
            
            return {
                "exchange": "binance",
                "market": "spot",
                "prices": prices,
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Error getting Binance spot price: {e}")
        return {"error": str(e)}


async def binance_spot_orderbook(symbol: str = "BTCUSDT", depth: int = 100) -> Dict[str, Any]:
    """
    Get Binance spot orderbook
    
    Args:
        symbol: Trading pair
        depth: Number of levels (5, 10, 20, 50, 100, 500, 1000, 5000)
    
    Returns:
        Orderbook with bids, asks, and analysis
    """
    client = get_binance_spot_client()
    
    try:
        result = await client.get_orderbook(symbol, limit=depth)
        
        if "error" in result:
            return {"error": result["error"]}
        
        bids = result.get("bids", [])
        asks = result.get("asks", [])
        
        if not bids or not asks:
            return {"error": "No orderbook data"}
        
        # Calculate metrics
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
        
        bid_volume = sum(float(b[1]) for b in bids[:50])
        ask_volume = sum(float(a[1]) for a in asks[:50])
        total = bid_volume + ask_volume
        imbalance = ((bid_volume - ask_volume) / total * 100) if total > 0 else 0
        
        # Bid/ask walls
        bid_walls = sorted(bids, key=lambda x: float(x[1]), reverse=True)[:3]
        ask_walls = sorted(asks, key=lambda x: float(x[1]), reverse=True)[:3]
        
        return {
            "exchange": "binance",
            "market": "spot",
            "symbol": symbol,
            "last_update_id": result.get("lastUpdateId"),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": (best_bid + best_ask) / 2,
            "spread": spread,
            "spread_pct": round(spread_pct, 4),
            "bid_volume_50": bid_volume,
            "ask_volume_50": ask_volume,
            "imbalance_pct": round(imbalance, 2),
            "imbalance_signal": "🟢 BUY pressure" if imbalance > 15 else "🔴 SELL pressure" if imbalance < -15 else "⚪ Balanced",
            "bid_walls": [{"price": float(w[0]), "qty": float(w[1])} for w in bid_walls],
            "ask_walls": [{"price": float(w[0]), "qty": float(w[1])} for w in ask_walls],
            "depth_levels": len(bids),
            "bids_sample": [[float(b[0]), float(b[1])] for b in bids[:5]],
            "asks_sample": [[float(a[0]), float(a[1])] for a in asks[:5]],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Binance spot orderbook: {e}")
        return {"error": str(e)}


async def binance_spot_trades(symbol: str = "BTCUSDT", limit: int = 500) -> Dict[str, Any]:
    """
    Get recent Binance spot trades
    
    Args:
        symbol: Trading pair
        limit: Number of trades (max 1000)
    
    Returns:
        Recent trades with flow analysis
    """
    client = get_binance_spot_client()
    
    try:
        result = await client.get_recent_trades(symbol, limit=limit)
        
        if "error" in result:
            return {"error": result["error"]}
        
        if not isinstance(result, list) or not result:
            return {"error": "No trades data"}
        
        # Analyze trades
        # isBuyerMaker=True means the buyer made the order that was sitting (seller was aggressive)
        # isBuyerMaker=False means the seller made the order that was sitting (buyer was aggressive)
        
        buy_count = sum(1 for t in result if not t.get("isBuyerMaker", True))
        sell_count = len(result) - buy_count
        
        buy_volume = sum(float(t.get("qty", 0)) for t in result if not t.get("isBuyerMaker", True))
        sell_volume = sum(float(t.get("qty", 0)) for t in result if t.get("isBuyerMaker", True))
        total_volume = buy_volume + sell_volume
        
        buy_value = sum(float(t.get("quoteQty", 0)) for t in result if not t.get("isBuyerMaker", True))
        sell_value = sum(float(t.get("quoteQty", 0)) for t in result if t.get("isBuyerMaker", True))
        
        # Large trades
        avg_size = total_volume / len(result) if result else 0
        large_buys = [t for t in result if not t.get("isBuyerMaker", True) and float(t.get("qty", 0)) > avg_size * 3]
        large_sells = [t for t in result if t.get("isBuyerMaker", True) and float(t.get("qty", 0)) > avg_size * 3]
        
        return {
            "exchange": "binance",
            "market": "spot",
            "symbol": symbol,
            "trade_count": len(result),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_ratio_pct": round(buy_count / len(result) * 100, 1) if result else 0,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "total_volume": total_volume,
            "buy_value": buy_value,
            "sell_value": sell_value,
            "volume_imbalance_pct": round((buy_volume - sell_volume) / total_volume * 100, 2) if total_volume > 0 else 0,
            "value_imbalance_pct": round((buy_value - sell_value) / (buy_value + sell_value) * 100, 2) if (buy_value + sell_value) > 0 else 0,
            "avg_trade_size": avg_size,
            "large_buy_count": len(large_buys),
            "large_sell_count": len(large_sells),
            "latest_price": float(result[0].get("price", 0)),
            "flow_direction": "🟢 Buying" if buy_volume > sell_volume * 1.2 else "🔴 Selling" if sell_volume > buy_volume * 1.2 else "⚪ Neutral",
            "recent_trades": [
                {
                    "price": float(t.get("price", 0)),
                    "qty": float(t.get("qty", 0)),
                    "side": "BUY" if not t.get("isBuyerMaker", True) else "SELL",
                    "time": t.get("time")
                }
                for t in result[:10]
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Binance spot trades: {e}")
        return {"error": str(e)}


async def binance_spot_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get Binance spot klines/candlesticks
    
    Args:
        symbol: Trading pair
        interval: Kline interval (1s,1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M)
        limit: Number of klines (max 1000)
    
    Returns:
        OHLCV data with analysis
    """
    client = get_binance_spot_client()
    
    try:
        # Map interval string
        interval_map = {
            "1s": BinanceSpotInterval.SEC_1, "1m": BinanceSpotInterval.MIN_1,
            "3m": BinanceSpotInterval.MIN_3, "5m": BinanceSpotInterval.MIN_5,
            "15m": BinanceSpotInterval.MIN_15, "30m": BinanceSpotInterval.MIN_30,
            "1h": BinanceSpotInterval.HOUR_1, "2h": BinanceSpotInterval.HOUR_2,
            "4h": BinanceSpotInterval.HOUR_4, "6h": BinanceSpotInterval.HOUR_6,
            "8h": BinanceSpotInterval.HOUR_8, "12h": BinanceSpotInterval.HOUR_12,
            "1d": BinanceSpotInterval.DAY_1, "3d": BinanceSpotInterval.DAY_3,
            "1w": BinanceSpotInterval.WEEK_1, "1M": BinanceSpotInterval.MONTH_1
        }
        
        interval_enum = interval_map.get(interval, BinanceSpotInterval.HOUR_1)
        
        result = await client.get_klines(symbol, interval_enum, limit=limit)
        
        if "error" in result:
            return {"error": result["error"]}
        
        if not isinstance(result, list) or not result:
            return {"error": "No kline data"}
        
        # Parse klines [open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
        parsed = []
        for k in result[-50:]:  # Return last 50
            parsed.append({
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": int(k[6]),
                "quote_volume": float(k[7]),
                "trades": int(k[8]),
                "taker_buy_volume": float(k[9]),
                "taker_buy_quote": float(k[10])
            })
        
        # Stats
        closes = [float(k[4]) for k in result]
        highs = [float(k[2]) for k in result]
        lows = [float(k[3]) for k in result]
        volumes = [float(k[5]) for k in result]
        
        current = closes[-1]
        high_period = max(highs)
        low_period = min(lows)
        avg_volume = sum(volumes) / len(volumes)
        
        # Trend detection
        if len(closes) >= 10:
            recent_avg = sum(closes[-5:]) / 5
            older_avg = sum(closes[-10:-5]) / 5
            trend = "📈 Uptrend" if recent_avg > older_avg * 1.01 else "📉 Downtrend" if recent_avg < older_avg * 0.99 else "➡️ Sideways"
        else:
            trend = "Unknown"
        
        # Volume trend
        recent_vol = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else volumes[-1]
        vol_trend = "High" if recent_vol > avg_volume * 1.5 else "Low" if recent_vol < avg_volume * 0.5 else "Normal"
        
        return {
            "exchange": "binance",
            "market": "spot",
            "symbol": symbol,
            "interval": interval,
            "candle_count": len(result),
            "current_close": current,
            "current_close_formatted": f"${current:,.2f}",
            "period_high": high_period,
            "period_low": low_period,
            "period_range_pct": round((high_period - low_period) / low_period * 100, 2) if low_period > 0 else 0,
            "avg_volume": avg_volume,
            "latest_volume": volumes[-1],
            "volume_trend": vol_trend,
            "price_trend": trend,
            "klines": parsed[-20:],  # Latest 20
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Binance spot klines: {e}")
        return {"error": str(e)}


async def binance_spot_avg_price(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Get Binance spot average price (5 minute window)
    
    Args:
        symbol: Trading pair
    
    Returns:
        Current average price
    """
    client = get_binance_spot_client()
    
    try:
        result = await client.get_average_price(symbol)
        
        if "error" in result:
            return {"error": result["error"]}
        
        mins = int(result.get("mins", 5))
        price = float(result.get("price", 0))
        
        return {
            "exchange": "binance",
            "market": "spot",
            "symbol": symbol,
            "avg_price": price,
            "avg_price_formatted": f"${price:,.2f}",
            "window_minutes": mins,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Binance spot avg price: {e}")
        return {"error": str(e)}


async def binance_spot_book_ticker(symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Get Binance spot best bid/ask
    
    Args:
        symbol: Trading pair or None for all
    
    Returns:
        Best bid/ask prices and quantities
    """
    client = get_binance_spot_client()
    
    try:
        result = await client.get_book_ticker(symbol)
        
        if "error" in result:
            return {"error": result["error"]}
        
        if isinstance(result, list):
            # Multiple symbols
            tickers = []
            for t in result[:20]:  # Limit output
                bid = float(t.get("bidPrice", 0))
                ask = float(t.get("askPrice", 0))
                spread = ask - bid
                tickers.append({
                    "symbol": t.get("symbol"),
                    "bid_price": bid,
                    "bid_qty": float(t.get("bidQty", 0)),
                    "ask_price": ask,
                    "ask_qty": float(t.get("askQty", 0)),
                    "spread": spread,
                    "spread_pct": round(spread / bid * 100, 4) if bid > 0 else 0
                })
            
            return {
                "exchange": "binance",
                "market": "spot",
                "tickers": tickers,
                "count": len(result),
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Single symbol
            bid = float(result.get("bidPrice", 0))
            ask = float(result.get("askPrice", 0))
            spread = ask - bid
            
            return {
                "exchange": "binance",
                "market": "spot",
                "symbol": result.get("symbol"),
                "bid_price": bid,
                "bid_qty": float(result.get("bidQty", 0)),
                "ask_price": ask,
                "ask_qty": float(result.get("askQty", 0)),
                "spread": spread,
                "spread_pct": round(spread / bid * 100, 4) if bid > 0 else 0,
                "mid_price": (bid + ask) / 2,
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Error getting Binance spot book ticker: {e}")
        return {"error": str(e)}


async def binance_spot_agg_trades(
    symbol: str = "BTCUSDT",
    limit: int = 500
) -> Dict[str, Any]:
    """
    Get Binance spot aggregate trades
    
    Args:
        symbol: Trading pair
        limit: Number of trades (max 1000)
    
    Returns:
        Compressed trade data
    """
    client = get_binance_spot_client()
    
    try:
        result = await client.get_aggregate_trades(symbol, limit=limit)
        
        if "error" in result:
            return {"error": result["error"]}
        
        if not isinstance(result, list) or not result:
            return {"error": "No aggregate trades data"}
        
        # Analyze
        buy_vol = sum(float(t.get("q", 0)) for t in result if not t.get("m", True))
        sell_vol = sum(float(t.get("q", 0)) for t in result if t.get("m", True))
        total_vol = buy_vol + sell_vol
        
        prices = [float(t.get("p", 0)) for t in result]
        
        return {
            "exchange": "binance",
            "market": "spot",
            "symbol": symbol,
            "trade_count": len(result),
            "first_trade_id": result[-1].get("a") if result else None,
            "last_trade_id": result[0].get("a") if result else None,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "total_volume": total_vol,
            "volume_imbalance_pct": round((buy_vol - sell_vol) / total_vol * 100, 2) if total_vol > 0 else 0,
            "price_range": {
                "high": max(prices) if prices else 0,
                "low": min(prices) if prices else 0,
                "latest": prices[0] if prices else 0
            },
            "flow_direction": "🟢 Buying" if buy_vol > sell_vol * 1.2 else "🔴 Selling" if sell_vol > buy_vol * 1.2 else "⚪ Neutral",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Binance spot agg trades: {e}")
        return {"error": str(e)}


async def binance_spot_exchange_info(symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Get Binance spot exchange information
    
    Args:
        symbol: Specific symbol or None for all
    
    Returns:
        Exchange rules and symbol information
    """
    client = get_binance_spot_client()
    
    try:
        result = await client.get_exchange_info(symbol)
        
        if "error" in result:
            return {"error": result["error"]}
        
        symbols_info = result.get("symbols", [])
        
        if symbol:
            # Single symbol details
            sym_info = symbols_info[0] if symbols_info else {}
            
            filters = {f.get("filterType"): f for f in sym_info.get("filters", [])}
            price_filter = filters.get("PRICE_FILTER", {})
            lot_filter = filters.get("LOT_SIZE", {})
            notional_filter = filters.get("NOTIONAL", {})
            
            return {
                "exchange": "binance",
                "market": "spot",
                "symbol": sym_info.get("symbol"),
                "status": sym_info.get("status"),
                "base_asset": sym_info.get("baseAsset"),
                "quote_asset": sym_info.get("quoteAsset"),
                "base_precision": sym_info.get("baseAssetPrecision"),
                "quote_precision": sym_info.get("quoteAssetPrecision"),
                "order_types": sym_info.get("orderTypes"),
                "is_spot_trading": sym_info.get("isSpotTradingAllowed"),
                "is_margin_trading": sym_info.get("isMarginTradingAllowed"),
                "price_filter": {
                    "min_price": price_filter.get("minPrice"),
                    "max_price": price_filter.get("maxPrice"),
                    "tick_size": price_filter.get("tickSize")
                },
                "lot_size": {
                    "min_qty": lot_filter.get("minQty"),
                    "max_qty": lot_filter.get("maxQty"),
                    "step_size": lot_filter.get("stepSize")
                },
                "notional": {
                    "min_notional": notional_filter.get("minNotional")
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Summary
            trading = [s for s in symbols_info if s.get("status") == "TRADING"]
            usdt_pairs = [s for s in trading if s.get("quoteAsset") == "USDT"]
            
            return {
                "exchange": "binance",
                "market": "spot",
                "server_time": result.get("serverTime"),
                "timezone": result.get("timezone"),
                "total_symbols": len(symbols_info),
                "trading_symbols": len(trading),
                "usdt_pairs": len(usdt_pairs),
                "rate_limits": result.get("rateLimits"),
                "sample_symbols": [s.get("symbol") for s in usdt_pairs[:20]],
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Error getting Binance spot exchange info: {e}")
        return {"error": str(e)}


async def binance_spot_rolling_ticker(
    symbol: str = "BTCUSDT",
    window: str = "1d"
) -> Dict[str, Any]:
    """
    Get Binance spot rolling window ticker
    
    Args:
        symbol: Trading pair
        window: Window size (1m-59m, 1h-23h, 1d-7d)
    
    Returns:
        Price change stats for the window
    """
    client = get_binance_spot_client()
    
    try:
        result = await client.get_rolling_window_ticker(symbol, window_size=window)
        
        if "error" in result:
            return {"error": result["error"]}
        
        if isinstance(result, list):
            result = result[0] if result else {}
        
        price = float(result.get("lastPrice", 0))
        change = float(result.get("priceChangePercent", 0))
        
        return {
            "exchange": "binance",
            "market": "spot",
            "symbol": result.get("symbol"),
            "window": window,
            "price": price,
            "price_formatted": f"${price:,.2f}",
            "price_change": float(result.get("priceChange", 0)),
            "change_pct": round(change, 2),
            "open_price": float(result.get("openPrice", 0)),
            "high_price": float(result.get("highPrice", 0)),
            "low_price": float(result.get("lowPrice", 0)),
            "volume": float(result.get("volume", 0)),
            "quote_volume": float(result.get("quoteVolume", 0)),
            "trade_count": int(result.get("count", 0)),
            "weighted_avg_price": float(result.get("weightedAvgPrice", 0)),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Binance spot rolling ticker: {e}")
        return {"error": str(e)}


async def binance_spot_all_tickers() -> Dict[str, Any]:
    """
    Get all Binance spot tickers with summary
    
    Returns:
        All tickers with top movers analysis
    """
    client = get_binance_spot_client()
    
    try:
        result = await client.get_top_movers(limit=15)
        return {
            "exchange": "binance",
            "market": "spot",
            **result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Binance spot all tickers: {e}")
        return {"error": str(e)}


# ==================== ANALYSIS TOOLS ====================

async def binance_spot_snapshot(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Get comprehensive Binance spot market snapshot
    
    Args:
        symbol: Trading pair
    
    Returns:
        Complete market snapshot
    """
    client = get_binance_spot_client()
    
    try:
        result = await client.get_market_snapshot(symbol)
        return {
            "exchange": "binance",
            "market": "spot",
            **result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Binance spot snapshot: {e}")
        return {"error": str(e)}


async def binance_spot_full_analysis(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Get complete Binance spot market analysis
    
    Args:
        symbol: Trading pair
    
    Returns:
        Full analysis with signals
    """
    client = get_binance_spot_client()
    
    try:
        result = await client.get_full_analysis(symbol)
        return {
            "exchange": "binance",
            "market": "spot",
            **result
        }
        
    except Exception as e:
        logger.error(f"Error getting Binance spot full analysis: {e}")
        return {"error": str(e)}
