"""
Ticker Features Tool - Part of Institutional Features
=====================================================

Provides 10 ticker-based features for 24h market statistics.

Features:
1. volume_24h - 24-hour trading volume in quote currency
2. high_24h - 24-hour high price
3. low_24h - 24-hour low price
4. price_change_24h - Absolute price change in 24h
5. price_change_pct - Percentage price change in 24h
6. volume_ratio - Current volume vs average volume
7. range_24h - High - Low range
8. range_position - Where current price sits in 24h range (0-1)
9. volatility_24h - Estimated volatility from range
10. volume_profile - Volume distribution indicator
"""

from typing import Dict, Any
from datetime import datetime
import logging
import aiohttp

logger = logging.getLogger(__name__)


async def get_ticker_features(
    symbol: str = "BTCUSDT",
    exchange: str = "binance"
) -> Dict[str, Any]:
    """
    Get ticker features (10 features) for a symbol.
    
    Args:
        symbol: Trading pair symbol
        exchange: Exchange name
        
    Returns:
        Dictionary with 10 ticker features
    """
    try:
        ticker = await _fetch_ticker(symbol, exchange)
        
        # Extract values with safe defaults
        volume_24h = float(ticker.get("quoteVolume", ticker.get("volume24h", ticker.get("turnover24h", 0))))
        high_24h = float(ticker.get("highPrice", ticker.get("high24h", ticker.get("highPrice24h", 0))))
        low_24h = float(ticker.get("lowPrice", ticker.get("low24h", ticker.get("lowPrice24h", 0))))
        last_price = float(ticker.get("lastPrice", ticker.get("last", ticker.get("lastTradedPrice", 0))))
        open_price = float(ticker.get("openPrice", ticker.get("open24h", ticker.get("prevPrice24h", last_price))))
        
        # Calculate derived features
        price_change_24h = last_price - open_price
        price_change_pct = (price_change_24h / open_price * 100) if open_price else 0
        
        range_24h = high_24h - low_24h
        range_position = (last_price - low_24h) / range_24h if range_24h > 0 else 0.5
        
        # Volatility estimate from range (Parkinson estimator simplified)
        volatility_24h = (range_24h / last_price) if last_price > 0 else 0
        
        # Volume profile indicator
        taker_buy_volume = float(ticker.get("takerBuyQuoteVolume", ticker.get("takerBuyVolume", volume_24h * 0.5)))
        taker_sell_volume = volume_24h - taker_buy_volume
        volume_profile = taker_buy_volume / volume_24h if volume_24h > 0 else 0.5
        
        # Volume ratio (placeholder - would need historical average)
        volume_ratio = 1.0
        
        # VWAP approximation
        vwap = (high_24h + low_24h + last_price) / 3 if high_24h > 0 else last_price
        
        # ATR approximation from range
        atr_estimate = range_24h
        
        return {
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.utcnow().isoformat(),
            "feature_count": 10,
            
            # 10 Ticker Features
            "volume_24h": volume_24h,
            "high_24h": high_24h,
            "low_24h": low_24h,
            "price_change_24h": price_change_24h,
            "price_change_pct": price_change_pct,
            "volume_ratio": volume_ratio,
            "range_24h": range_24h,
            "range_position": range_position,
            "volatility_24h": volatility_24h,
            "volume_profile": volume_profile,
            
            # Additional derived metrics
            "last_price": last_price,
            "open_price": open_price,
            "taker_buy_volume": taker_buy_volume,
            "taker_sell_volume": taker_sell_volume,
            "vwap_estimate": vwap,
            "atr_estimate": atr_estimate,
        }
        
    except Exception as e:
        logger.error(f"Error getting ticker features for {symbol} on {exchange}: {e}")
        return {
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "feature_count": 10,
            
            # Return zeros for all features
            "volume_24h": 0,
            "high_24h": 0,
            "low_24h": 0,
            "price_change_24h": 0,
            "price_change_pct": 0,
            "volume_ratio": 1.0,
            "range_24h": 0,
            "range_position": 0.5,
            "volatility_24h": 0,
            "volume_profile": 0.5,
        }


async def _fetch_ticker(symbol: str, exchange: str) -> Dict[str, Any]:
    """Fetch ticker data from exchange API"""
    
    exchange_lower = exchange.lower()
    
    async with aiohttp.ClientSession() as session:
        if exchange_lower == "binance":
            url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                    
        elif exchange_lower == "bybit":
            url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("result", {}).get("list"):
                        return data["result"]["list"][0]
                        
        elif exchange_lower == "okx":
            # OKX uses different symbol format
            okx_symbol = symbol.replace("USDT", "-USDT-SWAP")
            url = f"https://www.okx.com/api/v5/market/ticker?instId={okx_symbol}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("data"):
                        return data["data"][0]
                        
        elif exchange_lower == "hyperliquid":
            url = "https://api.hyperliquid.xyz/info"
            payload = {"type": "allMids"}
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Return basic ticker from mids
                    return {"lastPrice": data.get(symbol, 0)}
                    
        elif exchange_lower in ["kraken", "gateio", "deribit"]:
            # Fallback to binance for these exchanges for now
            url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
        
        # Default fallback
        return {}


# Sync wrapper for non-async contexts
def get_ticker_features_sync(symbol: str = "BTCUSDT", exchange: str = "binance") -> Dict[str, Any]:
    """Synchronous wrapper for get_ticker_features"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(get_ticker_features(symbol, exchange))
