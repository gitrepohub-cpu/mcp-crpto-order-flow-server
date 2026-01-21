"""
Binance Futures REST API Client - Comprehensive data fetching.
Implements all publicly available REST endpoints for market data and analytics.

Base URL: https://fapi.binance.com

Endpoints Implemented:
=====================

HIGH PRIORITY (Market Data):
- /fapi/v1/openInterest - Current open interest
- /futures/data/openInterestHist - Historical OI (5m-1d intervals)
- /fapi/v1/fundingRate - Historical funding rates
- /fapi/v1/depth - Full orderbook (up to 1000 levels)
- /fapi/v1/aggTrades - Recent aggregated trades
- /fapi/v1/klines - Historical OHLCV
- /fapi/v1/indexPriceKlines - Index price candles
- /fapi/v1/markPriceKlines - Mark price candles
- /fapi/v1/premiumIndex - Premium index data

POSITIONING DATA:
- /futures/data/topLongShortAccountRatio - Top traders long/short
- /futures/data/topLongShortPositionRatio - Top positions long/short
- /futures/data/globalLongShortAccountRatio - Global long/short
- /futures/data/takerlongshortRatio - Taker buy/sell ratio

TICKER DATA:
- /fapi/v1/ticker/24hr - 24h statistics
- /fapi/v1/ticker/price - Current prices
- /fapi/v1/ticker/bookTicker - Best bid/ask

UNIQUE DATA:
- /futures/data/basis - Futures basis data
- /fapi/v1/forceOrders - Recent liquidations
- /fapi/v1/exchangeInfo - Exchange information
- /fapi/v1/leverageBracket - Leverage brackets
"""

import asyncio
import aiohttp
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class KlineInterval(str, Enum):
    """Kline/Candlestick intervals for Binance."""
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


class OIHistPeriod(str, Enum):
    """Open Interest History periods."""
    MINUTES_5 = "5m"
    MINUTES_15 = "15m"
    MINUTES_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_12 = "12h"
    DAY_1 = "1d"


class ContractType(str, Enum):
    """Contract types for Binance Futures."""
    PERPETUAL = "PERPETUAL"
    CURRENT_MONTH = "CURRENT_MONTH"
    NEXT_MONTH = "NEXT_MONTH"
    CURRENT_QUARTER = "CURRENT_QUARTER"
    NEXT_QUARTER = "NEXT_QUARTER"
    PERPETUAL_DELIVERING = "PERPETUAL_DELIVERING"


class BinanceFuturesREST:
    """
    Comprehensive Binance Futures REST API client.
    Fetches all available market data, positioning, and analytics endpoints.
    """
    
    BASE_URL = "https://fapi.binance.com"
    FUTURES_DATA_URL = "https://fapi.binance.com"  # Same base for futures data
    
    # Standard perpetual symbols
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    
    def __init__(self, rate_limit_delay: float = 0.1):
        """
        Initialize Binance Futures REST client.
        
        Args:
            rate_limit_delay: Delay between requests to avoid rate limits (seconds)
        """
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_delay = rate_limit_delay
        self._last_request_time = 0
        
        # Cache for exchange info (rarely changes)
        self._exchange_info_cache: Optional[Dict] = None
        self._exchange_info_timestamp: float = 0
        self._exchange_info_ttl = 3600  # 1 hour cache
        
        logger.info("BinanceFuturesREST client initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """
        Make a rate-limited request to Binance API.
        
        Args:
            endpoint: API endpoint (e.g., "/fapi/v1/openInterest")
            params: Query parameters
            
        Returns:
            JSON response data
        """
        # Rate limiting
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - elapsed)
        
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                self._last_request_time = time.time()
                
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    # Rate limit - only retry up to 3 times
                    retry_count = getattr(self, '_retry_count', 0)
                    if retry_count >= 3:
                        logger.error("Binance rate limit hit 3 times, giving up")
                        self._retry_count = 0
                        return None
                    self._retry_count = retry_count + 1
                    logger.warning(f"Binance rate limit hit, waiting 60s... (retry {self._retry_count}/3)")
                    await asyncio.sleep(60)
                    result = await self._request(endpoint, params)
                    self._retry_count = 0
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Binance API error {response.status}: {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching {endpoint}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {endpoint}: {e}")
            return None
    
    # ========================================================================
    # MARKET DATA - PRICES & TICKERS
    # ========================================================================
    
    async def get_ticker_price(self, symbol: Optional[str] = None) -> Dict:
        """
        Get latest price(s).
        
        GET /fapi/v1/ticker/price
        
        Args:
            symbol: Optional specific symbol, or None for all
            
        Returns:
            Dict with symbol -> price mapping
        """
        params = {"symbol": symbol} if symbol else {}
        data = await self._request("/fapi/v1/ticker/price", params)
        
        if data is None:
            return {}
        
        if isinstance(data, list):
            return {item["symbol"]: float(item["price"]) for item in data}
        return {data["symbol"]: float(data["price"])}
    
    async def get_ticker_24hr(self, symbol: Optional[str] = None) -> Dict:
        """
        Get 24h ticker statistics.
        
        GET /fapi/v1/ticker/24hr
        
        Args:
            symbol: Optional specific symbol
            
        Returns:
            Dict with comprehensive 24h stats
        """
        params = {"symbol": symbol} if symbol else {}
        data = await self._request("/fapi/v1/ticker/24hr", params)
        
        if data is None:
            return {}
        
        def parse_ticker(t):
            return {
                "symbol": t["symbol"],
                "price_change": float(t.get("priceChange", 0)),
                "price_change_percent": float(t.get("priceChangePercent", 0)),
                "weighted_avg_price": float(t.get("weightedAvgPrice", 0)),
                "last_price": float(t.get("lastPrice", 0)),
                "last_qty": float(t.get("lastQty", 0)),
                "open_price": float(t.get("openPrice", 0)),
                "high_price": float(t.get("highPrice", 0)),
                "low_price": float(t.get("lowPrice", 0)),
                "volume": float(t.get("volume", 0)),
                "quote_volume": float(t.get("quoteVolume", 0)),
                "open_time": int(t.get("openTime", 0)),
                "close_time": int(t.get("closeTime", 0)),
                "first_id": int(t.get("firstId", 0)),
                "last_id": int(t.get("lastId", 0)),
                "count": int(t.get("count", 0)),
            }
        
        if isinstance(data, list):
            return {item["symbol"]: parse_ticker(item) for item in data}
        return {data["symbol"]: parse_ticker(data)}
    
    async def get_book_ticker(self, symbol: Optional[str] = None) -> Dict:
        """
        Get best bid/ask prices.
        
        GET /fapi/v1/ticker/bookTicker
        
        Args:
            symbol: Optional specific symbol
            
        Returns:
            Dict with bid/ask prices and quantities
        """
        params = {"symbol": symbol} if symbol else {}
        data = await self._request("/fapi/v1/ticker/bookTicker", params)
        
        if data is None:
            return {}
        
        def parse_book_ticker(t):
            return {
                "symbol": t["symbol"],
                "bid_price": float(t.get("bidPrice", 0)),
                "bid_qty": float(t.get("bidQty", 0)),
                "ask_price": float(t.get("askPrice", 0)),
                "ask_qty": float(t.get("askQty", 0)),
                "time": int(t.get("time", 0)),
            }
        
        if isinstance(data, list):
            return {item["symbol"]: parse_book_ticker(item) for item in data}
        return {data["symbol"]: parse_book_ticker(data)}
    
    # ========================================================================
    # MARKET DATA - ORDERBOOK
    # ========================================================================
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """
        Get full orderbook depth.
        
        GET /fapi/v1/depth
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            limit: Depth limit (5, 10, 20, 50, 100, 500, 1000)
            
        Returns:
            Dict with bids, asks, and metadata
        """
        params = {"symbol": symbol, "limit": min(limit, 1000)}
        data = await self._request("/fapi/v1/depth", params)
        
        if data is None:
            return {}
        
        return {
            "symbol": symbol,
            "last_update_id": data.get("lastUpdateId", 0),
            "message_time": data.get("E", 0),
            "transaction_time": data.get("T", 0),
            "bids": [[float(b[0]), float(b[1])] for b in data.get("bids", [])],
            "asks": [[float(a[0]), float(a[1])] for a in data.get("asks", [])],
            "bid_depth": sum(float(b[1]) for b in data.get("bids", [])),
            "ask_depth": sum(float(a[1]) for a in data.get("asks", [])),
            "timestamp": int(time.time() * 1000),
        }
    
    async def get_all_orderbooks(self, symbols: Optional[List[str]] = None, limit: int = 100) -> Dict:
        """
        Get orderbooks for multiple symbols.
        
        Args:
            symbols: List of symbols (default: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT)
            limit: Depth per orderbook
            
        Returns:
            Dict mapping symbol -> orderbook data
        """
        symbols = symbols or self.SYMBOLS
        tasks = [self.get_orderbook(sym, limit) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            sym: result for sym, result in zip(symbols, results)
            if isinstance(result, dict) and result
        }
    
    # ========================================================================
    # MARKET DATA - TRADES
    # ========================================================================
    
    async def get_agg_trades(self, symbol: str, limit: int = 500, 
                             start_time: Optional[int] = None,
                             end_time: Optional[int] = None) -> List[Dict]:
        """
        Get recent aggregated trades.
        
        GET /fapi/v1/aggTrades
        
        Args:
            symbol: Trading pair
            limit: Number of trades (max 1000)
            start_time: Start timestamp (ms)
            end_time: End timestamp (ms)
            
        Returns:
            List of aggregated trades
        """
        params = {"symbol": symbol, "limit": min(limit, 1000)}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = await self._request("/fapi/v1/aggTrades", params)
        
        if data is None:
            return []
        
        return [{
            "agg_trade_id": t.get("a", 0),
            "price": float(t.get("p", 0)),
            "quantity": float(t.get("q", 0)),
            "first_trade_id": t.get("f", 0),
            "last_trade_id": t.get("l", 0),
            "timestamp": t.get("T", 0),
            "is_buyer_maker": t.get("m", False),
        } for t in data]
    
    # ========================================================================
    # MARKET DATA - KLINES (OHLCV)
    # ========================================================================
    
    async def get_klines(self, symbol: str, interval: str = "1m", 
                         limit: int = 500, start_time: Optional[int] = None,
                         end_time: Optional[int] = None) -> List[Dict]:
        """
        Get OHLCV candlestick data.
        
        GET /fapi/v1/klines
        
        Args:
            symbol: Trading pair
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Number of candles (max 1500)
            start_time: Start timestamp (ms)
            end_time: End timestamp (ms)
            
        Returns:
            List of OHLCV candles
        """
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = await self._request("/fapi/v1/klines", params)
        
        if data is None:
            return []
        
        return [{
            "open_time": k[0],
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "close_time": k[6],
            "quote_volume": float(k[7]),
            "trades": int(k[8]),
            "taker_buy_base_volume": float(k[9]),
            "taker_buy_quote_volume": float(k[10]),
        } for k in data]
    
    async def get_index_price_klines(self, pair: str, interval: str = "1m",
                                     limit: int = 500, start_time: Optional[int] = None,
                                     end_time: Optional[int] = None) -> List[Dict]:
        """
        Get index price klines.
        
        GET /fapi/v1/indexPriceKlines
        
        Args:
            pair: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval
            limit: Number of candles (max 1500)
            
        Returns:
            List of index price candles
        """
        params = {"pair": pair, "interval": interval, "limit": min(limit, 1500)}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = await self._request("/fapi/v1/indexPriceKlines", params)
        
        if data is None:
            return []
        
        return [{
            "open_time": k[0],
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "close_time": k[6],
        } for k in data]
    
    async def get_mark_price_klines(self, symbol: str, interval: str = "1m",
                                    limit: int = 500, start_time: Optional[int] = None,
                                    end_time: Optional[int] = None) -> List[Dict]:
        """
        Get mark price klines.
        
        GET /fapi/v1/markPriceKlines
        
        Args:
            symbol: Trading pair
            interval: Kline interval
            limit: Number of candles (max 1500)
            
        Returns:
            List of mark price candles
        """
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = await self._request("/fapi/v1/markPriceKlines", params)
        
        if data is None:
            return []
        
        return [{
            "open_time": k[0],
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "close_time": k[6],
        } for k in data]
    
    # ========================================================================
    # OPEN INTEREST
    # ========================================================================
    
    async def get_open_interest(self, symbol: str) -> Dict:
        """
        Get current open interest.
        
        GET /fapi/v1/openInterest
        
        Args:
            symbol: Trading pair
            
        Returns:
            Dict with OI data
        """
        params = {"symbol": symbol}
        data = await self._request("/fapi/v1/openInterest", params)
        
        if data is None:
            return {}
        
        return {
            "symbol": data.get("symbol", symbol),
            "open_interest": float(data.get("openInterest", 0)),
            "time": int(data.get("time", 0)),
            "timestamp": int(time.time() * 1000),
        }
    
    async def get_all_open_interest(self, symbols: Optional[List[str]] = None) -> Dict:
        """
        Get open interest for multiple symbols.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dict mapping symbol -> OI data
        """
        symbols = symbols or self.SYMBOLS
        tasks = [self.get_open_interest(sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            sym: result for sym, result in zip(symbols, results)
            if isinstance(result, dict) and result
        }
    
    async def get_open_interest_hist(self, symbol: str, period: str = "5m",
                                     limit: int = 500, start_time: Optional[int] = None,
                                     end_time: Optional[int] = None) -> List[Dict]:
        """
        Get historical open interest.
        
        GET /futures/data/openInterestHist
        
        Args:
            symbol: Trading pair
            period: Period interval (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            limit: Number of data points (max 500)
            start_time: Start timestamp (ms)
            end_time: End timestamp (ms)
            
        Returns:
            List of historical OI data points
        """
        params = {
            "symbol": symbol,
            "period": period,
            "limit": min(limit, 500)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = await self._request("/futures/data/openInterestHist", params)
        
        if data is None:
            return []
        
        return [{
            "symbol": d.get("symbol", symbol),
            "sum_open_interest": float(d.get("sumOpenInterest", 0)),
            "sum_open_interest_value": float(d.get("sumOpenInterestValue", 0)),
            "timestamp": int(d.get("timestamp", 0)),
        } for d in data]
    
    # ========================================================================
    # FUNDING RATE
    # ========================================================================
    
    async def get_funding_rate(self, symbol: str, limit: int = 100,
                               start_time: Optional[int] = None,
                               end_time: Optional[int] = None) -> List[Dict]:
        """
        Get historical funding rates.
        
        GET /fapi/v1/fundingRate
        
        Args:
            symbol: Trading pair
            limit: Number of records (max 1000)
            start_time: Start timestamp (ms)
            end_time: End timestamp (ms)
            
        Returns:
            List of funding rate records
        """
        params = {"symbol": symbol, "limit": min(limit, 1000)}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = await self._request("/fapi/v1/fundingRate", params)
        
        if data is None:
            return []
        
        return [{
            "symbol": d.get("symbol", symbol),
            "funding_rate": float(d.get("fundingRate", 0)),
            "funding_time": int(d.get("fundingTime", 0)),
            "mark_price": float(d.get("markPrice", 0)) if d.get("markPrice") else None,
        } for d in data]
    
    async def get_premium_index(self, symbol: Optional[str] = None) -> Dict:
        """
        Get premium index / mark price.
        
        GET /fapi/v1/premiumIndex
        
        Args:
            symbol: Optional specific symbol
            
        Returns:
            Dict with premium index data
        """
        params = {"symbol": symbol} if symbol else {}
        data = await self._request("/fapi/v1/premiumIndex", params)
        
        if data is None:
            return {}
        
        def parse_premium(p):
            return {
                "symbol": p.get("symbol", ""),
                "mark_price": float(p.get("markPrice", 0)),
                "index_price": float(p.get("indexPrice", 0)),
                "estimated_settle_price": float(p.get("estimatedSettlePrice", 0)),
                "last_funding_rate": float(p.get("lastFundingRate", 0)),
                "next_funding_time": int(p.get("nextFundingTime", 0)),
                "interest_rate": float(p.get("interestRate", 0)),
                "time": int(p.get("time", 0)),
            }
        
        if isinstance(data, list):
            return {item["symbol"]: parse_premium(item) for item in data}
        return {data["symbol"]: parse_premium(data)}
    
    # ========================================================================
    # LONG/SHORT RATIO - POSITIONING DATA
    # ========================================================================
    
    async def get_top_long_short_account_ratio(self, symbol: str, period: str = "5m",
                                               limit: int = 500,
                                               start_time: Optional[int] = None,
                                               end_time: Optional[int] = None) -> List[Dict]:
        """
        Get top traders long/short account ratio.
        
        GET /futures/data/topLongShortAccountRatio
        
        Args:
            symbol: Trading pair
            period: Period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            limit: Number of data points (max 500)
            
        Returns:
            List of long/short ratios
        """
        params = {
            "symbol": symbol,
            "period": period,
            "limit": min(limit, 500)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = await self._request("/futures/data/topLongShortAccountRatio", params)
        
        if data is None:
            return []
        
        return [{
            "symbol": d.get("symbol", symbol),
            "long_short_ratio": float(d.get("longShortRatio", 0)),
            "long_account": float(d.get("longAccount", 0)),
            "short_account": float(d.get("shortAccount", 0)),
            "timestamp": int(d.get("timestamp", 0)),
        } for d in data]
    
    async def get_top_long_short_position_ratio(self, symbol: str, period: str = "5m",
                                                limit: int = 500,
                                                start_time: Optional[int] = None,
                                                end_time: Optional[int] = None) -> List[Dict]:
        """
        Get top traders long/short position ratio.
        
        GET /futures/data/topLongShortPositionRatio
        
        Args:
            symbol: Trading pair
            period: Period
            limit: Number of data points (max 500)
            
        Returns:
            List of long/short position ratios
        """
        params = {
            "symbol": symbol,
            "period": period,
            "limit": min(limit, 500)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = await self._request("/futures/data/topLongShortPositionRatio", params)
        
        if data is None:
            return []
        
        return [{
            "symbol": d.get("symbol", symbol),
            "long_short_ratio": float(d.get("longShortRatio", 0)),
            "long_account": float(d.get("longAccount", 0)),
            "short_account": float(d.get("shortAccount", 0)),
            "timestamp": int(d.get("timestamp", 0)),
        } for d in data]
    
    async def get_global_long_short_account_ratio(self, symbol: str, period: str = "5m",
                                                  limit: int = 500,
                                                  start_time: Optional[int] = None,
                                                  end_time: Optional[int] = None) -> List[Dict]:
        """
        Get global long/short account ratio (all traders).
        
        GET /futures/data/globalLongShortAccountRatio
        
        Args:
            symbol: Trading pair
            period: Period
            limit: Number of data points (max 500)
            
        Returns:
            List of global long/short ratios
        """
        params = {
            "symbol": symbol,
            "period": period,
            "limit": min(limit, 500)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = await self._request("/futures/data/globalLongShortAccountRatio", params)
        
        if data is None:
            return []
        
        return [{
            "symbol": d.get("symbol", symbol),
            "long_short_ratio": float(d.get("longShortRatio", 0)),
            "long_account": float(d.get("longAccount", 0)),
            "short_account": float(d.get("shortAccount", 0)),
            "timestamp": int(d.get("timestamp", 0)),
        } for d in data]
    
    async def get_taker_long_short_ratio(self, symbol: str, period: str = "5m",
                                         limit: int = 500,
                                         start_time: Optional[int] = None,
                                         end_time: Optional[int] = None) -> List[Dict]:
        """
        Get taker buy/sell volume ratio.
        
        GET /futures/data/takerlongshortRatio
        
        Args:
            symbol: Trading pair
            period: Period
            limit: Number of data points (max 500)
            
        Returns:
            List of taker buy/sell ratios
        """
        params = {
            "symbol": symbol,
            "period": period,
            "limit": min(limit, 500)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = await self._request("/futures/data/takerlongshortRatio", params)
        
        if data is None:
            return []
        
        return [{
            "symbol": d.get("symbol", symbol),
            "buy_sell_ratio": float(d.get("buySellRatio", 0)),
            "buy_vol": float(d.get("buyVol", 0)),
            "sell_vol": float(d.get("sellVol", 0)),
            "timestamp": int(d.get("timestamp", 0)),
        } for d in data]
    
    # ========================================================================
    # BASIS DATA
    # ========================================================================
    
    async def get_basis(self, pair: str, contract_type: str = "PERPETUAL",
                        period: str = "5m", limit: int = 500,
                        start_time: Optional[int] = None,
                        end_time: Optional[int] = None) -> List[Dict]:
        """
        Get futures basis data.
        
        GET /futures/data/basis
        
        Args:
            pair: Trading pair (e.g., "BTCUSDT")
            contract_type: PERPETUAL, CURRENT_QUARTER, NEXT_QUARTER
            period: Period
            limit: Number of data points (max 500)
            
        Returns:
            List of basis data points
        """
        params = {
            "pair": pair,
            "contractType": contract_type,
            "period": period,
            "limit": min(limit, 500)
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = await self._request("/futures/data/basis", params)
        
        if data is None:
            return []
        
        return [{
            "pair": d.get("pair", pair),
            "contract_type": d.get("contractType", contract_type),
            "futures_price": float(d.get("futuresPrice", 0)),
            "index_price": float(d.get("indexPrice", 0)),
            "basis": float(d.get("basis", 0)),
            "basis_rate": float(d.get("basisRate", 0)),
            "timestamp": int(d.get("timestamp", 0)),
        } for d in data]
    
    # ========================================================================
    # LIQUIDATIONS
    # ========================================================================
    
    async def get_force_orders(self, symbol: Optional[str] = None,
                               auto_close_type: Optional[str] = None,
                               limit: int = 100,
                               start_time: Optional[int] = None,
                               end_time: Optional[int] = None) -> List[Dict]:
        """
        Get recent liquidation orders.
        
        GET /fapi/v1/forceOrders
        
        Args:
            symbol: Optional specific symbol
            auto_close_type: "LIQUIDATION" or "ADL"
            limit: Number of orders (max 1000)
            start_time: Start timestamp (ms)
            end_time: End timestamp (ms)
            
        Returns:
            List of liquidation orders
        """
        params = {"limit": min(limit, 1000)}
        if symbol:
            params["symbol"] = symbol
        if auto_close_type:
            params["autoCloseType"] = auto_close_type
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = await self._request("/fapi/v1/forceOrders", params)
        
        if data is None:
            return []
        
        return [{
            "order_id": d.get("orderId", 0),
            "symbol": d.get("symbol", ""),
            "status": d.get("status", ""),
            "client_order_id": d.get("clientOrderId", ""),
            "price": float(d.get("price", 0)),
            "avg_price": float(d.get("avgPrice", 0)),
            "orig_qty": float(d.get("origQty", 0)),
            "executed_qty": float(d.get("executedQty", 0)),
            "cum_quote": float(d.get("cumQuote", 0)),
            "time_in_force": d.get("timeInForce", ""),
            "type": d.get("type", ""),
            "side": d.get("side", ""),
            "time": int(d.get("time", 0)),
            "update_time": int(d.get("updateTime", 0)),
        } for d in data]
    
    # ========================================================================
    # EXCHANGE INFO
    # ========================================================================
    
    async def get_exchange_info(self, force_refresh: bool = False) -> Dict:
        """
        Get exchange information (cached).
        
        GET /fapi/v1/exchangeInfo
        
        Args:
            force_refresh: Force refresh cache
            
        Returns:
            Dict with exchange info
        """
        now = time.time()
        
        if (not force_refresh and self._exchange_info_cache 
            and (now - self._exchange_info_timestamp) < self._exchange_info_ttl):
            return self._exchange_info_cache
        
        data = await self._request("/fapi/v1/exchangeInfo")
        
        if data:
            self._exchange_info_cache = data
            self._exchange_info_timestamp = now
        
        return data or {}
    
    async def get_leverage_brackets(self, symbol: Optional[str] = None) -> Dict:
        """
        Get leverage bracket information.
        
        GET /fapi/v1/leverageBracket
        
        Note: This endpoint requires authentication for user-specific brackets.
        Public data returns symbol-level defaults.
        
        Args:
            symbol: Optional specific symbol
            
        Returns:
            Dict with leverage bracket info
        """
        params = {"symbol": symbol} if symbol else {}
        data = await self._request("/fapi/v1/leverageBracket", params)
        
        if data is None:
            return {}
        
        def parse_brackets(item):
            return {
                "symbol": item.get("symbol", ""),
                "brackets": [{
                    "bracket": b.get("bracket", 0),
                    "initial_leverage": b.get("initialLeverage", 0),
                    "notional_cap": b.get("notionalCap", 0),
                    "notional_floor": b.get("notionalFloor", 0),
                    "maint_margin_ratio": float(b.get("maintMarginRatio", 0)),
                    "cum": float(b.get("cum", 0)),
                } for b in item.get("brackets", [])]
            }
        
        if isinstance(data, list):
            return {item["symbol"]: parse_brackets(item) for item in data}
        return {data["symbol"]: parse_brackets(data)}
    
    # ========================================================================
    # COMPREHENSIVE DATA FETCHING
    # ========================================================================
    
    async def get_market_snapshot(self, symbol: str) -> Dict:
        """
        Get a comprehensive market snapshot for a symbol.
        
        Fetches:
        - Current prices and 24h stats
        - Open interest
        - Funding rate
        - Premium index
        - Best bid/ask
        
        Args:
            symbol: Trading pair
            
        Returns:
            Dict with comprehensive market data
        """
        tasks = [
            self.get_ticker_24hr(symbol),
            self.get_open_interest(symbol),
            self.get_premium_index(symbol),
            self.get_book_ticker(symbol),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        ticker = results[0].get(symbol, {}) if isinstance(results[0], dict) else {}
        oi = results[1] if isinstance(results[1], dict) else {}
        premium = results[2].get(symbol, {}) if isinstance(results[2], dict) else {}
        book = results[3].get(symbol, {}) if isinstance(results[3], dict) else {}
        
        return {
            "symbol": symbol,
            "timestamp": int(time.time() * 1000),
            "price": {
                "last": ticker.get("last_price", 0),
                "mark": premium.get("mark_price", 0),
                "index": premium.get("index_price", 0),
                "bid": book.get("bid_price", 0),
                "ask": book.get("ask_price", 0),
                "spread": book.get("ask_price", 0) - book.get("bid_price", 0),
            },
            "volume_24h": {
                "base": ticker.get("volume", 0),
                "quote": ticker.get("quote_volume", 0),
            },
            "price_change_24h": {
                "absolute": ticker.get("price_change", 0),
                "percent": ticker.get("price_change_percent", 0),
            },
            "range_24h": {
                "high": ticker.get("high_price", 0),
                "low": ticker.get("low_price", 0),
            },
            "open_interest": oi.get("open_interest", 0),
            "funding": {
                "rate": premium.get("last_funding_rate", 0),
                "next_time": premium.get("next_funding_time", 0),
            },
        }
    
    async def get_positioning_data(self, symbol: str, period: str = "5m",
                                   limit: int = 30) -> Dict:
        """
        Get comprehensive positioning data.
        
        Fetches:
        - Top trader long/short account ratio
        - Top trader long/short position ratio
        - Global long/short ratio
        - Taker buy/sell ratio
        
        Args:
            symbol: Trading pair
            period: Data period
            limit: Number of data points
            
        Returns:
            Dict with positioning data
        """
        tasks = [
            self.get_top_long_short_account_ratio(symbol, period, limit),
            self.get_top_long_short_position_ratio(symbol, period, limit),
            self.get_global_long_short_account_ratio(symbol, period, limit),
            self.get_taker_long_short_ratio(symbol, period, limit),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "symbol": symbol,
            "period": period,
            "timestamp": int(time.time() * 1000),
            "top_trader_accounts": results[0] if isinstance(results[0], list) else [],
            "top_trader_positions": results[1] if isinstance(results[1], list) else [],
            "global_accounts": results[2] if isinstance(results[2], list) else [],
            "taker_volume": results[3] if isinstance(results[3], list) else [],
            "latest": {
                "top_trader_ls_ratio": results[0][-1]["long_short_ratio"] if isinstance(results[0], list) and results[0] else 0,
                "global_ls_ratio": results[2][-1]["long_short_ratio"] if isinstance(results[2], list) and results[2] else 0,
                "taker_buy_sell_ratio": results[3][-1]["buy_sell_ratio"] if isinstance(results[3], list) and results[3] else 0,
            }
        }
    
    async def get_historical_analysis(self, symbol: str, period: str = "5m",
                                      limit: int = 200) -> Dict:
        """
        Get historical data for analysis.
        
        Fetches:
        - Historical OI
        - Historical funding
        - Basis data
        - OHLCV klines
        
        Args:
            symbol: Trading pair
            period: Data period
            limit: Number of data points
            
        Returns:
            Dict with historical data
        """
        # Map period to kline interval
        kline_interval = period if period in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"] else "5m"
        
        tasks = [
            self.get_open_interest_hist(symbol, period, limit),
            self.get_funding_rate(symbol, limit),
            self.get_basis(symbol, "PERPETUAL", period, limit),
            self.get_klines(symbol, kline_interval, limit),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "symbol": symbol,
            "period": period,
            "timestamp": int(time.time() * 1000),
            "open_interest_history": results[0] if isinstance(results[0], list) else [],
            "funding_rate_history": results[1] if isinstance(results[1], list) else [],
            "basis_history": results[2] if isinstance(results[2], list) else [],
            "klines": results[3] if isinstance(results[3], list) else [],
        }
    
    async def get_full_market_data(self, symbols: Optional[List[str]] = None) -> Dict:
        """
        Get full market data for all symbols.
        
        Args:
            symbols: List of symbols (default: all supported)
            
        Returns:
            Dict with complete market data
        """
        symbols = symbols or self.SYMBOLS
        
        # Fetch all data in parallel
        tasks = [self.get_market_snapshot(sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            sym: result for sym, result in zip(symbols, results)
            if isinstance(result, dict)
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_binance_rest_client: Optional[BinanceFuturesREST] = None


def get_binance_rest_client() -> BinanceFuturesREST:
    """Get singleton instance of Binance REST client."""
    global _binance_rest_client
    if _binance_rest_client is None:
        _binance_rest_client = BinanceFuturesREST()
    return _binance_rest_client


async def close_binance_rest_client():
    """Close the singleton REST client."""
    global _binance_rest_client
    if _binance_rest_client:
        await _binance_rest_client.close()
        _binance_rest_client = None
