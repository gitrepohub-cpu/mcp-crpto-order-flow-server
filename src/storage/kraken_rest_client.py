"""
Kraken REST API Client
Comprehensive implementation of all available public Kraken endpoints
Supports both Spot and Futures APIs
"""

import aiohttp
import asyncio
import time
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class KrakenInterval(int, Enum):
    """Kline intervals for Kraken (in minutes)"""
    MIN_1 = 1
    MIN_5 = 5
    MIN_15 = 15
    MIN_30 = 30
    HOUR_1 = 60
    HOUR_4 = 240
    DAY_1 = 1440
    WEEK_1 = 10080
    DAY_15 = 21600


class KrakenFuturesInterval(str, Enum):
    """Kline intervals for Kraken Futures"""
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


@dataclass
class KrakenRateLimitConfig:
    """Rate limit configuration for Kraken API"""
    # Kraken: 15 requests per second for public endpoints
    requests_per_second: int = 15
    weight_per_second: int = 100


class KrakenRESTClient:
    """
    Comprehensive Kraken REST API client.
    
    SPOT API ENDPOINTS (api.kraken.com):
    1. /0/public/Time - Server time
    2. /0/public/SystemStatus - System status
    3. /0/public/Assets - Asset info
    4. /0/public/AssetPairs - Trading pairs info
    5. /0/public/Ticker - Ticker information
    6. /0/public/OHLC - OHLC/Candlestick data
    7. /0/public/Depth - Order book
    8. /0/public/Trades - Recent trades
    9. /0/public/Spread - Recent spread data
    
    FUTURES API ENDPOINTS (futures.kraken.com):
    10. /derivatives/api/v3/tickers - All futures tickers
    11. /derivatives/api/v3/orderbook - Futures order book
    12. /derivatives/api/v3/history - Trade history
    13. /derivatives/api/v3/instruments - Instruments info
    14. /api/charts/v1/spot/{symbol}/{interval} - Spot charts
    15. /api/charts/v1/trade/{symbol}/{interval} - Futures charts
    16. /derivatives/api/v3/feeschedules/volumes - Fee schedules
    
    FUTURES MARKET DATA:
    17. /derivatives/api/v4/markets - Markets overview
    18. /derivatives/api/v4/markets/{symbol}/candles - Futures candles
    19. /derivatives/api/v4/markets/{symbol}/orderbook - Detailed orderbook
    20. /derivatives/api/v4/markets/{symbol}/trades - Recent trades
    21. /derivatives/api/v4/markets/{symbol}/ticker - Single ticker
    """
    
    SPOT_BASE_URL = "https://api.kraken.com"
    FUTURES_BASE_URL = "https://futures.kraken.com"
    
    def __init__(self, rate_limit: Optional[KrakenRateLimitConfig] = None):
        self.rate_limit = rate_limit or KrakenRateLimitConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0
        self._request_count = 0
        
        # Symbol mapping: standard -> Kraken spot format
        self._spot_symbol_map = {
            "BTC": "XBT",
            "DOGE": "XDG",
        }
        
        # Cache for static data
        self._assets_cache: Optional[Dict] = None
        self._pairs_cache: Optional[Dict] = None
        self._instruments_cache: Optional[List] = None
        self._cache_time = 0
        self._cache_ttl = 3600  # 1 hour
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """Close the client session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _rate_limit_wait(self):
        """Implement rate limiting"""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        
        if elapsed < 1.0:
            self._request_count += 1
            if self._request_count >= self.rate_limit.requests_per_second:
                await asyncio.sleep(1.0 - elapsed)
                self._request_count = 0
                self._last_request_time = time.time()
        else:
            self._request_count = 1
            self._last_request_time = current_time
    
    async def _request_spot(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Make request to Kraken Spot API"""
        await self._rate_limit_wait()
        
        session = await self._get_session()
        url = f"{self.SPOT_BASE_URL}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get("error") and len(data["error"]) > 0:
                    error_msg = ", ".join(data["error"])
                    logger.warning(f"Kraken Spot API error: {error_msg}")
                    return {"error": error_msg}
                
                return data.get("result", data)
                
        except aiohttp.ClientError as e:
            logger.error(f"Kraken Spot request error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Kraken Spot unexpected error: {e}")
            return {"error": str(e)}
    
    async def _request_futures(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None
    ) -> Any:
        """Make request to Kraken Futures API"""
        await self._rate_limit_wait()
        
        session = await self._get_session()
        url = f"{base_url or self.FUTURES_BASE_URL}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                # Futures API uses different error format
                if data.get("result") == "error":
                    error_msg = data.get("error", "Unknown error")
                    logger.warning(f"Kraken Futures API error: {error_msg}")
                    return {"error": error_msg}
                
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"Kraken Futures request error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Kraken Futures unexpected error: {e}")
            return {"error": str(e)}
    
    def _convert_symbol_to_kraken(self, symbol: str) -> str:
        """Convert standard symbol to Kraken format"""
        symbol = symbol.upper()
        for std, kraken in self._spot_symbol_map.items():
            symbol = symbol.replace(std, kraken)
        return symbol
    
    def _convert_symbol_from_kraken(self, symbol: str) -> str:
        """Convert Kraken symbol to standard format"""
        symbol = symbol.upper()
        for std, kraken in self._spot_symbol_map.items():
            symbol = symbol.replace(kraken, std)
        return symbol
    
    # ==================== SPOT API ENDPOINTS ====================
    
    async def get_server_time(self) -> Dict[str, Any]:
        """
        Get server time
        Endpoint: /0/public/Time
        """
        return await self._request_spot("/0/public/Time")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status
        Endpoint: /0/public/SystemStatus
        """
        return await self._request_spot("/0/public/SystemStatus")
    
    async def get_assets(self, assets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get asset information
        Endpoint: /0/public/Assets
        """
        # Check cache
        current_time = time.time()
        if self._assets_cache and (current_time - self._cache_time) < self._cache_ttl:
            if assets:
                return {k: v for k, v in self._assets_cache.items() if k in assets}
            return self._assets_cache
        
        params = {}
        if assets:
            params["asset"] = ",".join(assets)
        
        result = await self._request_spot("/0/public/Assets", params)
        
        if isinstance(result, dict) and "error" not in result:
            self._assets_cache = result
            self._cache_time = current_time
        
        return result
    
    async def get_asset_pairs(
        self,
        pairs: Optional[List[str]] = None,
        info: str = "info"  # info, leverage, fees, margin
    ) -> Dict[str, Any]:
        """
        Get tradable asset pairs
        Endpoint: /0/public/AssetPairs
        """
        # Check cache for full data
        current_time = time.time()
        if not pairs and self._pairs_cache and (current_time - self._cache_time) < self._cache_ttl:
            return self._pairs_cache
        
        params = {"info": info}
        if pairs:
            params["pair"] = ",".join(pairs)
        
        result = await self._request_spot("/0/public/AssetPairs", params)
        
        if isinstance(result, dict) and "error" not in result and not pairs:
            self._pairs_cache = result
            self._cache_time = current_time
        
        return result
    
    async def get_ticker(self, pairs: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Get ticker information for pairs
        Endpoint: /0/public/Ticker
        
        Returns for each pair:
        - a: ask [price, whole lot volume, lot volume]
        - b: bid [price, whole lot volume, lot volume]
        - c: last trade [price, lot volume]
        - v: volume [today, last 24h]
        - p: volume weighted average price [today, last 24h]
        - t: number of trades [today, last 24h]
        - l: low [today, last 24h]
        - h: high [today, last 24h]
        - o: opening price today
        """
        if isinstance(pairs, list):
            pair_str = ",".join(pairs)
        else:
            pair_str = pairs
        
        params = {"pair": pair_str}
        return await self._request_spot("/0/public/Ticker", params)
    
    async def get_ohlc(
        self,
        pair: str,
        interval: KrakenInterval = KrakenInterval.HOUR_1,
        since: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get OHLC/candlestick data
        Endpoint: /0/public/OHLC
        
        Returns: [time, open, high, low, close, vwap, volume, count]
        """
        params = {
            "pair": pair,
            "interval": interval.value
        }
        if since:
            params["since"] = since
        
        return await self._request_spot("/0/public/OHLC", params)
    
    async def get_orderbook(
        self,
        pair: str,
        count: int = 100  # Max 500
    ) -> Dict[str, Any]:
        """
        Get order book
        Endpoint: /0/public/Depth
        
        Returns: asks/bids as [price, volume, timestamp]
        """
        params = {
            "pair": pair,
            "count": min(count, 500)
        }
        return await self._request_spot("/0/public/Depth", params)
    
    async def get_trades(
        self,
        pair: str,
        since: Optional[str] = None,
        count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get recent trades
        Endpoint: /0/public/Trades
        
        Returns: [price, volume, time, buy/sell, market/limit, miscellaneous, trade_id]
        """
        params = {"pair": pair}
        if since:
            params["since"] = since
        if count:
            params["count"] = count
        
        return await self._request_spot("/0/public/Trades", params)
    
    async def get_spread(
        self,
        pair: str,
        since: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get recent spread data
        Endpoint: /0/public/Spread
        
        Returns: [time, bid, ask]
        """
        params = {"pair": pair}
        if since:
            params["since"] = since
        
        return await self._request_spot("/0/public/Spread", params)
    
    # ==================== FUTURES API ENDPOINTS ====================
    
    async def get_futures_instruments(self) -> List[Dict[str, Any]]:
        """
        Get all futures instruments
        Endpoint: /derivatives/api/v3/instruments
        """
        # Check cache
        current_time = time.time()
        if self._instruments_cache and (current_time - self._cache_time) < self._cache_ttl:
            return self._instruments_cache
        
        result = await self._request_futures("/derivatives/api/v3/instruments")
        
        if isinstance(result, dict) and "instruments" in result:
            self._instruments_cache = result["instruments"]
            self._cache_time = current_time
            return result["instruments"]
        
        return result if isinstance(result, list) else []
    
    async def get_futures_tickers(self) -> List[Dict[str, Any]]:
        """
        Get all futures tickers
        Endpoint: /derivatives/api/v3/tickers
        
        Returns ticker data including:
        - symbol, bid, ask, last, vol24h, openInterest, markPrice, etc.
        """
        result = await self._request_futures("/derivatives/api/v3/tickers")
        
        if isinstance(result, dict) and "tickers" in result:
            return result["tickers"]
        
        return result if isinstance(result, list) else []
    
    async def get_futures_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get single futures ticker
        """
        tickers = await self.get_futures_tickers()
        
        if isinstance(tickers, list):
            for ticker in tickers:
                if ticker.get("symbol", "").upper() == symbol.upper():
                    return ticker
        
        return {"error": f"Ticker not found for {symbol}"}
    
    async def get_futures_orderbook(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Get futures order book
        Endpoint: /derivatives/api/v3/orderbook
        """
        params = {"symbol": symbol}
        result = await self._request_futures("/derivatives/api/v3/orderbook", params)
        
        if isinstance(result, dict) and "orderBook" in result:
            return result["orderBook"]
        
        return result
    
    async def get_futures_trades(
        self,
        symbol: str,
        last_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get futures trade history
        Endpoint: /derivatives/api/v3/history
        """
        params = {"symbol": symbol}
        if last_time:
            params["lastTime"] = last_time
        
        result = await self._request_futures("/derivatives/api/v3/history", params)
        
        if isinstance(result, dict) and "history" in result:
            return result["history"]
        
        return result if isinstance(result, list) else []
    
    async def get_futures_candles(
        self,
        symbol: str,
        interval: KrakenFuturesInterval = KrakenFuturesInterval.HOUR_1,
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get futures candlestick data
        Endpoint: /api/charts/v1/trade/{symbol}/{interval}
        """
        endpoint = f"/api/charts/v1/trade/{symbol}/{interval.value}"
        params = {}
        if from_ts:
            params["from"] = from_ts
        if to_ts:
            params["to"] = to_ts
        
        result = await self._request_futures(endpoint, params)
        
        if isinstance(result, dict) and "candles" in result:
            return result["candles"]
        
        return result if isinstance(result, list) else []
    
    async def get_fee_schedules(self) -> Dict[str, Any]:
        """
        Get fee schedules and volumes
        Endpoint: /derivatives/api/v3/feeschedules/volumes
        """
        return await self._request_futures("/derivatives/api/v3/feeschedules/volumes")
    
    # ==================== DERIVED/COMPOSITE METHODS ====================
    
    async def get_spot_ticker_formatted(self, pair: str) -> Dict[str, Any]:
        """Get formatted spot ticker data"""
        result = await self.get_ticker(pair)
        
        if isinstance(result, dict) and "error" not in result:
            # Find the ticker in the result
            for key, data in result.items():
                if isinstance(data, dict):
                    return {
                        "symbol": self._convert_symbol_from_kraken(key),
                        "ask_price": float(data["a"][0]) if data.get("a") else 0,
                        "ask_volume": float(data["a"][2]) if data.get("a") and len(data["a"]) > 2 else 0,
                        "bid_price": float(data["b"][0]) if data.get("b") else 0,
                        "bid_volume": float(data["b"][2]) if data.get("b") and len(data["b"]) > 2 else 0,
                        "last_price": float(data["c"][0]) if data.get("c") else 0,
                        "last_volume": float(data["c"][1]) if data.get("c") and len(data["c"]) > 1 else 0,
                        "volume_today": float(data["v"][0]) if data.get("v") else 0,
                        "volume_24h": float(data["v"][1]) if data.get("v") and len(data["v"]) > 1 else 0,
                        "vwap_today": float(data["p"][0]) if data.get("p") else 0,
                        "vwap_24h": float(data["p"][1]) if data.get("p") and len(data["p"]) > 1 else 0,
                        "trades_today": int(data["t"][0]) if data.get("t") else 0,
                        "trades_24h": int(data["t"][1]) if data.get("t") and len(data["t"]) > 1 else 0,
                        "low_today": float(data["l"][0]) if data.get("l") else 0,
                        "low_24h": float(data["l"][1]) if data.get("l") and len(data["l"]) > 1 else 0,
                        "high_today": float(data["h"][0]) if data.get("h") else 0,
                        "high_24h": float(data["h"][1]) if data.get("h") and len(data["h"]) > 1 else 0,
                        "open_today": float(data["o"]) if data.get("o") else 0
                    }
        
        return result
    
    async def get_futures_ticker_formatted(self, symbol: str) -> Dict[str, Any]:
        """Get formatted futures ticker data"""
        result = await self.get_futures_ticker(symbol)
        
        if isinstance(result, dict) and "error" not in result:
            return {
                "symbol": result.get("symbol", symbol),
                "pair": result.get("pair", ""),
                "tag": result.get("tag", ""),
                "last_price": float(result.get("last", 0)),
                "bid_price": float(result.get("bid", 0)),
                "bid_size": float(result.get("bidSize", 0)),
                "ask_price": float(result.get("ask", 0)),
                "ask_size": float(result.get("askSize", 0)),
                "volume_24h": float(result.get("vol24h", 0)),
                "open_interest": float(result.get("openInterest", 0)),
                "mark_price": float(result.get("markPrice", 0)),
                "index_price": float(result.get("indexPrice", 0)),
                "funding_rate": float(result.get("fundingRate", 0)),
                "funding_rate_prediction": float(result.get("fundingRatePrediction", 0)),
                "change_24h": float(result.get("change24h", 0)),
                "premium": float(result.get("premium", 0)) if result.get("premium") else 0,
                "suspended": result.get("suspended", False),
                "post_only": result.get("postOnly", False)
            }
        
        return result
    
    async def get_market_snapshot(self, symbol: str = "BTC") -> Dict[str, Any]:
        """
        Get comprehensive market snapshot for a symbol
        Combines spot and futures data
        """
        # Prepare pair names
        spot_pair = f"{self._convert_symbol_to_kraken(symbol)}USD"
        futures_symbol = f"PF_{symbol.upper()}USD"
        
        try:
            # Fetch data in parallel
            spot_task = self.get_spot_ticker_formatted(spot_pair)
            futures_task = self.get_futures_ticker_formatted(futures_symbol)
            
            spot, futures = await asyncio.gather(
                spot_task, futures_task,
                return_exceptions=True
            )
            
            result = {
                "symbol": symbol.upper(),
                "timestamp": int(time.time() * 1000)
            }
            
            # Add spot data
            if isinstance(spot, dict) and "error" not in spot:
                result["spot"] = spot
            
            # Add futures data
            if isinstance(futures, dict) and "error" not in futures:
                result["perpetual"] = futures
                
                # Calculate basis
                if isinstance(spot, dict) and "error" not in spot:
                    spot_price = spot.get("last_price", 0)
                    futures_price = futures.get("last_price", 0)
                    if spot_price > 0 and futures_price > 0:
                        basis = (futures_price - spot_price) / spot_price * 100
                        result["analysis"] = {
                            "basis_pct": round(basis, 4),
                            "basis_annualized": round(basis * 365 / 8, 2),  # Assuming 8h funding
                            "funding_rate": futures.get("funding_rate", 0),
                            "funding_rate_pct": futures.get("funding_rate", 0) * 100
                        }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting Kraken market snapshot: {e}")
            return {"error": str(e)}
    
    async def get_all_perpetuals(self) -> List[Dict[str, Any]]:
        """Get all perpetual futures contracts"""
        tickers = await self.get_futures_tickers()
        
        if isinstance(tickers, list):
            perpetuals = []
            for t in tickers:
                symbol = t.get("symbol", "")
                # Filter for perpetuals (usually PF_ prefix or no expiry)
                if symbol.startswith("PF_") or symbol.startswith("PI_"):
                    perpetuals.append({
                        "symbol": symbol,
                        "pair": t.get("pair", ""),
                        "last_price": float(t.get("last", 0)),
                        "mark_price": float(t.get("markPrice", 0)),
                        "index_price": float(t.get("indexPrice", 0)),
                        "volume_24h": float(t.get("vol24h", 0)),
                        "open_interest": float(t.get("openInterest", 0)),
                        "funding_rate": float(t.get("fundingRate", 0)),
                        "funding_rate_pct": float(t.get("fundingRate", 0)) * 100,
                        "change_24h": float(t.get("change24h", 0)),
                        "change_24h_pct": float(t.get("change24h", 0)) * 100
                    })
            
            return sorted(perpetuals, key=lambda x: x["volume_24h"], reverse=True)
        
        return []
    
    async def get_funding_rates(self) -> List[Dict[str, Any]]:
        """Get funding rates for all perpetuals"""
        perpetuals = await self.get_all_perpetuals()
        
        funding_data = []
        for p in perpetuals:
            funding_data.append({
                "symbol": p["symbol"],
                "pair": p["pair"],
                "funding_rate": p["funding_rate"],
                "funding_rate_pct": p["funding_rate_pct"],
                "predicted_rate": 0,  # Not available in tickers
                "next_funding_time": None
            })
        
        # Sort by absolute funding rate
        return sorted(funding_data, key=lambda x: abs(x["funding_rate"]), reverse=True)
    
    async def get_open_interest_all(self) -> List[Dict[str, Any]]:
        """Get open interest for all perpetuals"""
        perpetuals = await self.get_all_perpetuals()
        
        oi_data = []
        for p in perpetuals:
            oi_data.append({
                "symbol": p["symbol"],
                "pair": p["pair"],
                "open_interest": p["open_interest"],
                "open_interest_usd": p["open_interest"] * p["mark_price"],
                "volume_24h": p["volume_24h"],
                "last_price": p["last_price"]
            })
        
        return sorted(oi_data, key=lambda x: x["open_interest_usd"], reverse=True)
    
    async def get_top_movers(self, limit: int = 10) -> Dict[str, Any]:
        """Get top gainers and losers from futures"""
        perpetuals = await self.get_all_perpetuals()
        
        if not perpetuals:
            return {"error": "No perpetual data available"}
        
        # Sort by change
        sorted_by_change = sorted(perpetuals, key=lambda x: x["change_24h"], reverse=True)
        
        gainers = sorted_by_change[:limit]
        losers = sorted_by_change[-limit:][::-1]
        
        return {
            "timestamp": int(time.time() * 1000),
            "top_gainers": [
                {
                    "symbol": g["symbol"],
                    "pair": g["pair"],
                    "last_price": g["last_price"],
                    "change_24h_pct": g["change_24h_pct"],
                    "volume_24h": g["volume_24h"]
                }
                for g in gainers if g["change_24h"] > 0
            ],
            "top_losers": [
                {
                    "symbol": l["symbol"],
                    "pair": l["pair"],
                    "last_price": l["last_price"],
                    "change_24h_pct": l["change_24h_pct"],
                    "volume_24h": l["volume_24h"]
                }
                for l in losers if l["change_24h"] < 0
            ]
        }
    
    async def get_full_analysis(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Get comprehensive analysis with trading signals"""
        
        try:
            # Get market snapshot
            snapshot = await self.get_market_snapshot(symbol)
            
            if isinstance(snapshot, dict) and "error" in snapshot:
                return snapshot
            
            result = {
                "symbol": symbol.upper(),
                "timestamp": int(time.time() * 1000),
                "snapshot": snapshot
            }
            
            signals = []
            analysis = {}
            
            # Analyze funding rate
            if "perpetual" in snapshot:
                perp = snapshot["perpetual"]
                funding_rate = perp.get("funding_rate", 0)
                
                if funding_rate > 0.0001:  # > 0.01%
                    signals.append("🔴 High positive funding (shorts getting paid)")
                    analysis["funding_signal"] = "BEARISH"
                elif funding_rate < -0.0001:  # < -0.01%
                    signals.append("🟢 Negative funding (longs getting paid)")
                    analysis["funding_signal"] = "BULLISH"
                else:
                    signals.append("⚪ Neutral funding")
                    analysis["funding_signal"] = "NEUTRAL"
            
            # Analyze basis
            if "analysis" in snapshot:
                basis = snapshot["analysis"].get("basis_pct", 0)
                
                if basis > 0.1:  # > 0.1% premium
                    signals.append("🟢 Futures at premium (bullish sentiment)")
                    analysis["basis_signal"] = "BULLISH"
                elif basis < -0.1:  # < -0.1% discount
                    signals.append("🔴 Futures at discount (bearish sentiment)")
                    analysis["basis_signal"] = "BEARISH"
                else:
                    signals.append("⚪ Neutral basis")
                    analysis["basis_signal"] = "NEUTRAL"
            
            # Analyze open interest
            if "perpetual" in snapshot:
                oi = snapshot["perpetual"].get("open_interest", 0)
                vol = snapshot["perpetual"].get("volume_24h", 0)
                
                if oi > 0 and vol > 0:
                    oi_to_vol = oi / vol if vol > 0 else 0
                    analysis["oi_to_volume_ratio"] = round(oi_to_vol, 2)
                    
                    if oi_to_vol > 2:
                        signals.append("🟡 High OI relative to volume (potential squeeze)")
                    elif oi_to_vol < 0.5:
                        signals.append("🟢 Low OI relative to volume (fresh positioning)")
            
            # Overall signal
            bullish = sum(1 for s in analysis.values() if s == "BULLISH")
            bearish = sum(1 for s in analysis.values() if s == "BEARISH")
            
            if bullish > bearish:
                overall = "🟢 BULLISH"
            elif bearish > bullish:
                overall = "🔴 BEARISH"
            else:
                overall = "⚪ NEUTRAL"
            
            result["analysis"] = {
                "signals": signals,
                "signal_breakdown": analysis,
                "overall_signal": {
                    "bullish_count": bullish,
                    "bearish_count": bearish,
                    "interpretation": overall
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting Kraken full analysis: {e}")
            return {"error": str(e)}


# Singleton instance
_kraken_client: Optional[KrakenRESTClient] = None


def get_kraken_rest_client() -> KrakenRESTClient:
    """Get singleton Kraken REST client"""
    global _kraken_client
    if _kraken_client is None:
        _kraken_client = KrakenRESTClient()
    return _kraken_client


async def close_kraken_rest_client():
    """Close the Kraken REST client"""
    global _kraken_client
    if _kraken_client:
        await _kraken_client.close()
        _kraken_client = None
