"""
Gate.io REST API Client
Comprehensive implementation of all available public Gate.io Futures endpoints
Supports USDT-settled and BTC-settled perpetual futures, delivery futures, and options
"""

import aiohttp
import asyncio
import time
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class GateioSettle(str, Enum):
    """Settlement currency for Gate.io futures"""
    USDT = "usdt"
    BTC = "btc"


class GateioInterval(str, Enum):
    """Kline intervals for Gate.io"""
    SEC_10 = "10s"
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_8 = "8h"
    DAY_1 = "1d"
    WEEK_1 = "7d"
    MONTH_1 = "30d"


class GateioContractStatInterval(str, Enum):
    """Intervals for contract statistics"""
    MIN_5 = "5m"
    HOUR_1 = "1h"
    DAY_1 = "1d"


@dataclass
class GateioRateLimitConfig:
    """Rate limit configuration for Gate.io API"""
    # Gate.io: 900 requests per minute for public endpoints
    requests_per_second: int = 15
    weight_per_second: int = 100


class GateioRESTClient:
    """
    Comprehensive Gate.io REST API client.
    
    PERPETUAL FUTURES ENDPOINTS (/futures/{settle}/):
    1. contracts - List all futures contracts
    2. contracts/{contract} - Single contract info
    3. order_book - Order book depth
    4. trades - Recent trades
    5. candlesticks - OHLCV candlesticks
    6. tickers - All tickers
    7. funding_rate - Current funding rate
    8. insurance - Insurance fund balance
    9. contract_stats - Contract statistics (OI, volume, etc.)
    10. index_constituents/{index} - Index constituents
    11. liq_orders - Liquidation history
    12. risk_limit_tiers - Risk limit tiers
    
    DELIVERY FUTURES ENDPOINTS (/delivery/{settle}/):
    13. contracts - Delivery contracts
    14. order_book - Delivery order book
    15. trades - Delivery trades
    16. candlesticks - Delivery candlesticks
    17. tickers - Delivery tickers
    18. insurance - Delivery insurance fund
    
    OPTIONS ENDPOINTS (/options/):
    19. underlyings - Options underlyings
    20. expirations - Options expirations
    21. contracts - Options contracts
    22. settlements - Options settlements
    23. tickers - Options tickers
    24. underlying/tickers/{underlying} - Underlying ticker
    25. candlesticks - Options candlesticks
    26. order_book - Options order book
    27. trades - Options trades
    """
    
    BASE_URL = "https://api.gateio.ws/api/v4"
    
    def __init__(self, rate_limit: Optional[GateioRateLimitConfig] = None):
        self.rate_limit = rate_limit or GateioRateLimitConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0
        self._request_count = 0
        
        # Cache for static data
        self._contracts_cache: Dict[str, List] = {}
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
    
    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Make request to Gate.io API"""
        await self._rate_limit_wait()
        
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    # Rate limited - retry with backoff
                    retry_count = getattr(self, '_retry_count', 0)
                    if retry_count >= 3:
                        logger.error("Gate.io rate limit hit 3 times, giving up")
                        self._retry_count = 0
                        return {"error": "Rate limit exceeded", "code": 429}
                    self._retry_count = retry_count + 1
                    logger.warning(f"Gate.io rate limited, waiting 2s... (retry {self._retry_count}/3)")
                    await asyncio.sleep(2)
                    result = await self._request(endpoint, params)
                    self._retry_count = 0
                    return result
                else:
                    error_text = await response.text()
                    logger.warning(f"Gate.io API error {response.status}: {error_text}")
                    return {"error": error_text, "code": response.status}
                
        except aiohttp.ClientError as e:
            logger.error(f"Gate.io request error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Gate.io unexpected error: {e}")
            return {"error": str(e)}
    
    # ==================== PERPETUAL FUTURES ENDPOINTS ====================
    
    async def get_futures_contracts(
        self,
        settle: GateioSettle = GateioSettle.USDT,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all futures contracts
        Endpoint: /futures/{settle}/contracts
        """
        # Check cache
        cache_key = settle.value
        current_time = time.time()
        if cache_key in self._contracts_cache and (current_time - self._cache_time) < self._cache_ttl:
            return self._contracts_cache[cache_key]
        
        params = {"limit": limit, "offset": offset}
        result = await self._request(f"/futures/{settle.value}/contracts", params)
        
        if isinstance(result, list):
            self._contracts_cache[cache_key] = result
            self._cache_time = current_time
        
        return result if isinstance(result, list) else []
    
    async def get_futures_contract(
        self,
        contract: str,
        settle: GateioSettle = GateioSettle.USDT
    ) -> Dict[str, Any]:
        """
        Get single futures contract info
        Endpoint: /futures/{settle}/contracts/{contract}
        """
        return await self._request(f"/futures/{settle.value}/contracts/{contract}")
    
    async def get_futures_orderbook(
        self,
        contract: str,
        settle: GateioSettle = GateioSettle.USDT,
        interval: str = "0",  # 0 = no aggregation
        limit: int = 50,  # Max 50
        with_id: bool = False
    ) -> Dict[str, Any]:
        """
        Get futures order book
        Endpoint: /futures/{settle}/order_book
        """
        params = {
            "contract": contract,
            "interval": interval,
            "limit": min(limit, 50),
            "with_id": str(with_id).lower()
        }
        return await self._request(f"/futures/{settle.value}/order_book", params)
    
    async def get_futures_trades(
        self,
        contract: str,
        settle: GateioSettle = GateioSettle.USDT,
        limit: int = 100,
        offset: int = 0,
        last_id: Optional[str] = None,
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get futures recent trades
        Endpoint: /futures/{settle}/trades
        """
        params = {
            "contract": contract,
            "limit": min(limit, 1000)
        }
        if offset:
            params["offset"] = offset
        if last_id:
            params["last_id"] = last_id
        if from_ts:
            params["from"] = from_ts
        if to_ts:
            params["to"] = to_ts
        
        result = await self._request(f"/futures/{settle.value}/trades", params)
        return result if isinstance(result, list) else []
    
    async def get_futures_candlesticks(
        self,
        contract: str,
        settle: GateioSettle = GateioSettle.USDT,
        interval: GateioInterval = GateioInterval.HOUR_1,
        limit: int = 100,
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get futures candlesticks/OHLCV
        Endpoint: /futures/{settle}/candlesticks
        
        Returns: {t, v, c, h, l, o, sum} for each candle
        """
        params = {
            "contract": contract,
            "interval": interval.value,
            "limit": min(limit, 2000)
        }
        if from_ts:
            params["from"] = from_ts
        if to_ts:
            params["to"] = to_ts
        
        result = await self._request(f"/futures/{settle.value}/candlesticks", params)
        return result if isinstance(result, list) else []
    
    async def get_futures_tickers(
        self,
        settle: GateioSettle = GateioSettle.USDT,
        contract: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get futures tickers
        Endpoint: /futures/{settle}/tickers
        """
        params = {}
        if contract:
            params["contract"] = contract
        
        result = await self._request(f"/futures/{settle.value}/tickers", params)
        return result if isinstance(result, list) else []
    
    async def get_futures_ticker(
        self,
        contract: str,
        settle: GateioSettle = GateioSettle.USDT
    ) -> Dict[str, Any]:
        """Get single futures ticker"""
        tickers = await self.get_futures_tickers(settle, contract)
        if isinstance(tickers, list) and tickers:
            return tickers[0]
        return {"error": f"Ticker not found for {contract}"}
    
    async def get_funding_rate(
        self,
        contract: str,
        settle: GateioSettle = GateioSettle.USDT,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get funding rate history
        Endpoint: /futures/{settle}/funding_rate
        """
        params = {
            "contract": contract,
            "limit": min(limit, 1000)
        }
        result = await self._request(f"/futures/{settle.value}/funding_rate", params)
        return result if isinstance(result, list) else []
    
    async def get_insurance_fund(
        self,
        settle: GateioSettle = GateioSettle.USDT,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get insurance fund balance history
        Endpoint: /futures/{settle}/insurance
        """
        params = {"limit": min(limit, 100)}
        result = await self._request(f"/futures/{settle.value}/insurance", params)
        return result if isinstance(result, list) else []
    
    async def get_contract_stats(
        self,
        contract: str,
        settle: GateioSettle = GateioSettle.USDT,
        interval: GateioContractStatInterval = GateioContractStatInterval.HOUR_1,
        limit: int = 100,
        from_ts: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get contract statistics (OI, volume, liquidations, etc.)
        Endpoint: /futures/{settle}/contract_stats
        
        Returns: {time, lsr_taker, lsr_account, long_liq_size, long_liq_amount,
                  long_liq_usd, short_liq_size, short_liq_amount, short_liq_usd,
                  open_interest, open_interest_usd, top_lsr_account, top_lsr_size}
        """
        params = {
            "contract": contract,
            "interval": interval.value,
            "limit": min(limit, 100)
        }
        if from_ts:
            params["from"] = from_ts
        
        result = await self._request(f"/futures/{settle.value}/contract_stats", params)
        return result if isinstance(result, list) else []
    
    async def get_index_constituents(
        self,
        index: str,
        settle: GateioSettle = GateioSettle.USDT
    ) -> Dict[str, Any]:
        """
        Get index constituents
        Endpoint: /futures/{settle}/index_constituents/{index}
        """
        return await self._request(f"/futures/{settle.value}/index_constituents/{index}")
    
    async def get_liquidation_history(
        self,
        settle: GateioSettle = GateioSettle.USDT,
        contract: Optional[str] = None,
        limit: int = 100,
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get liquidation history
        Endpoint: /futures/{settle}/liq_orders
        """
        params = {"limit": min(limit, 1000)}
        if contract:
            params["contract"] = contract
        if from_ts:
            params["from"] = from_ts
        if to_ts:
            params["to"] = to_ts
        
        result = await self._request(f"/futures/{settle.value}/liq_orders", params)
        return result if isinstance(result, list) else []
    
    async def get_risk_limit_tiers(
        self,
        contract: str,
        settle: GateioSettle = GateioSettle.USDT
    ) -> List[Dict[str, Any]]:
        """
        Get risk limit tiers
        Endpoint: /futures/{settle}/risk_limit_tiers
        """
        params = {"contract": contract}
        result = await self._request(f"/futures/{settle.value}/risk_limit_tiers", params)
        return result if isinstance(result, list) else []
    
    # ==================== DELIVERY FUTURES ENDPOINTS ====================
    
    async def get_delivery_contracts(
        self,
        settle: GateioSettle = GateioSettle.USDT
    ) -> List[Dict[str, Any]]:
        """
        Get all delivery contracts
        Endpoint: /delivery/{settle}/contracts
        """
        result = await self._request(f"/delivery/{settle.value}/contracts")
        return result if isinstance(result, list) else []
    
    async def get_delivery_orderbook(
        self,
        contract: str,
        settle: GateioSettle = GateioSettle.USDT,
        interval: str = "0",
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get delivery futures order book
        Endpoint: /delivery/{settle}/order_book
        """
        params = {
            "contract": contract,
            "interval": interval,
            "limit": min(limit, 50)
        }
        return await self._request(f"/delivery/{settle.value}/order_book", params)
    
    async def get_delivery_trades(
        self,
        contract: str,
        settle: GateioSettle = GateioSettle.USDT,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get delivery futures trades
        Endpoint: /delivery/{settle}/trades
        """
        params = {
            "contract": contract,
            "limit": min(limit, 1000)
        }
        result = await self._request(f"/delivery/{settle.value}/trades", params)
        return result if isinstance(result, list) else []
    
    async def get_delivery_candlesticks(
        self,
        contract: str,
        settle: GateioSettle = GateioSettle.USDT,
        interval: GateioInterval = GateioInterval.HOUR_1,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get delivery futures candlesticks
        Endpoint: /delivery/{settle}/candlesticks
        """
        params = {
            "contract": contract,
            "interval": interval.value,
            "limit": min(limit, 2000)
        }
        result = await self._request(f"/delivery/{settle.value}/candlesticks", params)
        return result if isinstance(result, list) else []
    
    async def get_delivery_tickers(
        self,
        settle: GateioSettle = GateioSettle.USDT,
        contract: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get delivery futures tickers
        Endpoint: /delivery/{settle}/tickers
        """
        params = {}
        if contract:
            params["contract"] = contract
        
        result = await self._request(f"/delivery/{settle.value}/tickers", params)
        return result if isinstance(result, list) else []
    
    async def get_delivery_insurance(
        self,
        settle: GateioSettle = GateioSettle.USDT,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get delivery futures insurance fund
        Endpoint: /delivery/{settle}/insurance
        """
        params = {"limit": min(limit, 100)}
        result = await self._request(f"/delivery/{settle.value}/insurance", params)
        return result if isinstance(result, list) else []
    
    # ==================== OPTIONS ENDPOINTS ====================
    
    async def get_options_underlyings(self) -> List[Dict[str, Any]]:
        """
        Get options underlying assets
        Endpoint: /options/underlyings
        """
        result = await self._request("/options/underlyings")
        return result if isinstance(result, list) else []
    
    async def get_options_expirations(
        self,
        underlying: str
    ) -> List[int]:
        """
        Get options expiration dates
        Endpoint: /options/expirations
        """
        params = {"underlying": underlying}
        result = await self._request("/options/expirations", params)
        return result if isinstance(result, list) else []
    
    async def get_options_contracts(
        self,
        underlying: str,
        expiration: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get options contracts
        Endpoint: /options/contracts
        """
        params = {"underlying": underlying}
        if expiration:
            params["expiration"] = expiration
        
        result = await self._request("/options/contracts", params)
        return result if isinstance(result, list) else []
    
    async def get_options_settlements(
        self,
        underlying: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get options settlement history
        Endpoint: /options/settlements
        """
        params = {
            "underlying": underlying,
            "limit": min(limit, 100)
        }
        result = await self._request("/options/settlements", params)
        return result if isinstance(result, list) else []
    
    async def get_options_tickers(
        self,
        underlying: str
    ) -> List[Dict[str, Any]]:
        """
        Get options tickers
        Endpoint: /options/tickers
        """
        params = {"underlying": underlying}
        result = await self._request("/options/tickers", params)
        return result if isinstance(result, list) else []
    
    async def get_options_underlying_ticker(
        self,
        underlying: str
    ) -> Dict[str, Any]:
        """
        Get underlying ticker for options
        Endpoint: /options/underlying/tickers/{underlying}
        """
        return await self._request(f"/options/underlying/tickers/{underlying}")
    
    async def get_options_candlesticks(
        self,
        contract: str,
        interval: GateioInterval = GateioInterval.HOUR_1,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get options candlesticks
        Endpoint: /options/candlesticks
        """
        params = {
            "contract": contract,
            "interval": interval.value,
            "limit": min(limit, 1000)
        }
        result = await self._request("/options/candlesticks", params)
        return result if isinstance(result, list) else []
    
    async def get_options_orderbook(
        self,
        contract: str,
        interval: str = "0",
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get options order book
        Endpoint: /options/order_book
        """
        params = {
            "contract": contract,
            "interval": interval,
            "limit": min(limit, 50)
        }
        return await self._request("/options/order_book", params)
    
    async def get_options_trades(
        self,
        contract: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get options trades
        Endpoint: /options/trades
        """
        params = {
            "contract": contract,
            "limit": min(limit, 1000)
        }
        result = await self._request("/options/trades", params)
        return result if isinstance(result, list) else []
    
    # ==================== COMPOSITE/DERIVED METHODS ====================
    
    async def get_market_snapshot(
        self,
        symbol: str = "BTC",
        settle: GateioSettle = GateioSettle.USDT
    ) -> Dict[str, Any]:
        """
        Get comprehensive market snapshot for a symbol
        Combines ticker, funding, and contract stats
        """
        contract = f"{symbol}_USDT"
        
        try:
            # Fetch data in parallel
            ticker_task = self.get_futures_ticker(contract, settle)
            funding_task = self.get_funding_rate(contract, settle, limit=1)
            stats_task = self.get_contract_stats(contract, settle, limit=1)
            orderbook_task = self.get_futures_orderbook(contract, settle, limit=20)
            
            ticker, funding, stats, orderbook = await asyncio.gather(
                ticker_task, funding_task, stats_task, orderbook_task,
                return_exceptions=True
            )
            
            result = {
                "symbol": symbol,
                "contract": contract,
                "settle": settle.value,
                "timestamp": int(time.time() * 1000)
            }
            
            # Process ticker
            if isinstance(ticker, dict) and "error" not in ticker:
                result["ticker"] = {
                    "last_price": float(ticker.get("last", 0)),
                    "mark_price": float(ticker.get("mark_price", 0)),
                    "index_price": float(ticker.get("index_price", 0)),
                    "funding_rate": float(ticker.get("funding_rate", 0)),
                    "funding_rate_pct": float(ticker.get("funding_rate", 0)) * 100,
                    "volume_24h": float(ticker.get("volume_24h", 0)),
                    "volume_24h_usd": float(ticker.get("volume_24h_usd", 0)),
                    "volume_24h_btc": float(ticker.get("volume_24h_btc", 0)),
                    "change_pct": float(ticker.get("change_percentage", 0)),
                    "high_24h": float(ticker.get("high_24h", 0)),
                    "low_24h": float(ticker.get("low_24h", 0)),
                    "total_size": float(ticker.get("total_size", 0)),
                    "quanto_base_rate": ticker.get("quanto_base_rate")
                }
            
            # Process funding
            if isinstance(funding, list) and funding:
                latest = funding[0]
                result["funding"] = {
                    "rate": float(latest.get("r", 0)),
                    "rate_pct": float(latest.get("r", 0)) * 100,
                    "timestamp": latest.get("t")
                }
            
            # Process contract stats
            if isinstance(stats, list) and stats:
                latest = stats[0]
                result["stats"] = {
                    "open_interest": float(latest.get("open_interest", 0)),
                    "open_interest_usd": float(latest.get("open_interest_usd", 0)),
                    "long_short_taker": float(latest.get("lsr_taker", 0)),
                    "long_short_account": float(latest.get("lsr_account", 0)),
                    "top_trader_long_short_account": float(latest.get("top_lsr_account", 0)),
                    "top_trader_long_short_size": float(latest.get("top_lsr_size", 0)),
                    "long_liq_size": float(latest.get("long_liq_size", 0)),
                    "short_liq_size": float(latest.get("short_liq_size", 0)),
                    "long_liq_usd": float(latest.get("long_liq_usd", 0)),
                    "short_liq_usd": float(latest.get("short_liq_usd", 0))
                }
            
            # Process orderbook
            if isinstance(orderbook, dict) and "asks" in orderbook:
                asks = orderbook.get("asks", [])
                bids = orderbook.get("bids", [])
                
                bid_vol = sum(float(b.get("s", 0)) for b in bids[:10]) if bids else 0
                ask_vol = sum(float(a.get("s", 0)) for a in asks[:10]) if asks else 0
                
                best_bid = float(bids[0].get("p", 0)) if bids else 0
                best_ask = float(asks[0].get("p", 0)) if asks else 0
                spread = best_ask - best_bid
                spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
                
                result["orderbook"] = {
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread": spread,
                    "spread_pct": round(spread_pct, 4),
                    "bid_depth_10": bid_vol,
                    "ask_depth_10": ask_vol,
                    "imbalance_pct": round((bid_vol - ask_vol) / (bid_vol + ask_vol) * 100, 2) if (bid_vol + ask_vol) > 0 else 0
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting Gate.io market snapshot: {e}")
            return {"error": str(e)}
    
    async def get_all_perpetuals(
        self,
        settle: GateioSettle = GateioSettle.USDT
    ) -> List[Dict[str, Any]]:
        """Get all perpetual futures with formatted data"""
        tickers = await self.get_futures_tickers(settle)
        
        if not isinstance(tickers, list):
            return []
        
        perpetuals = []
        for t in tickers:
            contract = t.get("contract", "")
            # Filter perpetuals (exclude delivery futures)
            if "_" in contract and not any(x in contract for x in ["_QUARTER", "_BI"]):
                perpetuals.append({
                    "contract": contract,
                    "last_price": float(t.get("last", 0)),
                    "mark_price": float(t.get("mark_price", 0)),
                    "index_price": float(t.get("index_price", 0)),
                    "funding_rate": float(t.get("funding_rate", 0)),
                    "funding_rate_pct": float(t.get("funding_rate", 0)) * 100,
                    "volume_24h": float(t.get("volume_24h", 0)),
                    "volume_24h_usd": float(t.get("volume_24h_usd", 0)),
                    "change_pct": float(t.get("change_percentage", 0)),
                    "high_24h": float(t.get("high_24h", 0)),
                    "low_24h": float(t.get("low_24h", 0)),
                    "total_size": float(t.get("total_size", 0))
                })
        
        # Sort by volume
        return sorted(perpetuals, key=lambda x: x["volume_24h_usd"], reverse=True)
    
    async def get_funding_rates_all(
        self,
        settle: GateioSettle = GateioSettle.USDT
    ) -> List[Dict[str, Any]]:
        """Get funding rates for all perpetuals"""
        perpetuals = await self.get_all_perpetuals(settle)
        
        funding_data = []
        for p in perpetuals:
            funding_data.append({
                "contract": p["contract"],
                "funding_rate": p["funding_rate"],
                "funding_rate_pct": p["funding_rate_pct"],
                "last_price": p["last_price"],
                "volume_24h_usd": p["volume_24h_usd"]
            })
        
        # Sort by absolute funding rate
        return sorted(funding_data, key=lambda x: abs(x["funding_rate"]), reverse=True)
    
    async def get_open_interest_all(
        self,
        settle: GateioSettle = GateioSettle.USDT
    ) -> List[Dict[str, Any]]:
        """Get open interest for top contracts"""
        perpetuals = await self.get_all_perpetuals(settle)
        
        # Get stats for top contracts
        oi_data = []
        for p in perpetuals[:20]:  # Top 20 by volume
            try:
                stats = await self.get_contract_stats(p["contract"], settle, limit=1)
                if isinstance(stats, list) and stats:
                    latest = stats[0]
                    oi_data.append({
                        "contract": p["contract"],
                        "open_interest": float(latest.get("open_interest", 0)),
                        "open_interest_usd": float(latest.get("open_interest_usd", 0)),
                        "last_price": p["last_price"],
                        "volume_24h_usd": p["volume_24h_usd"]
                    })
            except Exception as e:
                logger.warning(f"Error getting OI for {p['contract']}: {e}")
        
        return sorted(oi_data, key=lambda x: x["open_interest_usd"], reverse=True)
    
    async def get_top_movers(
        self,
        settle: GateioSettle = GateioSettle.USDT,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Get top gainers and losers"""
        perpetuals = await self.get_all_perpetuals(settle)
        
        if not perpetuals:
            return {"error": "No perpetual data available"}
        
        # Sort by change percentage
        sorted_by_change = sorted(perpetuals, key=lambda x: x["change_pct"], reverse=True)
        
        gainers = sorted_by_change[:limit]
        losers = sorted_by_change[-limit:][::-1]
        
        return {
            "timestamp": int(time.time() * 1000),
            "settle": settle.value,
            "top_gainers": [
                {
                    "contract": g["contract"],
                    "last_price": g["last_price"],
                    "change_pct": g["change_pct"],
                    "volume_24h_usd": g["volume_24h_usd"]
                }
                for g in gainers if g["change_pct"] > 0
            ],
            "top_losers": [
                {
                    "contract": l["contract"],
                    "last_price": l["last_price"],
                    "change_pct": l["change_pct"],
                    "volume_24h_usd": l["volume_24h_usd"]
                }
                for l in losers if l["change_pct"] < 0
            ]
        }
    
    async def get_full_analysis(
        self,
        symbol: str = "BTC",
        settle: GateioSettle = GateioSettle.USDT
    ) -> Dict[str, Any]:
        """Get comprehensive analysis with trading signals"""
        contract = f"{symbol}_USDT"
        
        try:
            # Get snapshot
            snapshot = await self.get_market_snapshot(symbol, settle)
            
            if isinstance(snapshot, dict) and "error" in snapshot:
                return snapshot
            
            result = {
                "symbol": symbol,
                "contract": contract,
                "settle": settle.value,
                "timestamp": int(time.time() * 1000),
                "snapshot": snapshot
            }
            
            signals = []
            analysis = {}
            
            # Analyze funding rate
            if "ticker" in snapshot:
                funding = snapshot["ticker"].get("funding_rate", 0)
                
                if funding > 0.0001:  # > 0.01%
                    signals.append("🔴 High positive funding (shorts getting paid)")
                    analysis["funding_signal"] = "BEARISH"
                elif funding < -0.0001:  # < -0.01%
                    signals.append("🟢 Negative funding (longs getting paid)")
                    analysis["funding_signal"] = "BULLISH"
                else:
                    signals.append("⚪ Neutral funding")
                    analysis["funding_signal"] = "NEUTRAL"
            
            # Analyze basis (mark vs index)
            if "ticker" in snapshot:
                mark = snapshot["ticker"].get("mark_price", 0)
                index = snapshot["ticker"].get("index_price", 0)
                
                if mark > 0 and index > 0:
                    basis = (mark - index) / index * 100
                    analysis["basis_pct"] = round(basis, 4)
                    
                    if basis > 0.1:
                        signals.append("🟢 Futures at premium (bullish sentiment)")
                        analysis["basis_signal"] = "BULLISH"
                    elif basis < -0.1:
                        signals.append("🔴 Futures at discount (bearish sentiment)")
                        analysis["basis_signal"] = "BEARISH"
                    else:
                        signals.append("⚪ Neutral basis")
                        analysis["basis_signal"] = "NEUTRAL"
            
            # Analyze long/short ratio
            if "stats" in snapshot:
                lsr_taker = snapshot["stats"].get("long_short_taker", 1)
                lsr_account = snapshot["stats"].get("long_short_account", 1)
                
                analysis["lsr_taker"] = lsr_taker
                analysis["lsr_account"] = lsr_account
                
                if lsr_taker > 1.2:
                    signals.append("🟢 Takers bullish (L/S > 1.2)")
                elif lsr_taker < 0.8:
                    signals.append("🔴 Takers bearish (L/S < 0.8)")
                
                # Top traders sentiment
                top_lsr = snapshot["stats"].get("top_trader_long_short_account", 1)
                if top_lsr > 1.5:
                    signals.append("🟢 Top traders bullish")
                    analysis["top_traders_signal"] = "BULLISH"
                elif top_lsr < 0.7:
                    signals.append("🔴 Top traders bearish")
                    analysis["top_traders_signal"] = "BEARISH"
            
            # Analyze orderbook imbalance
            if "orderbook" in snapshot:
                imbalance = snapshot["orderbook"].get("imbalance_pct", 0)
                
                if imbalance > 20:
                    signals.append("🟢 Strong bid imbalance (buying pressure)")
                    analysis["orderbook_signal"] = "BULLISH"
                elif imbalance < -20:
                    signals.append("🔴 Strong ask imbalance (selling pressure)")
                    analysis["orderbook_signal"] = "BEARISH"
                else:
                    analysis["orderbook_signal"] = "NEUTRAL"
            
            # Analyze liquidations
            if "stats" in snapshot:
                long_liq = snapshot["stats"].get("long_liq_usd", 0)
                short_liq = snapshot["stats"].get("short_liq_usd", 0)
                
                if long_liq > short_liq * 2:
                    signals.append("🔴 Heavy long liquidations")
                    analysis["liquidation_signal"] = "BEARISH"
                elif short_liq > long_liq * 2:
                    signals.append("🟢 Heavy short liquidations")
                    analysis["liquidation_signal"] = "BULLISH"
            
            # Overall signal
            bullish = sum(1 for v in analysis.values() if v == "BULLISH")
            bearish = sum(1 for v in analysis.values() if v == "BEARISH")
            
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
            logger.error(f"Error getting Gate.io full analysis: {e}")
            return {"error": str(e)}


# Singleton instance
_gateio_client: Optional[GateioRESTClient] = None


def get_gateio_rest_client() -> GateioRESTClient:
    """Get singleton Gate.io REST client"""
    global _gateio_client
    if _gateio_client is None:
        _gateio_client = GateioRESTClient()
    return _gateio_client


async def close_gateio_rest_client():
    """Close the Gate.io REST client"""
    global _gateio_client
    if _gateio_client:
        await _gateio_client.close()
        _gateio_client = None
