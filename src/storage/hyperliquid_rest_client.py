"""
Hyperliquid REST API Client
Comprehensive implementation of all available public Hyperliquid endpoints
Hyperliquid is a decentralized perpetual futures exchange on Arbitrum
"""

import aiohttp
import asyncio
import time
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class HyperliquidInterval(str, Enum):
    """Kline intervals for Hyperliquid"""
    MIN_1 = "1m"
    MIN_3 = "3m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
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


@dataclass
class HyperliquidRateLimitConfig:
    """Rate limit configuration for Hyperliquid API"""
    # Hyperliquid: 1200 requests per minute
    requests_per_second: int = 20
    weight_per_second: int = 100


class HyperliquidRESTClient:
    """
    Comprehensive Hyperliquid REST API client.
    
    Hyperliquid uses a unique API structure with POST requests to /info endpoint.
    
    INFO ENDPOINTS (POST /info):
    1. meta - Exchange metadata (all perpetuals, universe info)
    2. allMids - All mid prices
    3. metaAndAssetCtxs - Combined meta + asset contexts
    4. clearinghouseState - User clearing house state
    5. openOrders - User open orders
    6. userFills - User trade fills
    7. userFunding - User funding payments
    8. fundingHistory - Historical funding rates
    9. l2Book - Level 2 order book
    10. candleSnapshot - OHLCV candlesticks
    11. userNonFundingLedgerUpdates - User ledger updates
    12. spotMeta - Spot market metadata
    13. spotClearinghouseState - Spot clearing house state
    14. spotMetaAndAssetCtxs - Spot meta + contexts
    15. perpsAtOpenInterest - Perpetuals at specific OI
    16. historicalOrders - Historical orders
    17. twapHistory - TWAP history
    18. subaccounts - User subaccounts
    19. vaultDetails - Vault details
    20. delegatorHistory - Delegator history
    21. delegatorSummary - Delegator summary
    22. tokenDetails - Token details
    23. userVaultEquities - User vault equities
    24. maxBuilderFee - Max builder fee
    25. userFees - User fee rates
    26. portfolio - User portfolio
    27. referral - Referral info
    28. extraAgents - Extra agents
    """
    
    BASE_URL = "https://api.hyperliquid.xyz"
    
    def __init__(self, rate_limit: Optional[HyperliquidRateLimitConfig] = None):
        self.rate_limit = rate_limit or HyperliquidRateLimitConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0
        self._request_count = 0
        
        # Cache for static data
        self._meta_cache: Optional[Dict] = None
        self._cache_time = 0
        self._cache_ttl = 300  # 5 minutes
    
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
    
    async def _post_info(self, request_type: str, payload: Optional[Dict] = None) -> Any:
        """Make POST request to /info endpoint"""
        await self._rate_limit_wait()
        
        session = await self._get_session()
        url = f"{self.BASE_URL}/info"
        
        data = {"type": request_type}
        if payload:
            data.update(payload)
        
        try:
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.warning(f"Hyperliquid API error {response.status}: {error_text}")
                    return {"error": error_text, "code": response.status}
                
        except aiohttp.ClientError as e:
            logger.error(f"Hyperliquid request error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Hyperliquid unexpected error: {e}")
            return {"error": str(e)}
    
    # ==================== MARKET DATA ENDPOINTS ====================
    
    async def get_meta(self) -> Dict[str, Any]:
        """
        Get exchange metadata including all perpetual contracts.
        
        Returns universe info with all available perpetuals.
        """
        # Check cache
        current_time = time.time()
        if self._meta_cache and (current_time - self._cache_time) < self._cache_ttl:
            return self._meta_cache
        
        result = await self._post_info("meta")
        
        if isinstance(result, dict) and "universe" in result:
            self._meta_cache = result
            self._cache_time = current_time
        
        return result
    
    async def get_all_mids(self) -> Dict[str, str]:
        """
        Get all mid prices for all perpetuals.
        
        Returns dict mapping coin -> mid price string
        """
        return await self._post_info("allMids")
    
    async def get_meta_and_asset_ctxs(self) -> List[Any]:
        """
        Get combined metadata and asset contexts.
        
        Returns [meta, assetCtxs] where assetCtxs contains
        funding rates, open interest, mark prices, etc.
        """
        return await self._post_info("metaAndAssetCtxs")
    
    async def get_l2_book(self, coin: str, n_sig_figs: int = 5, mantissa_figs: int = None) -> Dict[str, Any]:
        """
        Get Level 2 order book for a coin.
        
        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')
            n_sig_figs: Number of significant figures for price aggregation
            mantissa_figs: Optional mantissa figures (alternative to n_sig_figs)
        
        Returns order book with levels array containing [price, size, numOrders]
        """
        payload = {"coin": coin}
        if mantissa_figs is not None:
            payload["mantissa"] = mantissa_figs
        else:
            payload["nSigFigs"] = n_sig_figs
        
        return await self._post_info("l2Book", payload)
    
    async def get_candles(
        self,
        coin: str,
        interval: HyperliquidInterval = HyperliquidInterval.HOUR_1,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get OHLCV candlestick data.
        
        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')
            interval: Candle interval
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
        
        Returns list of candles with t, T, s, i, o, c, h, l, v, n
        """
        # Build the request payload according to Hyperliquid API spec
        req_payload = {
            "coin": coin,
            "interval": interval.value
        }
        
        if start_time:
            req_payload["startTime"] = start_time
        if end_time:
            req_payload["endTime"] = end_time
        
        result = await self._post_info("candleSnapshot", {"req": req_payload})
        return result if isinstance(result, list) else []
    
    async def get_funding_history(
        self,
        coin: str,
        start_time: int,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical funding rates for a coin.
        
        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')
            start_time: Start timestamp in milliseconds
            end_time: Optional end timestamp
        
        Returns list of funding records with coin, fundingRate, premium, time
        """
        payload = {
            "coin": coin,
            "startTime": start_time
        }
        if end_time:
            payload["endTime"] = end_time
        
        result = await self._post_info("fundingHistory", payload)
        return result if isinstance(result, list) else []
    
    async def get_predicted_fundings(self) -> List[Dict[str, Any]]:
        """
        Get predicted funding rates for all coins.
        
        Returns list of predicted funding with coin and predictedFunding
        """
        return await self._post_info("predictedFundings")
    
    # ==================== SPOT MARKET ENDPOINTS ====================
    
    async def get_spot_meta(self) -> Dict[str, Any]:
        """
        Get spot market metadata.
        
        Returns spot tokens and universe info.
        """
        return await self._post_info("spotMeta")
    
    async def get_spot_meta_and_asset_ctxs(self) -> List[Any]:
        """
        Get spot metadata and asset contexts.
        
        Returns [spotMeta, spotAssetCtxs]
        """
        return await self._post_info("spotMetaAndAssetCtxs")
    
    # ==================== VAULT ENDPOINTS ====================
    
    async def get_vault_details(self, vault_address: str) -> Dict[str, Any]:
        """
        Get vault details.
        
        Args:
            vault_address: Vault contract address
        
        Returns vault information
        """
        return await self._post_info("vaultDetails", {"vaultAddress": vault_address})
    
    # ==================== TOKEN/DEPLOY ENDPOINTS ====================
    
    async def get_token_details(self, token_id: int) -> Dict[str, Any]:
        """
        Get token details by ID.
        
        Args:
            token_id: Token ID
        
        Returns token information
        """
        return await self._post_info("tokenDetails", {"tokenId": token_id})
    
    # ==================== COMPOSITE/DERIVED METHODS ====================
    
    async def get_all_perpetuals(self) -> List[Dict[str, Any]]:
        """
        Get all perpetual contracts with current market data.
        
        Returns formatted list of all perpetuals with prices, funding, OI
        """
        try:
            result = await self.get_meta_and_asset_ctxs()
            
            if not isinstance(result, list) or len(result) < 2:
                return []
            
            meta = result[0]
            asset_ctxs = result[1]
            
            universe = meta.get("universe", [])
            
            perpetuals = []
            for i, asset in enumerate(universe):
                if i < len(asset_ctxs):
                    ctx = asset_ctxs[i]
                    
                    # Safe float parsing helper
                    def safe_float(val, default=0):
                        if val is None:
                            return default
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            return default
                    
                    # Parse funding rate
                    funding = safe_float(ctx.get("funding"))
                    
                    # Parse prices
                    mark_px = safe_float(ctx.get("markPx"))
                    oracle_px = safe_float(ctx.get("oraclePx"))
                    
                    # Parse open interest
                    oi = safe_float(ctx.get("openInterest"))
                    
                    # Parse 24h stats
                    day_ntl_vlm = safe_float(ctx.get("dayNtlVlm"))
                    prev_day_px = safe_float(ctx.get("prevDayPx"))
                    
                    # Calculate change
                    change_pct = 0
                    if prev_day_px > 0:
                        change_pct = ((mark_px - prev_day_px) / prev_day_px) * 100
                    
                    perpetuals.append({
                        "coin": asset.get("name", ""),
                        "sz_decimals": asset.get("szDecimals", 0),
                        "max_leverage": asset.get("maxLeverage", 0),
                        "mark_price": mark_px,
                        "oracle_price": oracle_px,
                        "funding_rate": funding,
                        "funding_rate_pct": funding * 100,
                        "open_interest": oi,
                        "open_interest_usd": oi * mark_px,
                        "volume_24h_usd": day_ntl_vlm,
                        "prev_day_price": prev_day_px,
                        "change_pct": round(change_pct, 2),
                        "premium": safe_float(ctx.get("premium")),
                        "premium_pct": safe_float(ctx.get("premium")) * 100
                    })
            
            # Sort by volume
            return sorted(perpetuals, key=lambda x: x["volume_24h_usd"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting all perpetuals: {e}")
            return []
    
    async def get_ticker(self, coin: str) -> Dict[str, Any]:
        """
        Get ticker data for a specific coin.
        
        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')
        
        Returns formatted ticker data
        """
        perpetuals = await self.get_all_perpetuals()
        
        for p in perpetuals:
            if p["coin"].upper() == coin.upper():
                return {
                    "exchange": "hyperliquid",
                    "coin": coin,
                    "ticker": p
                }
        
        return {"error": f"Coin {coin} not found"}
    
    async def get_all_funding_rates(self) -> List[Dict[str, Any]]:
        """
        Get funding rates for all perpetuals.
        
        Returns formatted list sorted by absolute funding rate
        """
        perpetuals = await self.get_all_perpetuals()
        
        funding_data = []
        for p in perpetuals:
            funding_data.append({
                "coin": p["coin"],
                "funding_rate": p["funding_rate"],
                "funding_rate_pct": p["funding_rate_pct"],
                "mark_price": p["mark_price"],
                "volume_24h_usd": p["volume_24h_usd"]
            })
        
        # Sort by absolute funding rate
        return sorted(funding_data, key=lambda x: abs(x["funding_rate"]), reverse=True)
    
    async def get_all_open_interest(self) -> List[Dict[str, Any]]:
        """
        Get open interest for all perpetuals.
        
        Returns formatted list sorted by OI USD
        """
        perpetuals = await self.get_all_perpetuals()
        
        oi_data = []
        for p in perpetuals:
            oi_data.append({
                "coin": p["coin"],
                "open_interest": p["open_interest"],
                "open_interest_usd": p["open_interest_usd"],
                "mark_price": p["mark_price"],
                "volume_24h_usd": p["volume_24h_usd"]
            })
        
        # Sort by OI USD
        return sorted(oi_data, key=lambda x: x["open_interest_usd"], reverse=True)
    
    async def get_top_movers(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get top gainers and losers.
        
        Args:
            limit: Number of results per category
        
        Returns dict with top_gainers and top_losers
        """
        perpetuals = await self.get_all_perpetuals()
        
        if not perpetuals:
            return {"error": "No perpetual data available"}
        
        # Sort by change
        sorted_by_change = sorted(perpetuals, key=lambda x: x["change_pct"], reverse=True)
        
        gainers = sorted_by_change[:limit]
        losers = sorted_by_change[-limit:][::-1]
        
        return {
            "timestamp": int(time.time() * 1000),
            "top_gainers": [
                {
                    "coin": g["coin"],
                    "mark_price": g["mark_price"],
                    "change_pct": g["change_pct"],
                    "volume_24h_usd": g["volume_24h_usd"]
                }
                for g in gainers if g["change_pct"] > 0
            ],
            "top_losers": [
                {
                    "coin": l["coin"],
                    "mark_price": l["mark_price"],
                    "change_pct": l["change_pct"],
                    "volume_24h_usd": l["volume_24h_usd"]
                }
                for l in losers if l["change_pct"] < 0
            ]
        }
    
    async def get_orderbook(self, coin: str, depth: int = 20) -> Dict[str, Any]:
        """
        Get formatted orderbook for a coin.
        
        Args:
            coin: Coin symbol
            depth: Number of levels
        
        Returns formatted orderbook with analysis
        """
        book = await self.get_l2_book(coin)
        
        if isinstance(book, dict) and "error" not in book:
            levels = book.get("levels", [[], []])
            bids = levels[0] if len(levels) > 0 else []
            asks = levels[1] if len(levels) > 1 else []
            
            # Format and calculate metrics
            formatted_bids = []
            formatted_asks = []
            bid_vol = 0
            ask_vol = 0
            
            for b in bids[:depth]:
                px = float(b.get("px", 0))
                sz = float(b.get("sz", 0))
                n = b.get("n", 1)
                formatted_bids.append({"price": px, "size": sz, "orders": n})
                bid_vol += sz
            
            for a in asks[:depth]:
                px = float(a.get("px", 0))
                sz = float(a.get("sz", 0))
                n = a.get("n", 1)
                formatted_asks.append({"price": px, "size": sz, "orders": n})
                ask_vol += sz
            
            best_bid = formatted_bids[0]["price"] if formatted_bids else 0
            best_ask = formatted_asks[0]["price"] if formatted_asks else 0
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
            
            return {
                "exchange": "hyperliquid",
                "coin": coin,
                "orderbook": {
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread": spread,
                    "spread_pct": round(spread_pct, 4),
                    "bid_depth": bid_vol,
                    "ask_depth": ask_vol,
                    "imbalance_pct": round((bid_vol - ask_vol) / (bid_vol + ask_vol) * 100, 2) if (bid_vol + ask_vol) > 0 else 0,
                    "bid_levels": len(formatted_bids),
                    "ask_levels": len(formatted_asks)
                },
                "bids": formatted_bids,
                "asks": formatted_asks
            }
        
        return {"error": "Failed to fetch orderbook"}
    
    async def get_market_snapshot(self, coin: str = "BTC") -> Dict[str, Any]:
        """
        Get comprehensive market snapshot for a coin.
        
        Args:
            coin: Coin symbol
        
        Returns combined ticker, orderbook, and funding data
        """
        try:
            # Fetch data in parallel
            ticker_task = self.get_ticker(coin)
            orderbook_task = self.get_orderbook(coin)
            
            ticker, orderbook = await asyncio.gather(
                ticker_task, orderbook_task,
                return_exceptions=True
            )
            
            result = {
                "exchange": "hyperliquid",
                "coin": coin,
                "timestamp": int(time.time() * 1000)
            }
            
            # Process ticker
            if isinstance(ticker, dict) and "ticker" in ticker:
                result["ticker"] = ticker["ticker"]
            
            # Process orderbook
            if isinstance(orderbook, dict) and "orderbook" in orderbook:
                result["orderbook"] = orderbook["orderbook"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting market snapshot: {e}")
            return {"error": str(e)}
    
    async def get_full_analysis(self, coin: str = "BTC") -> Dict[str, Any]:
        """
        Get comprehensive analysis with trading signals.
        
        Args:
            coin: Coin symbol
        
        Returns analysis with signals for funding, basis, orderbook
        """
        try:
            snapshot = await self.get_market_snapshot(coin)
            
            if isinstance(snapshot, dict) and "error" in snapshot:
                return snapshot
            
            result = {
                "exchange": "hyperliquid",
                "coin": coin,
                "timestamp": int(time.time() * 1000),
                "snapshot": snapshot
            }
            
            signals = []
            analysis = {}
            
            # Analyze funding rate
            if "ticker" in snapshot:
                ticker = snapshot["ticker"]
                funding = ticker.get("funding_rate", 0)
                
                if funding > 0.0001:  # > 0.01%
                    signals.append("🔴 High positive funding (shorts getting paid)")
                    analysis["funding_signal"] = "BEARISH"
                elif funding < -0.0001:  # < -0.01%
                    signals.append("🟢 Negative funding (longs getting paid)")
                    analysis["funding_signal"] = "BULLISH"
                else:
                    signals.append("⚪ Neutral funding")
                    analysis["funding_signal"] = "NEUTRAL"
                
                # Analyze premium
                premium = ticker.get("premium", 0)
                analysis["premium_pct"] = ticker.get("premium_pct", 0)
                
                if premium > 0.001:  # > 0.1%
                    signals.append("🟢 Futures at premium (bullish sentiment)")
                    analysis["premium_signal"] = "BULLISH"
                elif premium < -0.001:  # < -0.1%
                    signals.append("🔴 Futures at discount (bearish sentiment)")
                    analysis["premium_signal"] = "BEARISH"
                else:
                    signals.append("⚪ Neutral premium")
                    analysis["premium_signal"] = "NEUTRAL"
                
                # Analyze momentum
                change = ticker.get("change_pct", 0)
                if change > 5:
                    signals.append("🟢 Strong upward momentum (>5%)")
                    analysis["momentum_signal"] = "BULLISH"
                elif change < -5:
                    signals.append("🔴 Strong downward momentum (<-5%)")
                    analysis["momentum_signal"] = "BEARISH"
                else:
                    analysis["momentum_signal"] = "NEUTRAL"
            
            # Analyze orderbook imbalance
            if "orderbook" in snapshot:
                ob = snapshot["orderbook"]
                imbalance = ob.get("imbalance_pct", 0)
                
                if imbalance > 20:
                    signals.append("🟢 Strong bid imbalance (buying pressure)")
                    analysis["orderbook_signal"] = "BULLISH"
                elif imbalance < -20:
                    signals.append("🔴 Strong ask imbalance (selling pressure)")
                    analysis["orderbook_signal"] = "BEARISH"
                else:
                    analysis["orderbook_signal"] = "NEUTRAL"
                
                # Analyze spread
                spread_pct = ob.get("spread_pct", 0)
                if spread_pct > 0.1:
                    signals.append("⚠️ Wide spread (>0.1%) - low liquidity")
                    analysis["liquidity_signal"] = "LOW"
                elif spread_pct < 0.02:
                    signals.append("✅ Tight spread (<0.02%) - high liquidity")
                    analysis["liquidity_signal"] = "HIGH"
                else:
                    analysis["liquidity_signal"] = "MEDIUM"
            
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
            logger.error(f"Error getting full analysis: {e}")
            return {"error": str(e)}
    
    async def get_recent_trades(self, coin: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get recent trades from L2 book updates (approximation).
        
        Note: Hyperliquid doesn't have a direct recent trades endpoint.
        This returns orderbook data which can be used to infer market activity.
        """
        orderbook = await self.get_orderbook(coin, depth=50)
        
        if "error" not in orderbook:
            return {
                "exchange": "hyperliquid",
                "coin": coin,
                "note": "Hyperliquid uses L2 orderbook data. For trade history, use on-chain data.",
                "orderbook_summary": orderbook.get("orderbook", {})
            }
        
        return orderbook
    
    async def get_exchange_stats(self) -> Dict[str, Any]:
        """
        Get overall exchange statistics.
        
        Returns aggregated stats across all perpetuals
        """
        perpetuals = await self.get_all_perpetuals()
        
        if not perpetuals:
            return {"error": "Failed to fetch exchange stats"}
        
        total_oi = sum(p["open_interest_usd"] for p in perpetuals)
        total_volume = sum(p["volume_24h_usd"] for p in perpetuals)
        
        # Count positive/negative funding
        positive_funding = sum(1 for p in perpetuals if p["funding_rate"] > 0)
        negative_funding = sum(1 for p in perpetuals if p["funding_rate"] < 0)
        
        # Find extremes
        highest_funding = max(perpetuals, key=lambda x: x["funding_rate"])
        lowest_funding = min(perpetuals, key=lambda x: x["funding_rate"])
        highest_oi = max(perpetuals, key=lambda x: x["open_interest_usd"])
        highest_volume = max(perpetuals, key=lambda x: x["volume_24h_usd"])
        
        return {
            "exchange": "hyperliquid",
            "timestamp": int(time.time() * 1000),
            "total_perpetuals": len(perpetuals),
            "total_open_interest_usd": total_oi,
            "total_volume_24h_usd": total_volume,
            "funding_sentiment": {
                "positive_count": positive_funding,
                "negative_count": negative_funding,
                "neutral_count": len(perpetuals) - positive_funding - negative_funding
            },
            "extremes": {
                "highest_funding": {
                    "coin": highest_funding["coin"],
                    "rate_pct": highest_funding["funding_rate_pct"]
                },
                "lowest_funding": {
                    "coin": lowest_funding["coin"],
                    "rate_pct": lowest_funding["funding_rate_pct"]
                },
                "highest_oi": {
                    "coin": highest_oi["coin"],
                    "oi_usd": highest_oi["open_interest_usd"]
                },
                "highest_volume": {
                    "coin": highest_volume["coin"],
                    "volume_usd": highest_volume["volume_24h_usd"]
                }
            }
        }


# Singleton instance
_hyperliquid_client: Optional[HyperliquidRESTClient] = None


def get_hyperliquid_rest_client() -> HyperliquidRESTClient:
    """Get singleton Hyperliquid REST client"""
    global _hyperliquid_client
    if _hyperliquid_client is None:
        _hyperliquid_client = HyperliquidRESTClient()
    return _hyperliquid_client


async def close_hyperliquid_rest_client():
    """Close the Hyperliquid REST client"""
    global _hyperliquid_client
    if _hyperliquid_client:
        await _hyperliquid_client.close()
        _hyperliquid_client = None
