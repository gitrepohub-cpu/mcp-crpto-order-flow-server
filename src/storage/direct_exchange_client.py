"""
Direct exchange WebSocket client - connects directly to crypto exchanges.
No dependency on the Go arbitrage scanner backend.
Suitable for cloud deployment on FastMCP.
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional
from datetime import datetime

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:
    raise ImportError("websockets package required. Install with: pip install websockets")

logger = logging.getLogger(__name__)


class DirectExchangeClient:
    """
    Connects directly to crypto exchanges without needing the Go backend.
    Suitable for cloud deployment.
    
    Supported Exchanges (matching Go Crypto Arbitrage Scanner):
    - Binance Futures (wss://fstream.binance.com)
    - Binance Spot (wss://stream.binance.com)
    - Bybit Futures (wss://stream.bybit.com)
    - Bybit Spot (wss://stream.bybit.com)
    - OKX Futures (wss://ws.okx.com)
    - Kraken Futures (wss://futures.kraken.com)
    - Gate.io Futures (wss://fx-ws.gateio.ws)
    - Hyperliquid (wss://api.hyperliquid.xyz)
    - Pyth Oracle (wss://hermes.pyth.network)
    
    Supported Symbols:
    - BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT
    """
    
    SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    
    # Exchange display names (matching Go scanner)
    EXCHANGE_NAMES = {
        "binance_futures": "Binance Futures",
        "binance_spot": "Binance Spot",
        "bybit_futures": "Bybit Futures",
        "bybit_spot": "Bybit Spot",
        "okx_futures": "OKX Futures",
        "kraken_futures": "Kraken Futures",
        "gate_futures": "Gate.io Futures",
        "hyperliquid_futures": "Hyperliquid",
        "pyth": "Pyth Oracle",
    }
    
    def __init__(self):
        # ====================================================================
        # DATA STORES - Comprehensive market data
        # ====================================================================
        
        # Price data (existing)
        self.prices: Dict[str, Dict[str, Dict]] = {}  # symbol -> exchange -> {price, bid, ask, timestamp}
        
        # Order Book Depth - Top levels per exchange
        self.orderbooks: Dict[str, Dict[str, Dict]] = {}  # symbol -> exchange -> {bids: [], asks: [], timestamp}
        
        # Trade Stream - Recent trades
        self.trades: Dict[str, Dict[str, List]] = {}  # symbol -> exchange -> [trades]
        self.max_trades_per_symbol = 100
        
        # Funding Rates (Futures only)
        self.funding_rates: Dict[str, Dict[str, Dict]] = {}  # symbol -> exchange -> {rate, next_time, timestamp}
        
        # Mark Price & Index Price
        self.mark_prices: Dict[str, Dict[str, Dict]] = {}  # symbol -> exchange -> {mark, index, timestamp}
        
        # Liquidations
        self.liquidations: Dict[str, List] = {}  # symbol -> [liquidation events]
        self.max_liquidations = 50
        
        # Open Interest
        self.open_interest: Dict[str, Dict[str, Dict]] = {}  # symbol -> exchange -> {oi, oi_value, timestamp}
        
        # 24h Statistics
        self.ticker_24h: Dict[str, Dict[str, Dict]] = {}  # symbol -> exchange -> {volume, high, low, change_pct}
        
        # ====================================================================
        # NEW: OHLCV Candles - 1m candlestick data for technical indicators
        # ====================================================================
        self.candles: Dict[str, Dict[str, List]] = {}  # symbol -> exchange -> [candles]
        self.max_candles = 200  # Keep 200 candles (~3.3 hours of 1m data)
        
        # ====================================================================
        # NEW: Index Prices - Spot index for basis calculation
        # ====================================================================
        self.index_prices: Dict[str, Dict[str, Dict]] = {}  # symbol -> exchange -> {price, timestamp}
        
        # Arbitrage opportunities
        self.arbitrage_opportunities: List[Dict] = []
        self.max_opportunities = 100
        
        # Connection tracking
        self.connected_exchanges: Dict[str, bool] = {}
        self._tasks: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        self._running = False
        self._started = False
        
        # Initialize data structures for all symbols
        for symbol in self.SUPPORTED_SYMBOLS:
            self.prices[symbol] = {}
            self.orderbooks[symbol] = {}
            self.trades[symbol] = {}
            self.funding_rates[symbol] = {}
            self.mark_prices[symbol] = {}
            self.liquidations[symbol] = []
            self.open_interest[symbol] = {}
            self.ticker_24h[symbol] = {}
            self.candles[symbol] = {}  # NEW
            self.index_prices[symbol] = {}  # NEW
        
        logger.info("DirectExchangeClient initialized")
    
    async def start(self) -> bool:
        """Start connections to all exchanges."""
        if self._running:
            return True
        
        self._running = True
        self._started = True
        logger.info("Starting direct exchange connections...")
        
        # Define exchange connections (matching Go Crypto Arbitrage Scanner)
        exchanges = [
            ("binance_futures", self._connect_binance_futures),
            ("binance_spot", self._connect_binance_spot),
            ("bybit_futures", self._connect_bybit_futures),
            ("bybit_spot", self._connect_bybit_spot),
            ("okx_futures", self._connect_okx_futures),
            ("kraken_futures", self._connect_kraken_futures),
            ("gate_futures", self._connect_gate_futures),
            ("hyperliquid_futures", self._connect_hyperliquid),
            ("pyth", self._connect_pyth),
        ]
        
        for name, connect_func in exchanges:
            task = asyncio.create_task(self._run_exchange(name, connect_func))
            self._tasks.append(task)
        
        # Wait for initial connections
        await asyncio.sleep(3)
        
        connected_count = sum(1 for v in self.connected_exchanges.values() if v)
        logger.info(f"Connected to {connected_count} exchanges")
        
        return connected_count > 0
    
    async def stop(self):
        """Stop all exchange connections."""
        self._running = False
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        self.connected_exchanges.clear()
        logger.info("All exchange connections stopped")
    
    async def _run_exchange(self, name: str, connect_func):
        """Run an exchange connection with auto-reconnect."""
        reconnect_delay = 5
        max_delay = 60
        
        while self._running:
            try:
                logger.info(f"Connecting to {name}...")
                await connect_func()
            except asyncio.CancelledError:
                logger.info(f"{name} connection cancelled")
                break
            except Exception as e:
                logger.warning(f"{name} connection error: {e}")
                self.connected_exchanges[name] = False
                
            if self._running:
                logger.info(f"{name} reconnecting in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, max_delay)
    
    # ========================================================================
    # Exchange Connection Methods
    # ========================================================================
    
    async def _connect_binance_futures(self):
        """Connect to Binance Futures WebSocket - ALL streams including klines & OI."""
        # Build comprehensive stream list for all symbols
        streams = []
        for sym in self.SUPPORTED_SYMBOLS:
            s = sym.lower()
            streams.extend([
                f"{s}@bookTicker",      # Best bid/ask (fastest)
                f"{s}@depth20@100ms",   # Top 20 orderbook levels (increased from 10)
                f"{s}@aggTrade",        # Aggregated trades
                f"{s}@markPrice@1s",    # Mark price + funding rate + index price
                f"{s}@forceOrder",      # Liquidations
                f"{s}@ticker",          # 24hr ticker stats
                f"{s}@kline_1m",        # 1-minute candlesticks (NEW)
            ])
        
        url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            self.connected_exchanges["binance_futures"] = True
            logger.info("✓ Connected to Binance Futures (ALL STREAMS)")
            
            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    if "data" not in data:
                        continue
                    
                    payload = data["data"]
                    event_type = payload.get("e", "")
                    symbol = payload.get("s", "").upper()
                    
                    if symbol not in self.SUPPORTED_SYMBOLS:
                        continue
                    
                    # Route to appropriate handler
                    if event_type == "bookTicker" or "b" in payload and "a" in payload and not event_type:
                        # Best bid/ask
                        bid = float(payload.get("b", 0))
                        ask = float(payload.get("a", 0))
                        if bid > 0 and ask > 0:
                            mid_price = (bid + ask) / 2
                            await self._update_price(symbol, "binance_futures", mid_price, bid, ask)
                    
                    elif event_type == "depthUpdate" or "bids" in payload:
                        # Orderbook depth
                        await self._update_orderbook(
                            symbol, "binance_futures",
                            payload.get("bids", payload.get("b", [])),
                            payload.get("asks", payload.get("a", []))
                        )
                    
                    elif event_type == "aggTrade":
                        # Aggregated trade
                        await self._update_trade(
                            symbol, "binance_futures",
                            price=float(payload.get("p", 0)),
                            quantity=float(payload.get("q", 0)),
                            is_buyer_maker=payload.get("m", False),
                            timestamp=payload.get("T", int(time.time() * 1000))
                        )
                    
                    elif event_type == "markPriceUpdate":
                        # Mark price + funding rate + index price
                        mark_price = float(payload.get("p", 0))
                        index_price = float(payload.get("i", 0))
                        await self._update_mark_price(
                            symbol, "binance_futures",
                            mark_price=mark_price,
                            index_price=index_price,
                            funding_rate=float(payload.get("r", 0)),
                            next_funding_time=payload.get("T", 0)
                        )
                        # Also update index price separately for basis calculation
                        if index_price > 0:
                            await self._update_index_price(symbol, "binance_futures", index_price)
                    
                    elif event_type == "forceOrder":
                        # Liquidation
                        order = payload.get("o", {})
                        await self._update_liquidation(
                            symbol, "binance_futures",
                            side=order.get("S", ""),
                            price=float(order.get("p", 0)),
                            quantity=float(order.get("q", 0)),
                            timestamp=order.get("T", int(time.time() * 1000))
                        )
                    
                    elif event_type == "24hrTicker":
                        # 24h statistics
                        await self._update_ticker_24h(
                            symbol, "binance_futures",
                            volume=float(payload.get("v", 0)),
                            quote_volume=float(payload.get("q", 0)),
                            high=float(payload.get("h", 0)),
                            low=float(payload.get("l", 0)),
                            price_change_pct=float(payload.get("P", 0)),
                            trades_count=int(payload.get("n", 0))
                        )
                    
                    elif event_type == "kline":
                        # 1-minute candlestick (NEW)
                        kline = payload.get("k", {})
                        if kline.get("x", False):  # Only process closed candles
                            await self._update_candle(
                                symbol, "binance_futures",
                                open_time=kline.get("t", 0),
                                open_price=float(kline.get("o", 0)),
                                high_price=float(kline.get("h", 0)),
                                low_price=float(kline.get("l", 0)),
                                close_price=float(kline.get("c", 0)),
                                volume=float(kline.get("v", 0)),
                                close_time=kline.get("T", 0),
                                quote_volume=float(kline.get("q", 0)),
                                trades=int(kline.get("n", 0)),
                                taker_buy_volume=float(kline.get("V", 0)),
                                taker_buy_quote_volume=float(kline.get("Q", 0))
                            )
                        
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.debug(f"Binance futures parse error: {e}")
    
    async def _connect_binance_spot(self):
        """Connect to Binance Spot WebSocket - ALL streams including klines."""
        # Build comprehensive stream list
        streams = []
        for sym in self.SUPPORTED_SYMBOLS:
            s = sym.lower()
            streams.extend([
                f"{s}@bookTicker",      # Best bid/ask
                f"{s}@depth20@100ms",   # Top 20 orderbook (increased from 10)
                f"{s}@aggTrade",        # Aggregated trades
                f"{s}@ticker",          # 24hr ticker stats
                f"{s}@kline_1m",        # 1-minute candlesticks (NEW)
            ])
        
        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            self.connected_exchanges["binance_spot"] = True
            logger.info("✓ Connected to Binance Spot (ALL STREAMS)")
            
            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    if "data" not in data:
                        continue
                    
                    payload = data["data"]
                    event_type = payload.get("e", "")
                    symbol = payload.get("s", "").upper()
                    
                    if symbol not in self.SUPPORTED_SYMBOLS:
                        continue
                    
                    if event_type == "bookTicker" or ("b" in payload and "a" in payload and not event_type):
                        bid = float(payload.get("b", 0))
                        ask = float(payload.get("a", 0))
                        if bid > 0 and ask > 0:
                            mid_price = (bid + ask) / 2
                            await self._update_price(symbol, "binance_spot", mid_price, bid, ask)
                    
                    elif event_type == "depthUpdate" or "bids" in payload:
                        await self._update_orderbook(
                            symbol, "binance_spot",
                            payload.get("bids", payload.get("b", [])),
                            payload.get("asks", payload.get("a", []))
                        )
                    
                    elif event_type == "aggTrade":
                        await self._update_trade(
                            symbol, "binance_spot",
                            price=float(payload.get("p", 0)),
                            quantity=float(payload.get("q", 0)),
                            is_buyer_maker=payload.get("m", False),
                            timestamp=payload.get("T", int(time.time() * 1000))
                        )
                    
                    elif event_type == "24hrTicker":
                        await self._update_ticker_24h(
                            symbol, "binance_spot",
                            volume=float(payload.get("v", 0)),
                            quote_volume=float(payload.get("q", 0)),
                            high=float(payload.get("h", 0)),
                            low=float(payload.get("l", 0)),
                            price_change_pct=float(payload.get("P", 0)),
                            trades_count=int(payload.get("n", 0))
                        )
                    
                    elif event_type == "kline":
                        # 1-minute candlestick (NEW)
                        kline = payload.get("k", {})
                        if kline.get("x", False):  # Only process closed candles
                            await self._update_candle(
                                symbol, "binance_spot",
                                open_time=kline.get("t", 0),
                                open_price=float(kline.get("o", 0)),
                                high_price=float(kline.get("h", 0)),
                                low_price=float(kline.get("l", 0)),
                                close_price=float(kline.get("c", 0)),
                                volume=float(kline.get("v", 0)),
                                close_time=kline.get("T", 0),
                                quote_volume=float(kline.get("q", 0)),
                                trades=int(kline.get("n", 0)),
                                taker_buy_volume=float(kline.get("V", 0)),
                                taker_buy_quote_volume=float(kline.get("Q", 0))
                            )
                        
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.debug(f"Binance spot parse error: {e}")
    
    async def _connect_bybit_futures(self):
        """Connect to Bybit Futures WebSocket - ALL streams including klines."""
        url = "wss://stream.bybit.com/v5/public/linear"
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            # Subscribe to ALL available streams
            args = []
            for sym in self.SUPPORTED_SYMBOLS:
                args.extend([
                    f"tickers.{sym}",           # Ticker with bid/ask/volume/OI/funding
                    f"orderbook.50.{sym}",      # Top 50 orderbook levels (increased from 25)
                    f"publicTrade.{sym}",       # Public trades
                    f"liquidation.{sym}",       # Liquidations
                    f"kline.1.{sym}",           # 1-minute candlesticks (NEW)
                ])
            
            subscribe_msg = {"op": "subscribe", "args": args}
            await ws.send(json.dumps(subscribe_msg))
            
            self.connected_exchanges["bybit_futures"] = True
            logger.info("✓ Connected to Bybit Futures (ALL STREAMS)")
            
            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    topic = data.get("topic", "")
                    payload = data.get("data", {})
                    
                    # Extract symbol from topic
                    symbol = None
                    for sym in self.SUPPORTED_SYMBOLS:
                        if sym in topic:
                            symbol = sym
                            break
                    
                    if not symbol:
                        continue
                    
                    if topic.startswith("tickers."):
                        # Ticker data with funding, OI, volume
                        bid = float(payload.get("bid1Price", 0) or 0)
                        ask = float(payload.get("ask1Price", 0) or 0)
                        last = float(payload.get("lastPrice", 0) or 0)
                        
                        if bid > 0 and ask > 0:
                            mid_price = (bid + ask) / 2
                            await self._update_price(symbol, "bybit_futures", mid_price, bid, ask)
                        
                        # Funding rate
                        funding_rate = float(payload.get("fundingRate", 0) or 0)
                        if funding_rate != 0:
                            mark = float(payload.get("markPrice", 0) or 0)
                            index = float(payload.get("indexPrice", 0) or 0)
                            next_time = int(payload.get("nextFundingTime", 0) or 0)
                            await self._update_mark_price(symbol, "bybit_futures", mark, index, funding_rate, next_time)
                        
                        # Open Interest
                        oi = float(payload.get("openInterest", 0) or 0)
                        oi_value = float(payload.get("openInterestValue", 0) or 0)
                        if oi > 0:
                            await self._update_open_interest(symbol, "bybit_futures", oi, oi_value)
                        
                        # 24h stats
                        volume = float(payload.get("volume24h", 0) or 0)
                        turnover = float(payload.get("turnover24h", 0) or 0)
                        high = float(payload.get("highPrice24h", 0) or 0)
                        low = float(payload.get("lowPrice24h", 0) or 0)
                        change_pct = float(payload.get("price24hPcnt", 0) or 0) * 100
                        await self._update_ticker_24h(symbol, "bybit_futures", volume, turnover, high, low, change_pct, 0)
                    
                    elif topic.startswith("orderbook."):
                        # Orderbook
                        bids = [[float(b[0]), float(b[1])] for b in payload.get("b", [])]
                        asks = [[float(a[0]), float(a[1])] for a in payload.get("a", [])]
                        await self._update_orderbook(symbol, "bybit_futures", bids, asks)
                    
                    elif topic.startswith("publicTrade."):
                        # Trades - data is a list
                        trades = payload if isinstance(payload, list) else [payload]
                        for trade in trades:
                            await self._update_trade(
                                symbol, "bybit_futures",
                                price=float(trade.get("p", 0)),
                                quantity=float(trade.get("v", 0)),
                                is_buyer_maker=trade.get("S", "") == "Sell",
                                timestamp=int(trade.get("T", time.time() * 1000))
                            )
                    
                    elif topic.startswith("liquidation."):
                        # Liquidation
                        await self._update_liquidation(
                            symbol, "bybit_futures",
                            side=payload.get("side", ""),
                            price=float(payload.get("price", 0)),
                            quantity=float(payload.get("size", 0)),
                            timestamp=int(payload.get("updatedTime", time.time() * 1000))
                        )
                    
                    elif topic.startswith("kline."):
                        # 1-minute candlestick (NEW)
                        candles = payload if isinstance(payload, list) else [payload]
                        for candle in candles:
                            if candle.get("confirm", False):  # Only process confirmed/closed candles
                                await self._update_candle(
                                    symbol, "bybit_futures",
                                    open_time=int(candle.get("start", 0)),
                                    open_price=float(candle.get("open", 0)),
                                    high_price=float(candle.get("high", 0)),
                                    low_price=float(candle.get("low", 0)),
                                    close_price=float(candle.get("close", 0)),
                                    volume=float(candle.get("volume", 0)),
                                    close_time=int(candle.get("end", 0)),
                                    quote_volume=float(candle.get("turnover", 0)),
                                    trades=0,  # Not provided by Bybit
                                    taker_buy_volume=0,
                                    taker_buy_quote_volume=0
                                )
                        
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.debug(f"Bybit futures parse error: {e}")
    
    async def _connect_okx_futures(self):
        """Connect to OKX Futures WebSocket - ALL streams."""
        url = "wss://ws.okx.com:8443/ws/v5/public"
        
        # OKX uses different symbol format
        okx_symbols = {
            "BTCUSDT": "BTC-USDT-SWAP",
            "ETHUSDT": "ETH-USDT-SWAP", 
            "SOLUSDT": "SOL-USDT-SWAP",
            "XRPUSDT": "XRP-USDT-SWAP"
        }
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            # Subscribe to ALL available streams
            args = []
            for inst in okx_symbols.values():
                args.extend([
                    {"channel": "tickers", "instId": inst},           # Ticker
                    {"channel": "books5", "instId": inst},            # Top 5 orderbook
                    {"channel": "trades", "instId": inst},            # Trades
                    {"channel": "mark-price", "instId": inst},        # Mark price
                    {"channel": "funding-rate", "instId": inst},      # Funding rate
                    {"channel": "open-interest", "instId": inst},     # Open interest
                    {"channel": "candle1m", "instId": inst},          # 1-minute candles (NEW)
                    {"channel": "liquidation-orders", "instType": "SWAP"},  # Liquidations
                ])
            
            subscribe_msg = {"op": "subscribe", "args": args}
            await ws.send(json.dumps(subscribe_msg))
            
            self.connected_exchanges["okx_futures"] = True
            logger.info("✓ Connected to OKX Futures (ALL STREAMS)")
            
            # Reverse mapping for symbol lookup
            reverse_symbols = {v: k for k, v in okx_symbols.items()}
            
            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    
                    if "data" not in data:
                        continue
                    
                    channel = data.get("arg", {}).get("channel", "")
                    inst_id = data.get("arg", {}).get("instId", "")
                    symbol = reverse_symbols.get(inst_id)
                    
                    for item in data["data"]:
                        # Get symbol from item if not from arg
                        if not symbol:
                            item_inst = item.get("instId", "")
                            symbol = reverse_symbols.get(item_inst)
                        
                        if not symbol:
                            continue
                        
                        if channel == "tickers":
                            bid = float(item.get("bidPx", 0) or 0)
                            ask = float(item.get("askPx", 0) or 0)
                            last = float(item.get("last", 0) or 0)
                            
                            if bid > 0 and ask > 0:
                                mid_price = (bid + ask) / 2
                                await self._update_price(symbol, "okx_futures", mid_price, bid, ask)
                            
                            # 24h stats from ticker
                            vol = float(item.get("vol24h", 0) or 0)
                            volCcy = float(item.get("volCcy24h", 0) or 0)
                            high = float(item.get("high24h", 0) or 0)
                            low = float(item.get("low24h", 0) or 0)
                            await self._update_ticker_24h(symbol, "okx_futures", vol, volCcy, high, low, 0, 0)
                        
                        elif channel == "books5":
                            bids = [[float(b[0]), float(b[1])] for b in item.get("bids", [])]
                            asks = [[float(a[0]), float(a[1])] for a in item.get("asks", [])]
                            await self._update_orderbook(symbol, "okx_futures", bids, asks)
                        
                        elif channel == "trades":
                            await self._update_trade(
                                symbol, "okx_futures",
                                price=float(item.get("px", 0)),
                                quantity=float(item.get("sz", 0)),
                                is_buyer_maker=item.get("side", "") == "sell",
                                timestamp=int(item.get("ts", time.time() * 1000))
                            )
                        
                        elif channel == "mark-price":
                            mark = float(item.get("markPx", 0) or 0)
                            await self._update_mark_price(symbol, "okx_futures", mark, 0, 0, 0)
                        
                        elif channel == "funding-rate":
                            rate = float(item.get("fundingRate", 0) or 0)
                            next_time = int(item.get("nextFundingTime", 0) or 0)
                            await self._update_mark_price(symbol, "okx_futures", 0, 0, rate, next_time)
                        
                        elif channel == "open-interest":
                            oi = float(item.get("oi", 0) or 0)
                            oi_ccy = float(item.get("oiCcy", 0) or 0)
                            await self._update_open_interest(symbol, "okx_futures", oi, oi_ccy)
                        
                        elif channel == "liquidation-orders":
                            liq_inst = item.get("instId", "")
                            liq_symbol = reverse_symbols.get(liq_inst)
                            if liq_symbol:
                                await self._update_liquidation(
                                    liq_symbol, "okx_futures",
                                    side=item.get("side", ""),
                                    price=float(item.get("bkPx", 0)),
                                    quantity=float(item.get("sz", 0)),
                                    timestamp=int(item.get("ts", time.time() * 1000))
                                )
                        
                        elif channel == "candle1m":
                            # OKX candle format: [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
                            # item is a list: [1597026383085, "8533.02", "8553.74", "8527.17", "8548.26", "45247", ...]
                            if isinstance(item, list) and len(item) >= 6:
                                is_confirmed = item[8] == "1" if len(item) > 8 else True
                                if is_confirmed:
                                    await self._update_candle(
                                        symbol, "okx_futures",
                                        open_time=int(item[0]),
                                        open_price=float(item[1]),
                                        high_price=float(item[2]),
                                        low_price=float(item[3]),
                                        close_price=float(item[4]),
                                        volume=float(item[5]),
                                        close_time=int(item[0]) + 60000,  # 1 min later
                                        quote_volume=float(item[7]) if len(item) > 7 else 0,
                                        trades=0,
                                        taker_buy_volume=0,
                                        taker_buy_quote_volume=0
                                    )
                                
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.debug(f"OKX parse error: {e}")
    
    async def _connect_bybit_spot(self):
        """Connect to Bybit Spot WebSocket - ALL streams."""
        url = "wss://stream.bybit.com/v5/public/spot"
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            # Subscribe to ALL available spot streams
            args = []
            for sym in self.SUPPORTED_SYMBOLS:
                args.extend([
                    f"tickers.{sym}",           # Ticker with bid/ask/volume
                    f"orderbook.50.{sym}",      # Top 50 orderbook levels
                    f"publicTrade.{sym}",       # Public trades
                    f"kline.1.{sym}",           # 1-minute candles (NEW)
                ])
            
            subscribe_msg = {"op": "subscribe", "args": args}
            await ws.send(json.dumps(subscribe_msg))
            
            self.connected_exchanges["bybit_spot"] = True
            logger.info("✓ Connected to Bybit Spot (ALL STREAMS)")
            
            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    topic = data.get("topic", "")
                    payload = data.get("data", {})
                    
                    # Extract symbol from topic
                    symbol = None
                    for sym in self.SUPPORTED_SYMBOLS:
                        if sym in topic:
                            symbol = sym
                            break
                    
                    if not symbol:
                        continue
                    
                    if topic.startswith("tickers."):
                        bid = float(payload.get("bid1Price", 0) or 0)
                        ask = float(payload.get("ask1Price", 0) or 0)
                        last = float(payload.get("lastPrice", 0) or 0)
                        
                        if bid > 0 and ask > 0:
                            mid_price = (bid + ask) / 2
                            await self._update_price(symbol, "bybit_spot", mid_price, bid, ask)
                        
                        # 24h stats
                        volume = float(payload.get("volume24h", 0) or 0)
                        turnover = float(payload.get("turnover24h", 0) or 0)
                        high = float(payload.get("highPrice24h", 0) or 0)
                        low = float(payload.get("lowPrice24h", 0) or 0)
                        change_pct = float(payload.get("price24hPcnt", 0) or 0) * 100
                        await self._update_ticker_24h(symbol, "bybit_spot", volume, turnover, high, low, change_pct, 0)
                    
                    elif topic.startswith("orderbook."):
                        bids = [[float(b[0]), float(b[1])] for b in payload.get("b", [])]
                        asks = [[float(a[0]), float(a[1])] for a in payload.get("a", [])]
                        await self._update_orderbook(symbol, "bybit_spot", bids, asks)
                    
                    elif topic.startswith("publicTrade."):
                        trades = payload if isinstance(payload, list) else [payload]
                        for trade in trades:
                            await self._update_trade(
                                symbol, "bybit_spot",
                                price=float(trade.get("p", 0)),
                                quantity=float(trade.get("v", 0)),
                                is_buyer_maker=trade.get("S", "") == "Sell",
                                timestamp=int(trade.get("T", time.time() * 1000))
                            )
                    
                    elif topic.startswith("kline."):
                        # 1-minute candlestick (NEW)
                        candles = payload if isinstance(payload, list) else [payload]
                        for candle in candles:
                            if candle.get("confirm", False):  # Only process confirmed/closed candles
                                await self._update_candle(
                                    symbol, "bybit_spot",
                                    open_time=int(candle.get("start", 0)),
                                    open_price=float(candle.get("open", 0)),
                                    high_price=float(candle.get("high", 0)),
                                    low_price=float(candle.get("low", 0)),
                                    close_price=float(candle.get("close", 0)),
                                    volume=float(candle.get("volume", 0)),
                                    close_time=int(candle.get("end", 0)),
                                    quote_volume=float(candle.get("turnover", 0)),
                                    trades=0,
                                    taker_buy_volume=0,
                                    taker_buy_quote_volume=0
                                )
                            
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.debug(f"Bybit spot parse error: {e}")
    
    async def _connect_kraken_futures(self):
        """Connect to Kraken Futures WebSocket - ALL streams."""
        url = "wss://futures.kraken.com/ws/v1"
        
        # Kraken uses different symbol format (PF = Perpetual Futures)
        kraken_symbols = {
            "BTCUSDT": "PF_BTCUSD",
            "ETHUSDT": "PF_ETHUSD",
            "SOLUSDT": "PF_SOLUSD",
            "XRPUSDT": "PF_XRPUSD"
        }
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            product_ids = list(kraken_symbols.values())
            
            # Subscribe to multiple feeds
            # Ticker feed
            await ws.send(json.dumps({
                "event": "subscribe",
                "feed": "ticker",
                "product_ids": product_ids
            }))
            
            # Orderbook feed
            await ws.send(json.dumps({
                "event": "subscribe",
                "feed": "book",
                "product_ids": product_ids
            }))
            
            # Trade feed
            await ws.send(json.dumps({
                "event": "subscribe",
                "feed": "trade",
                "product_ids": product_ids
            }))
            
            # Open interest feed
            await ws.send(json.dumps({
                "event": "subscribe",
                "feed": "open_interest",
                "product_ids": product_ids
            }))
            
            self.connected_exchanges["kraken_futures"] = True
            logger.info("✓ Connected to Kraken Futures (ALL STREAMS)")
            
            # Reverse mapping
            reverse_symbols = {v: k for k, v in kraken_symbols.items()}
            
            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    feed = data.get("feed", "")
                    product_id = data.get("product_id", "")
                    symbol = reverse_symbols.get(product_id)
                    
                    if feed == "ticker" and symbol:
                        bid = float(data.get("bid", 0) or 0)
                        ask = float(data.get("ask", 0) or 0)
                        last = float(data.get("last", 0) or 0)
                        
                        if bid > 0 and ask > 0:
                            mid_price = (bid + ask) / 2
                            await self._update_price(symbol, "kraken_futures", mid_price, bid, ask)
                        
                        # Funding rate from ticker
                        funding_rate = float(data.get("fundingRate", 0) or 0)
                        funding_rate_pred = float(data.get("fundingRatePrediction", 0) or 0)
                        mark = float(data.get("markPrice", 0) or 0)
                        index = float(data.get("indexPrice", 0) or 0)
                        next_funding = int(data.get("nextFundingRateTime", 0) or 0)
                        
                        if funding_rate != 0 or mark > 0:
                            await self._update_mark_price(symbol, "kraken_futures", mark, index, funding_rate, next_funding)
                        
                        # 24h stats
                        vol = float(data.get("vol24h", 0) or 0)
                        open_price = float(data.get("open24h", 0) or 0)
                        high = float(data.get("high24h", 0) or 0)
                        low = float(data.get("low24h", 0) or 0)
                        change_pct = ((last - open_price) / open_price * 100) if open_price > 0 else 0
                        await self._update_ticker_24h(symbol, "kraken_futures", vol, 0, high, low, change_pct, 0)
                        
                        # Open interest from ticker
                        oi = float(data.get("openInterest", 0) or 0)
                        if oi > 0:
                            await self._update_open_interest(symbol, "kraken_futures", oi, 0)
                    
                    elif feed == "book_snapshot" and symbol:
                        bids = [[float(b["price"]), float(b["qty"])] for b in data.get("bids", [])[:10]]
                        asks = [[float(a["price"]), float(a["qty"])] for a in data.get("asks", [])[:10]]
                        await self._update_orderbook(symbol, "kraken_futures", bids, asks)
                    
                    elif feed == "trade" and symbol:
                        trades = data.get("trades", [data]) if "trades" in data else [data]
                        for trade in trades:
                            side = trade.get("side", "")
                            await self._update_trade(
                                symbol, "kraken_futures",
                                price=float(trade.get("price", 0)),
                                quantity=float(trade.get("qty", 0)),
                                is_buyer_maker=side == "sell",
                                timestamp=int(trade.get("time", time.time() * 1000))
                            )
                    
                    elif feed == "open_interest" and symbol:
                        oi = float(data.get("openInterest", 0) or 0)
                        await self._update_open_interest(symbol, "kraken_futures", oi, 0)
                        
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.debug(f"Kraken parse error: {e}")
    
    async def _connect_gate_futures(self):
        """Connect to Gate.io Futures WebSocket - ALL streams."""
        url = "wss://fx-ws.gateio.ws/v4/ws/usdt"
        
        # Gate uses format like "BTC_USDT"
        gate_symbols = {
            "BTCUSDT": "BTC_USDT",
            "ETHUSDT": "ETH_USDT",
            "SOLUSDT": "SOL_USDT",
            "XRPUSDT": "XRP_USDT"
        }
        contracts = list(gate_symbols.values())
        reverse_symbols = {v: k for k, v in gate_symbols.items()}
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            # Subscribe to multiple channels
            channels = [
                ("futures.tickers", contracts),          # Ticker with price/funding
                ("futures.order_book", contracts),       # Orderbook
                ("futures.trades", contracts),           # Trades
                ("futures.liquidates", contracts),       # Liquidations
                ("futures.candlesticks", contracts),     # 1-minute candles (NEW)
            ]
            
            for channel, payload in channels:
                subscribe_msg = {
                    "time": int(time.time()),
                    "channel": channel,
                    "event": "subscribe",
                    "payload": payload if channel != "futures.candlesticks" else [f"{c}_1m" for c in payload]
                }
                await ws.send(json.dumps(subscribe_msg))
            
            self.connected_exchanges["gate_futures"] = True
            logger.info("✓ Connected to Gate.io Futures (ALL STREAMS)")
            
            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    channel = data.get("channel", "")
                    event = data.get("event", "")
                    result = data.get("result", {})
                    
                    if event != "update":
                        continue
                    
                    if channel == "futures.tickers":
                        items = result if isinstance(result, list) else [result]
                        for ticker in items:
                            contract = ticker.get("contract", "")
                            symbol = reverse_symbols.get(contract)
                            
                            if symbol:
                                last = float(ticker.get("last", 0) or 0)
                                mark = float(ticker.get("mark_price", 0) or 0)
                                index = float(ticker.get("index_price", 0) or 0)
                                
                                if last > 0:
                                    await self._update_price(symbol, "gate_futures", last, last, last)
                                
                                # Funding rate
                                funding = float(ticker.get("funding_rate", 0) or 0)
                                next_funding = int(ticker.get("funding_next_apply", 0) or 0)
                                if funding != 0 or mark > 0:
                                    await self._update_mark_price(symbol, "gate_futures", mark, index, funding, next_funding)
                                
                                # 24h stats
                                vol = float(ticker.get("volume_24h", 0) or 0)
                                vol_usd = float(ticker.get("volume_24h_usd", 0) or 0)
                                high = float(ticker.get("high_24h", 0) or 0)
                                low = float(ticker.get("low_24h", 0) or 0)
                                change = float(ticker.get("change_percentage", 0) or 0)
                                await self._update_ticker_24h(symbol, "gate_futures", vol, vol_usd, high, low, change, 0)
                                
                                # Open interest
                                oi = float(ticker.get("total_size", 0) or 0)
                                if oi > 0:
                                    await self._update_open_interest(symbol, "gate_futures", oi, vol_usd)
                    
                    elif channel == "futures.order_book":
                        contract = result.get("contract", "")
                        symbol = reverse_symbols.get(contract)
                        if symbol:
                            bids = [[float(b["p"]), float(b["s"])] for b in result.get("bids", [])[:10]]
                            asks = [[float(a["p"]), float(a["s"])] for a in result.get("asks", [])[:10]]
                            await self._update_orderbook(symbol, "gate_futures", bids, asks)
                    
                    elif channel == "futures.trades":
                        trades = result if isinstance(result, list) else [result]
                        for trade in trades:
                            contract = trade.get("contract", "")
                            symbol = reverse_symbols.get(contract)
                            if symbol:
                                await self._update_trade(
                                    symbol, "gate_futures",
                                    price=float(trade.get("price", 0)),
                                    quantity=float(trade.get("size", 0)),
                                    is_buyer_maker=trade.get("side") == "sell",
                                    timestamp=int(trade.get("create_time", time.time()) * 1000)
                                )
                    
                    elif channel == "futures.liquidates":
                        liqs = result if isinstance(result, list) else [result]
                        for liq in liqs:
                            contract = liq.get("contract", "")
                            symbol = reverse_symbols.get(contract)
                            if symbol:
                                await self._update_liquidation(
                                    symbol, "gate_futures",
                                    side=liq.get("side", ""),
                                    price=float(liq.get("price", 0)),
                                    quantity=float(liq.get("size", 0)),
                                    timestamp=int(liq.get("time", time.time()) * 1000)
                                )
                    
                    elif channel == "futures.candlesticks":
                        # Gate.io candle format: {n: name_interval, t: timestamp, o, h, l, c, v, ...}
                        candles = result if isinstance(result, list) else [result]
                        for candle in candles:
                            # Parse contract from name (e.g., "BTC_USDT_1m" -> "BTC_USDT")
                            name = candle.get("n", "")
                            contract = name.rsplit("_", 1)[0] if "_1m" in name else name
                            symbol = reverse_symbols.get(contract)
                            if symbol:
                                await self._update_candle(
                                    symbol, "gate_futures",
                                    open_time=int(candle.get("t", 0)) * 1000,
                                    open_price=float(candle.get("o", 0)),
                                    high_price=float(candle.get("h", 0)),
                                    low_price=float(candle.get("l", 0)),
                                    close_price=float(candle.get("c", 0)),
                                    volume=float(candle.get("v", 0)),
                                    close_time=(int(candle.get("t", 0)) + 60) * 1000,
                                    quote_volume=float(candle.get("sum", 0)),
                                    trades=0,
                                    taker_buy_volume=0,
                                    taker_buy_quote_volume=0
                                )
                                
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.debug(f"Gate parse error: {e}")
    
    async def _connect_hyperliquid(self):
        """Connect to Hyperliquid WebSocket - ALL streams."""
        url = "wss://api.hyperliquid.xyz/ws"
        
        # Hyperliquid uses simple symbols like "BTC", "ETH"
        hl_symbols = {
            "BTCUSDT": "BTC",
            "ETHUSDT": "ETH",
            "SOLUSDT": "SOL",
            "XRPUSDT": "XRP"
        }
        reverse_symbols = {v: k for k, v in hl_symbols.items()}
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            # Subscribe to multiple streams
            # 1. All mids (mid prices for all assets)
            await ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "allMids"}
            }))
            
            # 2. L2 Orderbook for each symbol
            for hl_sym in hl_symbols.values():
                await ws.send(json.dumps({
                    "method": "subscribe",
                    "subscription": {"type": "l2Book", "coin": hl_sym}
                }))
            
            # 3. Trades for each symbol
            for hl_sym in hl_symbols.values():
                await ws.send(json.dumps({
                    "method": "subscribe",
                    "subscription": {"type": "trades", "coin": hl_sym}
                }))
            
            # 4. User fills / liquidations (public)
            await ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "activeAssetCtx"}
            }))
            
            self.connected_exchanges["hyperliquid_futures"] = True
            logger.info("✓ Connected to Hyperliquid (ALL STREAMS)")
            
            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    channel = data.get("channel", "")
                    payload = data.get("data", {})
                    
                    if channel == "allMids":
                        mids = payload.get("mids", {})
                        for hl_sym, our_sym in reverse_symbols.items():
                            if hl_sym in mids:
                                mid_price = float(mids[hl_sym])
                                if mid_price > 0:
                                    await self._update_price(our_sym, "hyperliquid_futures", mid_price, mid_price, mid_price)
                    
                    elif channel == "l2Book":
                        coin = payload.get("coin", "")
                        symbol = reverse_symbols.get(coin)
                        if symbol:
                            book = payload.get("levels", [[], []])
                            # levels[0] = bids, levels[1] = asks
                            bids = [[float(b["px"]), float(b["sz"])] for b in book[0][:10]] if len(book) > 0 else []
                            asks = [[float(a["px"]), float(a["sz"])] for a in book[1][:10]] if len(book) > 1 else []
                            await self._update_orderbook(symbol, "hyperliquid_futures", bids, asks)
                    
                    elif channel == "trades":
                        trades = payload if isinstance(payload, list) else [payload]
                        for trade in trades:
                            coin = trade.get("coin", "")
                            symbol = reverse_symbols.get(coin)
                            if symbol:
                                await self._update_trade(
                                    symbol, "hyperliquid_futures",
                                    price=float(trade.get("px", 0)),
                                    quantity=float(trade.get("sz", 0)),
                                    is_buyer_maker=trade.get("side", "") == "A",  # A = ask (sell)
                                    timestamp=int(trade.get("time", time.time() * 1000))
                                )
                    
                    elif channel == "activeAssetCtx":
                        # Contains funding rates and open interest
                        for ctx in payload if isinstance(payload, list) else [payload]:
                            coin = ctx.get("coin", "")
                            symbol = reverse_symbols.get(coin)
                            if symbol:
                                funding = float(ctx.get("funding", 0) or 0)
                                oi = float(ctx.get("openInterest", 0) or 0)
                                mark = float(ctx.get("markPx", 0) or 0)
                                
                                if funding != 0 or mark > 0:
                                    await self._update_mark_price(symbol, "hyperliquid_futures", mark, 0, funding, 0)
                                if oi > 0:
                                    await self._update_open_interest(symbol, "hyperliquid_futures", oi, 0)
                                    
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.debug(f"Hyperliquid parse error: {e}")
    
    async def _connect_pyth(self):
        """Connect to Pyth Network WebSocket for oracle prices."""
        url = "wss://hermes.pyth.network/ws"
        
        # Pyth price feed IDs for major crypto (these are the official Pyth price IDs)
        pyth_feeds = {
            # BTC/USD
            "BTCUSDT": "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
            # ETH/USD  
            "ETHUSDT": "ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
            # SOL/USD
            "SOLUSDT": "ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
            # XRP/USD
            "XRPUSDT": "ec5d399846a9209f3fe5881d70aae9268c94339ff9817e8d18ff19fa05eea1c8"
        }
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            # Subscribe to price feeds
            subscribe_msg = {
                "type": "subscribe",
                "ids": list(pyth_feeds.values())
            }
            await ws.send(json.dumps(subscribe_msg))
            
            self.connected_exchanges["pyth"] = True
            logger.info("✓ Connected to Pyth Oracle")
            
            # Reverse mapping
            reverse_feeds = {v: k for k, v in pyth_feeds.items()}
            
            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    
                    if data.get("type") == "price_update":
                        price_feed = data.get("price_feed", {})
                        feed_id = price_feed.get("id", "")
                        symbol = reverse_feeds.get(feed_id)
                        
                        if symbol:
                            price_data = price_feed.get("price", {})
                            price = float(price_data.get("price", 0) or 0)
                            expo = int(price_data.get("expo", 0) or 0)
                            
                            if price > 0:
                                # Pyth prices need to be adjusted by exponent
                                actual_price = price * (10 ** expo)
                                if actual_price > 0:
                                    await self._update_price(symbol, "pyth", actual_price, actual_price, actual_price)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.debug(f"Pyth parse error: {e}")
    
    # ========================================================================
    # Price Management
    # ========================================================================
    
    async def _update_price(self, symbol: str, exchange: str, price: float, bid: float = 0, ask: float = 0):
        """Update price and check for arbitrage."""
        timestamp = int(time.time() * 1000)
        
        async with self._lock:
            self.prices[symbol][exchange] = {
                "price": price,
                "bid": bid,
                "ask": ask,
                "timestamp": timestamp,
                "exchange": exchange,
                "updated_at": datetime.utcnow().isoformat()
            }
        
        # Check for arbitrage opportunities
        await self._check_arbitrage(symbol)
    
    async def _update_orderbook(self, symbol: str, exchange: str, bids: List, asks: List):
        """Update orderbook depth data."""
        timestamp = int(time.time() * 1000)
        
        async with self._lock:
            # Parse bids and asks (format: [[price, qty], ...])
            parsed_bids = []
            parsed_asks = []
            
            for bid in bids[:10]:  # Top 10 levels
                if isinstance(bid, list) and len(bid) >= 2:
                    parsed_bids.append({
                        "price": float(bid[0]),
                        "quantity": float(bid[1])
                    })
            
            for ask in asks[:10]:
                if isinstance(ask, list) and len(ask) >= 2:
                    parsed_asks.append({
                        "price": float(ask[0]),
                        "quantity": float(ask[1])
                    })
            
            self.orderbooks[symbol][exchange] = {
                "bids": parsed_bids,
                "asks": parsed_asks,
                "timestamp": timestamp,
                "exchange": exchange,
                "bid_depth": sum(b["quantity"] for b in parsed_bids),
                "ask_depth": sum(a["quantity"] for a in parsed_asks),
                "spread": parsed_asks[0]["price"] - parsed_bids[0]["price"] if parsed_bids and parsed_asks else 0,
                "spread_pct": ((parsed_asks[0]["price"] - parsed_bids[0]["price"]) / parsed_bids[0]["price"] * 100) if parsed_bids and parsed_asks and parsed_bids[0]["price"] > 0 else 0
            }
    
    async def _update_trade(self, symbol: str, exchange: str, price: float, quantity: float, 
                           is_buyer_maker: bool, timestamp: int):
        """Update trade stream data."""
        trade = {
            "price": price,
            "quantity": quantity,
            "side": "sell" if is_buyer_maker else "buy",  # If buyer is maker, taker sold
            "value": price * quantity,
            "timestamp": timestamp,
            "exchange": exchange
        }
        
        async with self._lock:
            if exchange not in self.trades[symbol]:
                self.trades[symbol][exchange] = []
            
            self.trades[symbol][exchange].insert(0, trade)
            
            # Keep only recent trades
            if len(self.trades[symbol][exchange]) > self.max_trades_per_symbol:
                self.trades[symbol][exchange] = self.trades[symbol][exchange][:self.max_trades_per_symbol]
    
    async def _update_mark_price(self, symbol: str, exchange: str, mark_price: float, 
                                 index_price: float, funding_rate: float, next_funding_time: int):
        """Update mark price and funding rate."""
        timestamp = int(time.time() * 1000)
        
        async with self._lock:
            self.mark_prices[symbol][exchange] = {
                "mark_price": mark_price,
                "index_price": index_price,
                "basis": mark_price - index_price if index_price > 0 else 0,
                "basis_pct": ((mark_price - index_price) / index_price * 100) if index_price > 0 else 0,
                "timestamp": timestamp,
                "exchange": exchange
            }
            
            self.funding_rates[symbol][exchange] = {
                "rate": funding_rate,
                "rate_pct": funding_rate * 100,
                "annualized_rate": funding_rate * 3 * 365 * 100,  # 8hr funding * 3 * 365
                "next_funding_time": next_funding_time,
                "timestamp": timestamp,
                "exchange": exchange
            }
    
    async def _update_liquidation(self, symbol: str, exchange: str, side: str, 
                                  price: float, quantity: float, timestamp: int):
        """Update liquidation events."""
        liquidation = {
            "side": side,
            "price": price,
            "quantity": quantity,
            "value": price * quantity,
            "timestamp": timestamp,
            "exchange": exchange,
            "detected_at": datetime.utcnow().isoformat()
        }
        
        async with self._lock:
            self.liquidations[symbol].insert(0, liquidation)
            
            if len(self.liquidations[symbol]) > self.max_liquidations:
                self.liquidations[symbol] = self.liquidations[symbol][:self.max_liquidations]
    
    async def _update_ticker_24h(self, symbol: str, exchange: str, volume: float, 
                                 quote_volume: float, high: float, low: float,
                                 price_change_pct: float, trades_count: int):
        """Update 24h ticker statistics."""
        timestamp = int(time.time() * 1000)
        
        async with self._lock:
            self.ticker_24h[symbol][exchange] = {
                "volume": volume,
                "quote_volume": quote_volume,
                "high_24h": high,
                "low_24h": low,
                "price_change_pct": price_change_pct,
                "trades_count": trades_count,
                "timestamp": timestamp,
                "exchange": exchange
            }
    
    async def _update_open_interest(self, symbol: str, exchange: str, 
                                    open_interest: float, open_interest_value: float):
        """Update open interest data."""
        timestamp = int(time.time() * 1000)
        
        async with self._lock:
            self.open_interest[symbol][exchange] = {
                "open_interest": open_interest,
                "open_interest_value": open_interest_value,
                "timestamp": timestamp,
                "exchange": exchange
            }
    
    async def _update_candle(self, symbol: str, exchange: str,
                             open_time: int, open_price: float, high_price: float,
                             low_price: float, close_price: float, volume: float,
                             close_time: int, quote_volume: float, trades: int,
                             taker_buy_volume: float, taker_buy_quote_volume: float):
        """Update OHLCV candle data."""
        async with self._lock:
            if symbol not in self.candles:
                self.candles[symbol] = {}
            if exchange not in self.candles[symbol]:
                self.candles[symbol][exchange] = []
            
            candle = {
                "open_time": open_time,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "close_time": close_time,
                "quote_volume": quote_volume,
                "trades": trades,
                "taker_buy_volume": taker_buy_volume,
                "taker_buy_quote_volume": taker_buy_quote_volume,
                "exchange": exchange,
                "timestamp": int(time.time() * 1000)
            }
            
            # Avoid duplicates based on open_time
            existing_times = {c["open_time"] for c in self.candles[symbol][exchange]}
            if open_time not in existing_times:
                self.candles[symbol][exchange].append(candle)
                # Keep only last N candles (sorted by open_time)
                self.candles[symbol][exchange] = sorted(
                    self.candles[symbol][exchange], 
                    key=lambda x: x["open_time"]
                )[-self.max_candles:]
    
    async def _update_index_price(self, symbol: str, exchange: str, index_price: float):
        """Update index price data."""
        timestamp = int(time.time() * 1000)
        
        async with self._lock:
            if symbol not in self.index_prices:
                self.index_prices[symbol] = {}
            
            self.index_prices[symbol][exchange] = {
                "index_price": index_price,
                "exchange": exchange,
                "timestamp": timestamp
            }
    
    async def _check_arbitrage(self, symbol: str):
        """Check for arbitrage opportunities after price update."""
        async with self._lock:
            prices = self.prices.get(symbol, {})
            if len(prices) < 2:
                return
            
            # Get valid prices
            price_list = [
                (ex, data["price"], data.get("bid", data["price"]), data.get("ask", data["price"]))
                for ex, data in prices.items()
                if data.get("price", 0) > 0
            ]
            
            if len(price_list) < 2:
                return
            
            # Find min/max considering bid/ask
            min_ask_exchange = min(price_list, key=lambda x: x[3])  # Lowest ask (buy here)
            max_bid_exchange = max(price_list, key=lambda x: x[2])  # Highest bid (sell here)
            
            buy_exchange, _, _, buy_price = min_ask_exchange
            sell_exchange, _, sell_price, _ = max_bid_exchange
            
            if buy_exchange == sell_exchange:
                return
            
            if buy_price > 0 and sell_price > buy_price:
                profit_pct = ((sell_price - buy_price) / buy_price) * 100
                
                # Only record if profit is meaningful (>0.02%)
                if profit_pct >= 0.02:
                    opportunity = {
                        "symbol": symbol,
                        "buy_source": buy_exchange,
                        "sell_source": sell_exchange,
                        "buy_price": buy_price,
                        "sell_price": sell_price,
                        "profit_pct": round(profit_pct, 4),
                        "profit_absolute": round(sell_price - buy_price, 6),
                        "timestamp": int(time.time() * 1000),
                        "detected_at": datetime.utcnow().isoformat()
                    }
                    
                    # Avoid duplicate opportunities (same route within 5 seconds)
                    key = f"{symbol}_{buy_exchange}_{sell_exchange}"
                    recent = [
                        o for o in self.arbitrage_opportunities[:10]
                        if f"{o['symbol']}_{o['buy_source']}_{o['sell_source']}" == key
                    ]
                    
                    if not recent or (time.time() * 1000 - recent[0]["timestamp"]) > 5000:
                        self.arbitrage_opportunities.insert(0, opportunity)
                        
                        # Trim to max size
                        if len(self.arbitrage_opportunities) > self.max_opportunities:
                            self.arbitrage_opportunities = self.arbitrage_opportunities[:self.max_opportunities]
                        
                        if profit_pct >= 0.05:
                            logger.info(
                                f"Arbitrage: {symbol} | Buy {buy_exchange} @ {buy_price:.2f} | "
                                f"Sell {sell_exchange} @ {sell_price:.2f} | Profit: {profit_pct:.4f}%"
                            )
    
    # ========================================================================
    # Public API Methods
    # ========================================================================
    
    async def connect(self) -> bool:
        """Alias for start() - compatibility with websocket_client interface."""
        return await self.start()
    
    async def disconnect(self):
        """Alias for stop() - compatibility with websocket_client interface."""
        await self.stop()
    
    async def get_prices_snapshot(self, symbol: Optional[str] = None) -> Dict:
        """
        Get current prices from all exchanges.
        
        Args:
            symbol: Specific symbol to get prices for, or None for all symbols
            
        Returns:
            Dict mapping symbols to exchange prices
        """
        if not self._started:
            await self.start()
        
        async with self._lock:
            if symbol:
                return {symbol: dict(self.prices.get(symbol, {}))}
            return {sym: dict(prices) for sym, prices in self.prices.items()}
    
    async def get_spreads_snapshot(self, symbol: str) -> Dict:
        """
        Calculate spread matrix for a symbol.
        
        Args:
            symbol: Symbol to calculate spreads for
            
        Returns:
            Dict with spread matrix between all exchanges
        """
        if not self._started:
            await self.start()
        
        async with self._lock:
            prices = self.prices.get(symbol, {})
            if len(prices) < 2:
                return {"spreads": {}, "timestamp": datetime.utcnow().isoformat()}
            
            spreads = {}
            exchanges = list(prices.keys())
            
            for ex1 in exchanges:
                spreads[ex1] = {}
                p1 = prices[ex1].get("price", 0)
                
                for ex2 in exchanges:
                    if ex1 != ex2:
                        p2 = prices[ex2].get("price", 0)
                        if p1 > 0 and p2 > 0:
                            spread_pct = ((p2 - p1) / p1) * 100
                            spreads[ex1][ex2] = round(spread_pct, 4)
            
            return {
                "symbol": symbol,
                "spreads": spreads,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_arbitrage_opportunities(
        self,
        symbol: Optional[str] = None,
        min_profit: float = 0.0,
        limit: int = 20
    ) -> List[Dict]:
        """
        Get recent arbitrage opportunities.
        
        Args:
            symbol: Filter by symbol (None for all)
            min_profit: Minimum profit percentage
            limit: Maximum number to return
            
        Returns:
            List of opportunity dicts
        """
        if not self._started:
            await self.start()
        
        async with self._lock:
            opportunities = list(self.arbitrage_opportunities)
        
        # Filter by symbol
        if symbol:
            opportunities = [o for o in opportunities if o.get("symbol") == symbol]
        
        # Filter by minimum profit
        if min_profit > 0:
            opportunities = [o for o in opportunities if o.get("profit_pct", 0) >= min_profit]
        
        return opportunities[:limit]
    
    async def get_orderbooks(self, symbol: Optional[str] = None, exchange: Optional[str] = None) -> Dict:
        """
        Get orderbook data from all or specific exchanges.
        
        Args:
            symbol: Specific symbol or None for all
            exchange: Specific exchange or None for all
            
        Returns:
            Dict with orderbook data including bids, asks, depth, spread
        """
        if not self._started:
            await self.start()
        
        async with self._lock:
            result = {}
            symbols = [symbol] if symbol else self.SUPPORTED_SYMBOLS
            
            for sym in symbols:
                sym_data = self.orderbooks.get(sym, {})
                if exchange:
                    if exchange in sym_data:
                        result[sym] = {exchange: sym_data[exchange]}
                else:
                    if sym_data:
                        result[sym] = dict(sym_data)
            
            return result
    
    async def get_trades(self, symbol: Optional[str] = None, exchange: Optional[str] = None, limit: int = 50) -> Dict:
        """
        Get recent trades from all or specific exchanges.
        
        Args:
            symbol: Specific symbol or None for all
            exchange: Specific exchange or None for all
            limit: Max trades per exchange
            
        Returns:
            Dict with trade data including price, quantity, side, value
        """
        if not self._started:
            await self.start()
        
        async with self._lock:
            result = {}
            symbols = [symbol] if symbol else self.SUPPORTED_SYMBOLS
            
            for sym in symbols:
                sym_data = self.trades.get(sym, {})
                if sym_data:
                    result[sym] = {}
                    for ex, trades in sym_data.items():
                        if exchange and ex != exchange:
                            continue
                        result[sym][ex] = trades[:limit]
            
            return result
    
    async def get_funding_rates(self, symbol: Optional[str] = None) -> Dict:
        """
        Get funding rates from all exchanges.
        
        Args:
            symbol: Specific symbol or None for all
            
        Returns:
            Dict with funding rate data including rate_pct, annualized_rate
        """
        if not self._started:
            await self.start()
        
        async with self._lock:
            if symbol:
                return {symbol: dict(self.funding_rates.get(symbol, {}))}
            return {sym: dict(data) for sym, data in self.funding_rates.items() if data}
    
    async def get_mark_prices(self, symbol: Optional[str] = None) -> Dict:
        """
        Get mark prices and basis from all exchanges.
        
        Args:
            symbol: Specific symbol or None for all
            
        Returns:
            Dict with mark_price, index_price, basis, basis_pct
        """
        if not self._started:
            await self.start()
        
        async with self._lock:
            if symbol:
                return {symbol: dict(self.mark_prices.get(symbol, {}))}
            return {sym: dict(data) for sym, data in self.mark_prices.items() if data}
    
    async def get_liquidations(self, symbol: Optional[str] = None, limit: int = 20) -> Dict:
        """
        Get recent liquidation events.
        
        Args:
            symbol: Specific symbol or None for all
            limit: Max liquidations to return
            
        Returns:
            Dict with liquidation events including side, price, quantity, value
        """
        if not self._started:
            await self.start()
        
        async with self._lock:
            if symbol:
                return {symbol: self.liquidations.get(symbol, [])[:limit]}
            return {sym: liqs[:limit] for sym, liqs in self.liquidations.items() if liqs}
    
    async def get_open_interest(self, symbol: Optional[str] = None) -> Dict:
        """
        Get open interest from all exchanges.
        
        Args:
            symbol: Specific symbol or None for all
            
        Returns:
            Dict with open_interest and open_interest_value per exchange
        """
        if not self._started:
            await self.start()
        
        async with self._lock:
            if symbol:
                return {symbol: dict(self.open_interest.get(symbol, {}))}
            return {sym: dict(data) for sym, data in self.open_interest.items() if data}
    
    async def get_ticker_24h(self, symbol: Optional[str] = None) -> Dict:
        """
        Get 24h ticker statistics from all exchanges.
        
        Args:
            symbol: Specific symbol or None for all
            
        Returns:
            Dict with volume, high, low, price_change_pct per exchange
        """
        if not self._started:
            await self.start()
        
        async with self._lock:
            if symbol:
                return {symbol: dict(self.ticker_24h.get(symbol, {}))}
            return {sym: dict(data) for sym, data in self.ticker_24h.items() if data}
    
    async def get_candles(self, symbol: Optional[str] = None, exchange: Optional[str] = None, limit: int = 60) -> Dict:
        """
        Get OHLCV candle data from all or specific exchanges.
        
        Args:
            symbol: Specific symbol or None for all
            exchange: Specific exchange or None for all
            limit: Max candles per exchange (default 60 = last hour of 1m candles)
            
        Returns:
            Dict with candle data including OHLCV, volume, and timestamp
        """
        if not self._started:
            await self.start()
        
        async with self._lock:
            result = {}
            symbols = [symbol] if symbol else self.SUPPORTED_SYMBOLS
            
            for sym in symbols:
                sym_data = self.candles.get(sym, {})
                if sym_data:
                    result[sym] = {}
                    for ex, candles in sym_data.items():
                        if exchange and ex != exchange:
                            continue
                        # Return last N candles (already sorted by open_time)
                        result[sym][ex] = candles[-limit:] if len(candles) > limit else candles
            
            return result
    
    async def get_index_prices(self, symbol: Optional[str] = None) -> Dict:
        """
        Get index prices from all exchanges.
        
        Args:
            symbol: Specific symbol or None for all
            
        Returns:
            Dict with index_price and timestamp per exchange
        """
        if not self._started:
            await self.start()
        
        async with self._lock:
            if symbol:
                return {symbol: dict(self.index_prices.get(symbol, {}))}
            return {sym: dict(data) for sym, data in self.index_prices.items() if data}
    
    async def get_market_summary(self, symbol: str) -> Dict:
        """
        Get comprehensive market summary for a symbol across all exchanges.
        
        Args:
            symbol: Symbol to get summary for
            
        Returns:
            Dict with prices, orderbooks, funding, OI, 24h stats combined
        """
        if not self._started:
            await self.start()
        
        async with self._lock:
            return {
                "symbol": symbol,
                "prices": dict(self.prices.get(symbol, {})),
                "orderbooks": dict(self.orderbooks.get(symbol, {})),
                "funding_rates": dict(self.funding_rates.get(symbol, {})),
                "mark_prices": dict(self.mark_prices.get(symbol, {})),
                "open_interest": dict(self.open_interest.get(symbol, {})),
                "ticker_24h": dict(self.ticker_24h.get(symbol, {})),
                "recent_trades_count": {
                    ex: len(trades) for ex, trades in self.trades.get(symbol, {}).items()
                },
                "recent_liquidations": self.liquidations.get(symbol, [])[:5],
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def health_check(self) -> Dict:
        """
        Check connection health.
        
        Returns:
            Dict with health status information
        """
        if not self._started:
            await self.start()
            await asyncio.sleep(2)
        
        async with self._lock:
            connected_exchanges = {k: v for k, v in self.connected_exchanges.items()}
            connected_count = sum(1 for v in connected_exchanges.values() if v)
            
            # Count prices per symbol
            prices_per_symbol = {}
            for sym, sources in self.prices.items():
                active_sources = [ex for ex, data in sources.items() if data.get("price", 0) > 0]
                if active_sources:
                    prices_per_symbol[sym] = active_sources
            
            return {
                "status": "healthy" if connected_count > 0 else "unhealthy",
                "connected": connected_count > 0,
                "mode": "direct_exchange",
                "exchanges": connected_exchanges,
                "connected_count": connected_count,
                "total_exchanges": len(connected_exchanges),
                "symbols_tracked": self.SUPPORTED_SYMBOLS,
                "sources_per_symbol": prices_per_symbol,
                "opportunities_cached": len(self.arbitrage_opportunities),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def close(self):
        """Close all connections."""
        await self.stop()


# ============================================================================
# Singleton Instance
# ============================================================================

_direct_client: Optional[DirectExchangeClient] = None
_client_lock = asyncio.Lock()


def get_direct_client() -> DirectExchangeClient:
    """Get or create the singleton direct client."""
    global _direct_client
    if _direct_client is None:
        _direct_client = DirectExchangeClient()
    return _direct_client


async def reset_direct_client():
    """Reset the singleton client."""
    global _direct_client
    if _direct_client is not None:
        await _direct_client.close()
        _direct_client = None
