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
        # Data stores
        self.prices: Dict[str, Dict[str, Dict]] = {}  # symbol -> exchange -> {price, timestamp}
        self.arbitrage_opportunities: List[Dict] = []
        self.max_opportunities = 100
        
        # Connection tracking
        self.connected_exchanges: Dict[str, bool] = {}
        self._tasks: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        self._running = False
        self._started = False
        
        # Initialize price structure
        for symbol in self.SUPPORTED_SYMBOLS:
            self.prices[symbol] = {}
        
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
        """Connect to Binance Futures WebSocket."""
        streams = [f"{sym.lower()}@bookTicker" for sym in self.SUPPORTED_SYMBOLS]
        url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            self.connected_exchanges["binance_futures"] = True
            logger.info("✓ Connected to Binance Futures")
            
            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    if "data" in data:
                        ticker = data["data"]
                        symbol = ticker.get("s", "").upper()
                        if symbol in self.SUPPORTED_SYMBOLS:
                            bid = float(ticker.get("b", 0))
                            ask = float(ticker.get("a", 0))
                            if bid > 0 and ask > 0:
                                mid_price = (bid + ask) / 2
                                await self._update_price(symbol, "binance_futures", mid_price, bid, ask)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.debug(f"Binance futures parse error: {e}")
    
    async def _connect_binance_spot(self):
        """Connect to Binance Spot WebSocket."""
        streams = [f"{sym.lower()}@bookTicker" for sym in self.SUPPORTED_SYMBOLS]
        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            self.connected_exchanges["binance_spot"] = True
            logger.info("✓ Connected to Binance Spot")
            
            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    if "data" in data:
                        ticker = data["data"]
                        symbol = ticker.get("s", "").upper()
                        if symbol in self.SUPPORTED_SYMBOLS:
                            bid = float(ticker.get("b", 0))
                            ask = float(ticker.get("a", 0))
                            if bid > 0 and ask > 0:
                                mid_price = (bid + ask) / 2
                                await self._update_price(symbol, "binance_spot", mid_price, bid, ask)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.debug(f"Binance spot parse error: {e}")
    
    async def _connect_bybit_futures(self):
        """Connect to Bybit Futures WebSocket."""
        url = "wss://stream.bybit.com/v5/public/linear"
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            # Subscribe to tickers (more reliable than orderbook)
            subscribe_msg = {
                "op": "subscribe",
                "args": [f"tickers.{sym}" for sym in self.SUPPORTED_SYMBOLS]
            }
            await ws.send(json.dumps(subscribe_msg))
            
            self.connected_exchanges["bybit_futures"] = True
            logger.info("✓ Connected to Bybit Futures")
            
            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    topic = data.get("topic", "")
                    
                    if topic.startswith("tickers."):
                        ticker_data = data.get("data", {})
                        symbol = ticker_data.get("symbol", "").upper()
                        if symbol in self.SUPPORTED_SYMBOLS:
                            bid = float(ticker_data.get("bid1Price", 0) or 0)
                            ask = float(ticker_data.get("ask1Price", 0) or 0)
                            last = float(ticker_data.get("lastPrice", 0) or 0)
                            
                            if bid > 0 and ask > 0:
                                mid_price = (bid + ask) / 2
                            elif last > 0:
                                mid_price = last
                                bid = ask = last
                            else:
                                continue
                                
                            await self._update_price(symbol, "bybit_futures", mid_price, bid, ask)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.debug(f"Bybit parse error: {e}")
    
    async def _connect_okx_futures(self):
        """Connect to OKX Futures WebSocket."""
        url = "wss://ws.okx.com:8443/ws/v5/public"
        
        # OKX uses different symbol format
        okx_symbols = {
            "BTCUSDT": "BTC-USDT-SWAP",
            "ETHUSDT": "ETH-USDT-SWAP", 
            "SOLUSDT": "SOL-USDT-SWAP",
            "XRPUSDT": "XRP-USDT-SWAP"
        }
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            # Subscribe to tickers
            subscribe_msg = {
                "op": "subscribe",
                "args": [{"channel": "tickers", "instId": inst} for inst in okx_symbols.values()]
            }
            await ws.send(json.dumps(subscribe_msg))
            
            self.connected_exchanges["okx_futures"] = True
            logger.info("✓ Connected to OKX Futures")
            
            # Reverse mapping for symbol lookup
            reverse_symbols = {v: k for k, v in okx_symbols.items()}
            
            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    
                    if "data" in data and data.get("arg", {}).get("channel") == "tickers":
                        for ticker in data["data"]:
                            inst_id = ticker.get("instId", "")
                            symbol = reverse_symbols.get(inst_id)
                            
                            if symbol:
                                bid = float(ticker.get("bidPx", 0) or 0)
                                ask = float(ticker.get("askPx", 0) or 0)
                                last = float(ticker.get("last", 0) or 0)
                                
                                if bid > 0 and ask > 0:
                                    mid_price = (bid + ask) / 2
                                elif last > 0:
                                    mid_price = last
                                    bid = ask = last
                                else:
                                    continue
                                    
                                await self._update_price(symbol, "okx_futures", mid_price, bid, ask)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.debug(f"OKX parse error: {e}")
    
    async def _connect_bybit_spot(self):
        """Connect to Bybit Spot WebSocket."""
        url = "wss://stream.bybit.com/v5/public/spot"
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            # Subscribe to tickers
            subscribe_msg = {
                "op": "subscribe",
                "args": [f"tickers.{sym}" for sym in self.SUPPORTED_SYMBOLS]
            }
            await ws.send(json.dumps(subscribe_msg))
            
            self.connected_exchanges["bybit_spot"] = True
            logger.info("✓ Connected to Bybit Spot")
            
            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    topic = data.get("topic", "")
                    
                    if topic.startswith("tickers."):
                        ticker_data = data.get("data", {})
                        symbol = ticker_data.get("symbol", "").upper()
                        if symbol in self.SUPPORTED_SYMBOLS:
                            bid = float(ticker_data.get("bid1Price", 0) or 0)
                            ask = float(ticker_data.get("ask1Price", 0) or 0)
                            last = float(ticker_data.get("lastPrice", 0) or 0)
                            
                            if bid > 0 and ask > 0:
                                mid_price = (bid + ask) / 2
                            elif last > 0:
                                mid_price = last
                                bid = ask = last
                            else:
                                continue
                                
                            await self._update_price(symbol, "bybit_spot", mid_price, bid, ask)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.debug(f"Bybit spot parse error: {e}")
    
    async def _connect_kraken_futures(self):
        """Connect to Kraken Futures WebSocket."""
        url = "wss://futures.kraken.com/ws/v1"
        
        # Kraken uses different symbol format (PI_BTCUSD for perpetual)
        kraken_symbols = {
            "BTCUSDT": "PF_BTCUSD",  # PF = Perpetual Futures
            "ETHUSDT": "PF_ETHUSD",
            "SOLUSDT": "PF_SOLUSD",
            "XRPUSDT": "PF_XRPUSD"
        }
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            # Subscribe to ticker
            subscribe_msg = {
                "event": "subscribe",
                "feed": "ticker",
                "product_ids": list(kraken_symbols.values())
            }
            await ws.send(json.dumps(subscribe_msg))
            
            self.connected_exchanges["kraken_futures"] = True
            logger.info("✓ Connected to Kraken Futures")
            
            # Reverse mapping
            reverse_symbols = {v: k for k, v in kraken_symbols.items()}
            
            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    
                    if data.get("feed") == "ticker":
                        product_id = data.get("product_id", "")
                        symbol = reverse_symbols.get(product_id)
                        
                        if symbol:
                            bid = float(data.get("bid", 0) or 0)
                            ask = float(data.get("ask", 0) or 0)
                            last = float(data.get("last", 0) or 0)
                            
                            if bid > 0 and ask > 0:
                                mid_price = (bid + ask) / 2
                            elif last > 0:
                                mid_price = last
                                bid = ask = last
                            else:
                                continue
                                
                            await self._update_price(symbol, "kraken_futures", mid_price, bid, ask)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.debug(f"Kraken parse error: {e}")
    
    async def _connect_gate_futures(self):
        """Connect to Gate.io Futures WebSocket."""
        url = "wss://fx-ws.gateio.ws/v4/ws/usdt"
        
        # Gate uses format like "BTC_USDT"
        gate_symbols = {
            "BTCUSDT": "BTC_USDT",
            "ETHUSDT": "ETH_USDT",
            "SOLUSDT": "SOL_USDT",
            "XRPUSDT": "XRP_USDT"
        }
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            # Subscribe to tickers
            subscribe_msg = {
                "time": int(time.time()),
                "channel": "futures.tickers",
                "event": "subscribe",
                "payload": list(gate_symbols.values())
            }
            await ws.send(json.dumps(subscribe_msg))
            
            self.connected_exchanges["gate_futures"] = True
            logger.info("✓ Connected to Gate.io Futures")
            
            # Reverse mapping
            reverse_symbols = {v: k for k, v in gate_symbols.items()}
            
            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    
                    if data.get("channel") == "futures.tickers" and data.get("event") == "update":
                        result = data.get("result", [])
                        for ticker in result if isinstance(result, list) else [result]:
                            contract = ticker.get("contract", "")
                            symbol = reverse_symbols.get(contract)
                            
                            if symbol:
                                last = float(ticker.get("last", 0) or 0)
                                mark = float(ticker.get("mark_price", 0) or 0)
                                
                                if last > 0:
                                    # Gate doesn't provide bid/ask in ticker, use last price
                                    await self._update_price(symbol, "gate_futures", last, last, last)
                                elif mark > 0:
                                    await self._update_price(symbol, "gate_futures", mark, mark, mark)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.debug(f"Gate parse error: {e}")
    
    async def _connect_hyperliquid(self):
        """Connect to Hyperliquid WebSocket."""
        url = "wss://api.hyperliquid.xyz/ws"
        
        # Hyperliquid uses simple symbols like "BTC", "ETH"
        hl_symbols = {
            "BTCUSDT": "BTC",
            "ETHUSDT": "ETH",
            "SOLUSDT": "SOL",
            "XRPUSDT": "XRP"
        }
        
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            # Subscribe to all mids (mid prices)
            subscribe_msg = {
                "method": "subscribe",
                "subscription": {"type": "allMids"}
            }
            await ws.send(json.dumps(subscribe_msg))
            
            self.connected_exchanges["hyperliquid_futures"] = True
            logger.info("✓ Connected to Hyperliquid")
            
            # Reverse mapping
            reverse_symbols = {v: k for k, v in hl_symbols.items()}
            
            async for message in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(message)
                    
                    if data.get("channel") == "allMids":
                        mids = data.get("data", {}).get("mids", {})
                        
                        for hl_sym, our_sym in reverse_symbols.items():
                            if hl_sym in mids:
                                mid_price = float(mids[hl_sym])
                                if mid_price > 0:
                                    await self._update_price(our_sym, "hyperliquid_futures", mid_price, mid_price, mid_price)
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
