"""
WebSocket client for connecting to the Crypto Arbitrage Scanner Go backend.
Provides real-time price data, spreads, and arbitrage opportunities.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, WebSocketException
except ImportError:
    raise ImportError("websockets package required. Install with: pip install websockets")

logger = logging.getLogger(__name__)


class CryptoArbitrageWebSocketClient:
    """
    WebSocket client that connects to the Go arbitrage scanner backend.
    Receives real-time prices, spreads, and arbitrage opportunities.
    """
    
    def __init__(self):
        self.ws_host = os.environ.get("ARBITRAGE_SCANNER_HOST", "localhost")
        self.ws_port = os.environ.get("ARBITRAGE_SCANNER_PORT", "8082")
        self.ws_url = f"ws://{self.ws_host}:{self.ws_port}/ws"
        
        self.websocket = None
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
        # Data stores
        self.latest_prices: Dict[str, Dict[str, Dict]] = {}  # symbol -> source -> price_data
        self.latest_spreads: Dict[str, Dict] = {}  # symbol -> spread_matrix
        self.arbitrage_opportunities: List[Dict] = []
        self.max_opportunities = 100
        
        # Callbacks for real-time updates
        self._on_price_update: Optional[Callable] = None
        self._on_arbitrage: Optional[Callable] = None
        self._on_spreads: Optional[Callable] = None
        
        # Background task
        self._receive_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
    async def connect(self) -> bool:
        """Establish WebSocket connection to the arbitrage scanner."""
        try:
            logger.info(f"Connecting to arbitrage scanner at {self.ws_url}")
            self.websocket = await websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=5
            )
            self.connected = True
            self.reconnect_attempts = 0
            logger.info("Successfully connected to arbitrage scanner")
            
            # Start background receiver
            self._receive_task = asyncio.create_task(self._receive_messages())
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to arbitrage scanner: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Close the WebSocket connection."""
        self.connected = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        logger.info("Disconnected from arbitrage scanner")
    
    async def _receive_messages(self):
        """Background task to receive and process WebSocket messages."""
        while self.connected and self.websocket:
            try:
                message = await self.websocket.recv()
                # Ensure message is a string
                if isinstance(message, bytes):
                    message = message.decode('utf-8')
                await self._process_message(str(message))
            except ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.connected = False
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
    
    async def _ensure_connected(self) -> bool:
        """Ensure we have an active connection."""
        if self.connected and self.websocket:
            return True
        return await self.connect()
    
    async def _process_message(self, raw_message: str):
        """Process incoming WebSocket message."""
        try:
            data = json.loads(raw_message)
            msg_type = data.get("type")
            
            if msg_type == "prices":
                await self._handle_prices(data)
            elif msg_type == "price_update":
                await self._handle_price_update(data)
            elif msg_type == "arbitrage":
                await self._handle_arbitrage(data)
            elif msg_type == "spreads":
                await self._handle_spreads(data)
            else:
                logger.debug(f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
    
    async def _handle_prices(self, data: Dict):
        """Handle batch price updates."""
        async with self._lock:
            prices = data.get("prices", {})
            for symbol, sources in prices.items():
                if symbol not in self.latest_prices:
                    self.latest_prices[symbol] = {}
                for source, price_info in sources.items():
                    if isinstance(price_info, dict):
                        self.latest_prices[symbol][source] = {
                            "price": price_info.get("price", 0),
                            "timestamp": price_info.get("timestamp", 0),
                            "source": source
                        }
                    elif isinstance(price_info, (int, float)):
                        self.latest_prices[symbol][source] = {
                            "price": price_info,
                            "timestamp": int(datetime.now().timestamp() * 1000),
                            "source": source
                        }
        
        if self._on_price_update:
            await self._on_price_update(self.latest_prices)
    
    async def _handle_price_update(self, data: Dict):
        """Handle single price update."""
        symbol = data.get("symbol")
        source = data.get("source")
        price = data.get("price")
        timestamp = data.get("timestamp", int(datetime.now().timestamp() * 1000))
        
        if symbol and source:
            async with self._lock:
                if symbol not in self.latest_prices:
                    self.latest_prices[symbol] = {}
                self.latest_prices[symbol][source] = {
                    "price": price,
                    "timestamp": timestamp,
                    "source": source
                }
    
    async def _handle_arbitrage(self, data: Dict):
        """Handle arbitrage opportunity."""
        opportunity = {
            "symbol": data.get("symbol"),
            "buy_source": data.get("buy_source"),
            "sell_source": data.get("sell_source"),
            "buy_price": data.get("buy_price"),
            "sell_price": data.get("sell_price"),
            "profit_pct": data.get("profit_pct"),
            "timestamp": data.get("timestamp"),
            "detected_at": datetime.utcnow().isoformat()
        }
        
        async with self._lock:
            self.arbitrage_opportunities.insert(0, opportunity)
            if len(self.arbitrage_opportunities) > self.max_opportunities:
                self.arbitrage_opportunities = self.arbitrage_opportunities[:self.max_opportunities]
        
        if self._on_arbitrage:
            await self._on_arbitrage(opportunity)
    
    async def _handle_spreads(self, data: Dict):
        """Handle spreads matrix update."""
        symbol = data.get("symbol")
        spreads = data.get("spreads", {})
        
        if symbol:
            async with self._lock:
                self.latest_spreads[symbol] = {
                    "spreads": spreads,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        if self._on_spreads:
            await self._on_spreads(symbol, spreads)
    
    # ========================================================================
    # Public API methods for MCP tools
    # ========================================================================
    
    async def get_prices_snapshot(self, symbol: Optional[str] = None) -> Dict:
        """Get current prices for all or specific symbol."""
        if not await self._ensure_connected():
            return {}
        
        # Wait briefly for initial data if we just connected
        if not self.latest_prices:
            await asyncio.sleep(1.0)
        
        async with self._lock:
            if symbol:
                return {symbol: self.latest_prices.get(symbol, {})}
            return dict(self.latest_prices)
    
    async def get_spreads_snapshot(self, symbol: str) -> Dict:
        """Get current spread matrix for a symbol."""
        if not await self._ensure_connected():
            return {}
        
        if not self.latest_spreads:
            await asyncio.sleep(1.0)
        
        async with self._lock:
            return dict(self.latest_spreads.get(symbol, {}))
    
    async def get_arbitrage_opportunities(
        self,
        symbol: Optional[str] = None,
        min_profit: float = 0.0,
        limit: int = 20
    ) -> List[Dict]:
        """Get recent arbitrage opportunities with optional filtering."""
        if not await self._ensure_connected():
            return []
        
        if not self.arbitrage_opportunities:
            await asyncio.sleep(1.0)
        
        async with self._lock:
            opportunities = list(self.arbitrage_opportunities)
        
        if symbol:
            opportunities = [o for o in opportunities if o.get("symbol") == symbol]
        
        if min_profit > 0:
            opportunities = [o for o in opportunities if o.get("profit_pct", 0) >= min_profit]
        
        return opportunities[:limit]
    
    async def health_check(self) -> Dict:
        """Check connection health."""
        # Try to connect if not connected
        if not self.connected:
            await self.connect()
            await asyncio.sleep(0.5)
        
        async with self._lock:
            return {
                "status": "healthy" if self.connected else "unhealthy",
                "connected": self.connected,
                "ws_url": self.ws_url,
                "symbols_tracked": list(self.latest_prices.keys()),
                "sources_per_symbol": {
                    sym: list(sources.keys()) 
                    for sym, sources in self.latest_prices.items()
                },
                "opportunities_cached": len(self.arbitrage_opportunities),
                "reconnect_attempts": self.reconnect_attempts,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def close(self):
        """Alias for disconnect."""
        await self.disconnect()


# Singleton instance
_client_instance: Optional[CryptoArbitrageWebSocketClient] = None


def get_arbitrage_client() -> CryptoArbitrageWebSocketClient:
    """Get or create the singleton WebSocket client."""
    global _client_instance
    if _client_instance is None:
        _client_instance = CryptoArbitrageWebSocketClient()
    return _client_instance


async def reset_arbitrage_client():
    """Reset the singleton client (useful for testing)."""
    global _client_instance
    if _client_instance is not None:
        await _client_instance.disconnect()
        _client_instance = None
