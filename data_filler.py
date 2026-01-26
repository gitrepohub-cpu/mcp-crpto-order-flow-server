"""
🎯 TARGETED DATA FILLER
=======================

This script fills the remaining gaps in data collection:
1. Populate symbol-specific liquidation tables from binance_all_liquidations
2. Add missing candles via REST API polling
3. Fill funding rates, open interest, and mark prices

Run alongside the enhanced_collector_100.py for best results.
"""

import asyncio
import logging
import time
import aiohttp
import duckdb
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

RAW_DB_PATH = Path("data/isolated_exchange_data.duckdb")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ARUSDT", 
           "BRETTUSDT", "POPCATUSDT", "WIFUSDT", "PNUTUSDT"]

# REST API endpoints for additional data
ADDITIONAL_ENDPOINTS = {
    # Binance funding rates
    'binance_funding': {
        'url': 'https://fapi.binance.com/fapi/v1/premiumIndex',
        'params': lambda s: {'symbol': s},
        'type': 'funding'
    },
    # Binance open interest
    'binance_oi': {
        'url': 'https://fapi.binance.com/fapi/v1/openInterest',
        'params': lambda s: {'symbol': s},
        'type': 'open_interest'
    },
    # Binance mark price
    'binance_mark': {
        'url': 'https://fapi.binance.com/fapi/v1/premiumIndex',
        'params': lambda s: {'symbol': s},
        'type': 'mark_price'
    },
    # Bybit funding rates
    'bybit_funding': {
        'url': 'https://api.bybit.com/v5/market/tickers',
        'params': lambda s: {'category': 'linear', 'symbol': s},
        'type': 'funding'
    },
    # OKX funding rates
    'okx_funding': {
        'url': 'https://www.okx.com/api/v5/public/funding-rate',
        'params': lambda s: {'instId': f'{s[:-4]}-USDT-SWAP'},
        'type': 'funding'
    },
    # Gate.io funding rates
    'gateio_funding': {
        'url': 'https://api.gateio.ws/api/v4/futures/usdt/contracts',
        'params': lambda s: None,  # Returns all contracts
        'type': 'funding_all'
    },
}


class DataFiller:
    """Fills gaps in data collection via REST APIs."""
    
    def __init__(self):
        self.conn: duckdb.DuckDBPyConnection = None
        self.http_session: aiohttp.ClientSession = None
        self.running = False
        self.stats = {
            'liquidations_distributed': 0,
            'funding_rates_added': 0,
            'open_interest_added': 0,
            'candles_added': 0,
            'errors': 0
        }
        self.id_counters: Dict[str, int] = {}
        
    def _get_next_id(self, table_name: str) -> int:
        """Get next ID for a table."""
        if table_name not in self.id_counters:
            try:
                result = self.conn.execute(f"SELECT MAX(id) FROM {table_name}").fetchone()
                self.id_counters[table_name] = (result[0] or 0)
            except:
                self.id_counters[table_name] = 0
        self.id_counters[table_name] += 1
        return self.id_counters[table_name]
        
    async def run(self, duration_seconds: int = 300):
        """Run the data filler for a specified duration."""
        print(f"""
===============================================================================
                         TARGETED DATA FILLER
===============================================================================

  This fills gaps by:
  [+] Distributing liquidations to symbol-specific tables
  [+] Fetching funding rates from all exchanges
  [+] Fetching open interest from all exchanges
  [+] Additional candle polling

  Running for {duration_seconds} seconds...
===============================================================================
""")
        
        self.running = True
        self.conn = duckdb.connect(str(RAW_DB_PATH))
        self.http_session = aiohttp.ClientSession()
        
        start_time = time.time()
        
        try:
            while self.running and (time.time() - start_time) < duration_seconds:
                # Run data collection tasks
                await self._distribute_liquidations()
                await self._fetch_funding_rates()
                await self._fetch_open_interest()
                await self._fetch_additional_candles()
                
                # Print status
                elapsed = int(time.time() - start_time)
                print(f"\r  Progress: {elapsed}/{duration_seconds}s | "
                      f"Liqs: {self.stats['liquidations_distributed']} | "
                      f"Funding: {self.stats['funding_rates_added']} | "
                      f"OI: {self.stats['open_interest_added']} | "
                      f"Candles: {self.stats['candles_added']}    ", end='', flush=True)
                
                await asyncio.sleep(10)  # Poll every 10 seconds
                
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        finally:
            self.running = False
            await self.http_session.close()
            self.conn.close()
            
        print(f"\n\nFinal stats: {self.stats}")
            
    async def _distribute_liquidations(self):
        """Distribute liquidations from combined table to symbol-specific tables."""
        try:
            # Check if combined table exists
            tables = [t[0] for t in self.conn.execute("SHOW TABLES").fetchall()]
            if 'binance_all_liquidations' not in tables:
                return
                
            # Get recent liquidations for our symbols
            for symbol in SYMBOLS:
                # Check if there are any liquidations for this symbol
                result = self.conn.execute(f"""
                    SELECT * FROM binance_all_liquidations 
                    WHERE symbol = '{symbol}'
                    ORDER BY id DESC 
                    LIMIT 50
                """).fetchall()
                
                if not result:
                    continue
                    
                # Create/populate symbol-specific table
                table_name = f"{symbol.lower()}_binance_futures_liquidations"
                
                self.conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id BIGINT PRIMARY KEY,
                        timestamp TIMESTAMP,
                        symbol VARCHAR,
                        side VARCHAR,
                        price DOUBLE,
                        quantity DOUBLE,
                        quote_quantity DOUBLE,
                        time_in_force VARCHAR,
                        order_type VARCHAR
                    )
                """)
                
                for row in result:
                    try:
                        next_id = self._get_next_id(table_name)
                        self.conn.execute(f"""
                            INSERT OR IGNORE INTO {table_name} 
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, [
                            next_id,
                            row[1],  # timestamp
                            row[3],  # symbol
                            row[4],  # side
                            row[8],  # price
                            row[7],  # original_qty
                            float(row[8]) * float(row[7]),  # quote_quantity
                            row[6],  # time_in_force
                            row[5],  # order_type
                        ])
                        self.stats['liquidations_distributed'] += 1
                    except Exception as e:
                        logger.debug(f"Liquidation insert error: {e}")
                        
        except Exception as e:
            logger.debug(f"Distribute liquidations error: {e}")
            
    async def _fetch_funding_rates(self):
        """Fetch funding rates from exchanges."""
        try:
            for symbol in SYMBOLS:
                # Binance funding rate
                await self._fetch_binance_funding(symbol)
                # Bybit funding rate
                await self._fetch_bybit_funding(symbol)
                # OKX funding rate
                await self._fetch_okx_funding(symbol)
                
                await asyncio.sleep(0.1)  # Rate limit
                
        except Exception as e:
            logger.debug(f"Funding rates error: {e}")
            
    async def _fetch_binance_funding(self, symbol: str):
        """Fetch Binance funding rate."""
        try:
            url = 'https://fapi.binance.com/fapi/v1/premiumIndex'
            async with self.http_session.get(url, params={'symbol': symbol}, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    await self._store_funding_rate('binance', 'futures', symbol, {
                        'funding_rate': float(data.get('lastFundingRate', 0)),
                        'mark_price': float(data.get('markPrice', 0)),
                        'index_price': float(data.get('indexPrice', 0)),
                        'next_funding_time': data.get('nextFundingTime', 0),
                    })
        except Exception as e:
            logger.debug(f"Binance funding error: {e}")
            
    async def _fetch_bybit_funding(self, symbol: str):
        """Fetch Bybit funding rate."""
        try:
            url = 'https://api.bybit.com/v5/market/tickers'
            async with self.http_session.get(url, params={'category': 'linear', 'symbol': symbol}, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('result', {}).get('list'):
                        item = data['result']['list'][0]
                        await self._store_funding_rate('bybit', 'futures', symbol, {
                            'funding_rate': float(item.get('fundingRate', 0)),
                            'mark_price': float(item.get('markPrice', 0)),
                            'index_price': float(item.get('indexPrice', 0)),
                            'next_funding_time': 0,
                        })
        except Exception as e:
            logger.debug(f"Bybit funding error: {e}")
            
    async def _fetch_okx_funding(self, symbol: str):
        """Fetch OKX funding rate."""
        try:
            inst_id = f'{symbol[:-4]}-USDT-SWAP'
            url = 'https://www.okx.com/api/v5/public/funding-rate'
            async with self.http_session.get(url, params={'instId': inst_id}, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('data'):
                        item = data['data'][0]
                        await self._store_funding_rate('okx', 'futures', symbol, {
                            'funding_rate': float(item.get('fundingRate', 0)),
                            'mark_price': 0,
                            'index_price': 0,
                            'next_funding_time': int(item.get('nextFundingTime', 0)),
                        })
        except Exception as e:
            logger.debug(f"OKX funding error: {e}")
            
    async def _store_funding_rate(self, exchange: str, market: str, symbol: str, data: dict):
        """Store funding rate data."""
        table_name = f"{symbol.lower()}_{exchange}_{market}_funding_rates"
        
        try:
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    funding_rate DOUBLE,
                    mark_price DOUBLE,
                    index_price DOUBLE,
                    next_funding_time BIGINT
                )
            """)
            
            next_id = self._get_next_id(table_name)
            self.conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                data['funding_rate'],
                data['mark_price'],
                data['index_price'],
                data['next_funding_time'],
            ])
            
            self.stats['funding_rates_added'] += 1
            
        except Exception as e:
            logger.debug(f"Store funding rate error: {e}")
            
    async def _fetch_open_interest(self):
        """Fetch open interest from exchanges."""
        try:
            for symbol in SYMBOLS:
                # Binance OI
                await self._fetch_binance_oi(symbol)
                # Bybit OI
                await self._fetch_bybit_oi(symbol)
                
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.debug(f"Open interest error: {e}")
            
    async def _fetch_binance_oi(self, symbol: str):
        """Fetch Binance open interest."""
        try:
            url = 'https://fapi.binance.com/fapi/v1/openInterest'
            async with self.http_session.get(url, params={'symbol': symbol}, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    await self._store_open_interest('binance', 'futures', symbol, {
                        'open_interest': float(data.get('openInterest', 0)),
                    })
        except Exception as e:
            logger.debug(f"Binance OI error: {e}")
            
    async def _fetch_bybit_oi(self, symbol: str):
        """Fetch Bybit open interest."""
        try:
            url = 'https://api.bybit.com/v5/market/open-interest'
            async with self.http_session.get(url, params={'category': 'linear', 'symbol': symbol, 'intervalTime': '5min', 'limit': 1}, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('result', {}).get('list'):
                        item = data['result']['list'][0]
                        await self._store_open_interest('bybit', 'futures', symbol, {
                            'open_interest': float(item.get('openInterest', 0)),
                        })
        except Exception as e:
            logger.debug(f"Bybit OI error: {e}")
            
    async def _store_open_interest(self, exchange: str, market: str, symbol: str, data: dict):
        """Store open interest data."""
        table_name = f"{symbol.lower()}_{exchange}_{market}_open_interest"
        
        try:
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    open_interest DOUBLE,
                    open_interest_value DOUBLE
                )
            """)
            
            next_id = self._get_next_id(table_name)
            self.conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                data['open_interest'],
                0,
            ])
            
            self.stats['open_interest_added'] += 1
            
        except Exception as e:
            logger.debug(f"Store OI error: {e}")
            
    async def _fetch_additional_candles(self):
        """Fetch candles from exchanges that might be missing."""
        exchanges = [
            ('binance_futures', 'https://fapi.binance.com/fapi/v1/klines', lambda s: {'symbol': s, 'interval': '1m', 'limit': 3}),
            ('binance_spot', 'https://api.binance.com/api/v3/klines', lambda s: {'symbol': s, 'interval': '1m', 'limit': 3}),
            ('bybit_futures', 'https://api.bybit.com/v5/market/kline', lambda s: {'category': 'linear', 'symbol': s, 'interval': '1', 'limit': 3}),
            ('bybit_spot', 'https://api.bybit.com/v5/market/kline', lambda s: {'category': 'spot', 'symbol': s, 'interval': '1', 'limit': 3}),
        ]
        
        try:
            for symbol in SYMBOLS:
                for exchange_name, url, params_fn in exchanges:
                    try:
                        async with self.http_session.get(url, params=params_fn(symbol), timeout=10) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                await self._store_candle(exchange_name, symbol, data)
                    except:
                        pass
                        
                await asyncio.sleep(0.2)
                
        except Exception as e:
            logger.debug(f"Additional candles error: {e}")
            
    async def _store_candle(self, exchange: str, symbol: str, data: Any):
        """Store candle data."""
        table_name = f"{symbol.lower()}_{exchange}_candles"
        
        try:
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    open_time BIGINT,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    close_time BIGINT,
                    quote_volume DOUBLE,
                    trades INTEGER
                )
            """)
            
            # Parse based on exchange format
            candles = []
            if 'binance' in exchange:
                candles = data
            elif 'bybit' in exchange:
                candles = data.get('result', {}).get('list', [])
                
            for c in candles[-1:]:  # Only store latest
                next_id = self._get_next_id(table_name)
                
                if 'binance' in exchange:
                    self.conn.execute(f"""
                        INSERT OR IGNORE INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        next_id,
                        datetime.now(timezone.utc),
                        c[0],
                        float(c[1]),
                        float(c[2]),
                        float(c[3]),
                        float(c[4]),
                        float(c[5]),
                        c[6],
                        float(c[7]),
                        int(c[8]),
                    ])
                else:  # Bybit
                    self.conn.execute(f"""
                        INSERT OR IGNORE INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        next_id,
                        datetime.now(timezone.utc),
                        int(c[0]),
                        float(c[1]),
                        float(c[2]),
                        float(c[3]),
                        float(c[4]),
                        float(c[5]),
                        int(c[0]) + 60000,
                        float(c[6]) if len(c) > 6 else 0,
                        0,
                    ])
                    
                self.stats['candles_added'] += 1
                
        except Exception as e:
            logger.debug(f"Store candle error: {e}")


async def main():
    filler = DataFiller()
    await filler.run(duration_seconds=120)  # Run for 2 minutes


if __name__ == "__main__":
    asyncio.run(main())
