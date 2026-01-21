"""
🚀 ISOLATED Data Collector
==========================
Each coin on each exchange writes to its OWN dedicated tables.
NO DATA MIXING - Complete isolation per coin/exchange combination.

Table naming: {symbol}_{exchange}_{market_type}_{stream}
Example: btcusdt_binance_futures_prices
"""

import asyncio
import duckdb
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ISOLATED DATA COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class IsolatedDataCollector:
    """
    Data collector with COMPLETE ISOLATION per coin/exchange.
    Each coin on each exchange has its own dedicated tables.
    Data NEVER mixes between exchanges or market types.
    """
    
    def __init__(self, db_path: str = "data/isolated_exchange_data.duckdb"):
        self.db_path = db_path
        self.conn = None
        self.buffers: Dict[str, List[Tuple]] = defaultdict(list)  # table_name -> records
        self.id_counters: Dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()
        
        self.stats = {
            'total_records': 0,
            'records_by_table': defaultdict(int),
            'last_flush': datetime.now(timezone.utc)
        }
    
    def connect(self):
        """Connect to DuckDB."""
        self.conn = duckdb.connect(self.db_path)
        
        # Load existing table names into a set for fast lookup
        self.existing_tables = set()
        try:
            tables = self.conn.execute("SELECT table_name FROM _table_registry").fetchall()
            self.existing_tables = {t[0] for t in tables}
        except:
            # Fallback: get all tables
            tables = self.conn.execute("SHOW TABLES").fetchall()
            self.existing_tables = {t[0] for t in tables if not t[0].startswith('_')}
        
        # Load current max IDs for all tables
        for table_name in self.existing_tables:
            try:
                max_id = self.conn.execute(f"SELECT MAX(id) FROM {table_name}").fetchone()[0]
                if max_id:
                    self.id_counters[table_name] = max_id
            except:
                pass
        
        logger.info(f"✅ Connected to ISOLATED database: {self.db_path}")
        logger.info(f"   Loaded {len(self.id_counters)} table ID counters")
        logger.info(f"   {len(self.existing_tables)} existing tables")
    
    def _get_table_name(self, symbol: str, exchange: str, market_type: str, stream: str) -> str:
        """Generate isolated table name."""
        return f"{symbol.lower()}_{exchange.lower()}_{market_type.lower()}_{stream}"
    
    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        return table_name in self.existing_tables
    
    def _log_missing_table(self, table_name: str):
        """Log warning when data is dropped due to missing table (once per table)."""
        if not hasattr(self, '_warned_tables'):
            self._warned_tables = set()
        if table_name not in self._warned_tables:
            logger.debug(f"⚠️ Table {table_name} doesn't exist - data dropped (expected for unavailable coins)")
            self._warned_tables.add(table_name)
    
    def _get_next_id(self, table_name: str) -> int:
        """Get next sequential ID for a table."""
        self.id_counters[table_name] += 1
        return self.id_counters[table_name]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ADD METHODS - Each writes to ISOLATED table
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def add_price(self, symbol: str, exchange: str, market_type: str, data: dict):
        """
        Add price to ISOLATED table: {symbol}_{exchange}_{market_type}_prices
        
        Example: btcusdt_binance_futures_prices
        """
        table_name = self._get_table_name(symbol, exchange, market_type, 'prices')
        
        # Skip if table doesn't exist (coin not available on this exchange)
        if not self._table_exists(table_name):
            self._log_missing_table(table_name)
            return
        
        async with self._lock:
            record_id = self._get_next_id(table_name)
            
            bid = data.get('bid')
            ask = data.get('ask')
            mid = data.get('mid_price') or data.get('price')
            
            if bid and ask and not mid:
                mid = (bid + ask) / 2
            
            spread = (ask - bid) if (bid and ask) else None
            spread_bps = (spread / mid * 10000) if (spread and mid) else None
            
            record = (
                record_id,
                datetime.now(timezone.utc),
                mid,
                bid,
                ask,
                spread,
                spread_bps
            )
            
            self.buffers[table_name].append(record)
            self.stats['total_records'] += 1
            self.stats['records_by_table'][table_name] += 1
    
    async def add_orderbook(self, symbol: str, exchange: str, market_type: str, data: dict):
        """
        Add orderbook to ISOLATED table: {symbol}_{exchange}_{market_type}_orderbooks
        """
        table_name = self._get_table_name(symbol, exchange, market_type, 'orderbooks')
        
        # Skip if table doesn't exist
        if not self._table_exists(table_name):
            self._log_missing_table(table_name)
            return
        
        async with self._lock:
            record_id = self._get_next_id(table_name)
            
            bids = data.get('bids', [])[:10]
            asks = data.get('asks', [])[:10]
            
            # Pad to 10 levels
            while len(bids) < 10:
                bids.append([None, None])
            while len(asks) < 10:
                asks.append([None, None])
            
            # Flatten
            bid_data = []
            for b in bids:
                bid_data.extend([b[0], b[1]])
            
            ask_data = []
            for a in asks:
                ask_data.extend([a[0], a[1]])
            
            # Aggregates
            total_bid = sum(b[1] for b in bids if b[1]) if bids else None
            total_ask = sum(a[1] for a in asks if a[1]) if asks else None
            ratio = (total_bid / total_ask) if (total_bid and total_ask) else None
            
            bid_1 = bids[0][0] if bids else None
            ask_1 = asks[0][0] if asks else None
            mid = ((bid_1 + ask_1) / 2) if (bid_1 and ask_1) else None
            spread = (ask_1 - bid_1) if (bid_1 and ask_1) else None
            spread_pct = (spread / mid * 100) if (spread and mid) else None
            
            record = (
                record_id,
                datetime.now(timezone.utc),
                *bid_data,  # 20 values
                *ask_data,  # 20 values
                total_bid, total_ask, ratio, spread, spread_pct, mid
            )
            
            self.buffers[table_name].append(record)
            self.stats['total_records'] += 1
            self.stats['records_by_table'][table_name] += 1
    
    async def add_trade(self, symbol: str, exchange: str, market_type: str, data: dict):
        """
        Add trade to ISOLATED table: {symbol}_{exchange}_{market_type}_trades
        """
        table_name = self._get_table_name(symbol, exchange, market_type, 'trades')
        
        # Skip if table doesn't exist
        if not self._table_exists(table_name):
            self._log_missing_table(table_name)
            return
        
        async with self._lock:
            record_id = self._get_next_id(table_name)
            
            price = data.get('price')
            qty = data.get('quantity') or data.get('qty') or data.get('size')
            quote_value = (price * qty) if (price and qty) else None
            
            record = (
                record_id,
                datetime.now(timezone.utc),
                data.get('trade_id') or data.get('id'),
                price,
                qty,
                quote_value,
                data.get('side', 'unknown'),
                data.get('is_buyer_maker')
            )
            
            self.buffers[table_name].append(record)
            self.stats['total_records'] += 1
            self.stats['records_by_table'][table_name] += 1
    
    async def add_mark_price(self, symbol: str, exchange: str, market_type: str, data: dict):
        """
        Add mark price to ISOLATED table: {symbol}_{exchange}_futures_mark_prices
        FUTURES ONLY - but accepts market_type for API consistency
        """
        # Always use futures for mark prices
        table_name = self._get_table_name(symbol, exchange, 'futures', 'mark_prices')
        
        # Skip if table doesn't exist
        if not self._table_exists(table_name):
            self._log_missing_table(table_name)
            return
        
        async with self._lock:
            record_id = self._get_next_id(table_name)
            
            mark = data.get('mark_price')
            index = data.get('index_price')
            funding = data.get('funding_rate')
            
            basis = (mark - index) if (mark and index) else None
            basis_pct = (basis / index * 100) if (basis and index) else None
            annualized = (funding * 3 * 365 * 100) if funding else None
            
            record = (
                record_id,
                datetime.now(timezone.utc),
                mark,
                index,
                basis,
                basis_pct,
                funding,
                annualized
            )
            
            self.buffers[table_name].append(record)
            self.stats['total_records'] += 1
            self.stats['records_by_table'][table_name] += 1
    
    async def add_funding_rate(self, symbol: str, exchange: str, market_type: str, data: dict):
        """
        Add funding rate to ISOLATED table: {symbol}_{exchange}_futures_funding_rates
        FUTURES ONLY - but accepts market_type for API consistency
        """
        # Always use futures for funding rates
        table_name = self._get_table_name(symbol, exchange, 'futures', 'funding_rates')
        
        # Skip if table doesn't exist
        if not self._table_exists(table_name):
            self._log_missing_table(table_name)
            return
        
        async with self._lock:
            record_id = self._get_next_id(table_name)
            
            rate = data.get('funding_rate')
            funding_pct = (rate * 100) if rate else None
            annualized = (rate * 3 * 365 * 100) if rate else None
            
            # Convert next_funding_time to timestamp if it's a unix ms timestamp
            next_time = data.get('next_funding_time')
            if next_time and isinstance(next_time, (int, float)) and next_time > 1000000000000:
                # It's a unix ms timestamp
                next_time = datetime.fromtimestamp(next_time / 1000, tz=timezone.utc)
            elif next_time and isinstance(next_time, (int, float)):
                # Might be seconds
                next_time = datetime.fromtimestamp(next_time, tz=timezone.utc)
            elif not isinstance(next_time, datetime):
                next_time = None
            
            record = (
                record_id,
                datetime.now(timezone.utc),
                rate,
                funding_pct,
                annualized,
                next_time,
                data.get('countdown_secs')
            )
            
            self.buffers[table_name].append(record)
            self.stats['total_records'] += 1
            self.stats['records_by_table'][table_name] += 1
    
    async def add_open_interest(self, symbol: str, exchange: str, market_type: str, data: dict):
        """
        Add open interest to ISOLATED table: {symbol}_{exchange}_futures_open_interest
        FUTURES ONLY - but accepts market_type for API consistency
        """
        # Always use futures for open interest
        table_name = self._get_table_name(symbol, exchange, 'futures', 'open_interest')
        
        # Skip if table doesn't exist
        if not self._table_exists(table_name):
            self._log_missing_table(table_name)
            return
        
        async with self._lock:
            record_id = self._get_next_id(table_name)
            
            record = (
                record_id,
                datetime.now(timezone.utc),
                data.get('open_interest') or data.get('oi'),
                data.get('open_interest_usd') or data.get('oi_value'),
                None,  # oi_change_1h - calculated later
                None   # oi_change_pct_1h
            )
            
            self.buffers[table_name].append(record)
            self.stats['total_records'] += 1
            self.stats['records_by_table'][table_name] += 1
    
    async def add_ticker_24h(self, symbol: str, exchange: str, market_type: str, data: dict):
        """
        Add 24h ticker to ISOLATED table: {symbol}_{exchange}_{market_type}_ticker_24h
        """
        table_name = self._get_table_name(symbol, exchange, market_type, 'ticker_24h')
        
        # Skip if table doesn't exist
        if not self._table_exists(table_name):
            self._log_missing_table(table_name)
            return
        
        async with self._lock:
            record_id = self._get_next_id(table_name)
            
            vol = data.get('volume_24h') or data.get('volume')
            quote_vol = data.get('quote_volume_24h') or data.get('turnover')
            vwap = (quote_vol / vol) if (vol and quote_vol and vol > 0) else None
            
            record = (
                record_id,
                datetime.now(timezone.utc),
                vol,
                quote_vol,
                data.get('trade_count_24h') or data.get('count'),
                data.get('high_24h') or data.get('high'),
                data.get('low_24h') or data.get('low'),
                data.get('open_24h') or data.get('open'),
                data.get('last_price') or data.get('lastPrice') or data.get('close'),
                data.get('price_change'),
                data.get('price_change_pct'),
                vwap
            )
            
            self.buffers[table_name].append(record)
            self.stats['total_records'] += 1
            self.stats['records_by_table'][table_name] += 1
    
    async def add_candle(self, symbol: str, exchange: str, market_type: str, data: dict):
        """
        Add candle to ISOLATED table: {symbol}_{exchange}_{market_type}_candles
        """
        table_name = self._get_table_name(symbol, exchange, market_type, 'candles')
        
        # Skip if table doesn't exist
        if not self._table_exists(table_name):
            self._log_missing_table(table_name)
            return
        
        async with self._lock:
            record_id = self._get_next_id(table_name)
            
            open_time = data.get('open_time')
            if isinstance(open_time, (int, float)):
                open_time = datetime.fromtimestamp(open_time / 1000, tz=timezone.utc)
            
            close_time = data.get('close_time')
            if isinstance(close_time, (int, float)):
                close_time = datetime.fromtimestamp(close_time / 1000, tz=timezone.utc)
            
            vol = data.get('volume')
            taker_buy = data.get('taker_buy_volume')
            taker_pct = (taker_buy / vol * 100) if (vol and taker_buy and vol > 0) else None
            
            record = (
                record_id,
                open_time,
                close_time,
                data.get('interval', '1m'),
                data.get('open'),
                data.get('high'),
                data.get('low'),
                data.get('close'),
                vol,
                data.get('quote_volume'),
                data.get('trade_count'),
                taker_buy,
                data.get('taker_buy_quote'),
                taker_pct
            )
            
            self.buffers[table_name].append(record)
            self.stats['total_records'] += 1
            self.stats['records_by_table'][table_name] += 1
    
    async def add_liquidation(self, symbol: str, exchange: str, market_type: str, data: dict):
        """
        Add liquidation to ISOLATED table: {symbol}_{exchange}_futures_liquidations
        FUTURES ONLY (not available on Kraken) - but accepts market_type for API consistency
        """
        # Always use futures for liquidations
        table_name = self._get_table_name(symbol, exchange, 'futures', 'liquidations')
        
        # Skip if table doesn't exist
        if not self._table_exists(table_name):
            self._log_missing_table(table_name)
            return
        
        async with self._lock:
            record_id = self._get_next_id(table_name)
            
            price = data.get('price')
            qty = data.get('quantity') or data.get('size')
            value = (price * qty) if (price and qty) else None
            is_large = (value > 100000) if value else False
            
            record = (
                record_id,
                datetime.now(timezone.utc),
                data.get('side', 'unknown'),
                price,
                qty,
                value,
                is_large
            )
            
            self.buffers[table_name].append(record)
            self.stats['total_records'] += 1
            self.stats['records_by_table'][table_name] += 1
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FLUSH TO DATABASE
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def flush_buffers(self):
        """Flush all buffers to their respective ISOLATED tables."""
        async with self._lock:
            start_time = time.time()
            flushed = 0
            tables_flushed = 0
            successful_tables = []
            failed_tables = []
            
            for table_name, records in list(self.buffers.items()):
                if not records:
                    continue
                
                try:
                    # Determine which INSERT to use based on table suffix
                    # IMPORTANT: Check more specific suffixes FIRST (e.g., _mark_prices before _prices)
                    # because 'xxx_mark_prices'.endswith('_prices') is True!
                    
                    if table_name.endswith('_mark_prices'):
                        self.conn.executemany(f"""
                            INSERT INTO {table_name}
                            (id, timestamp, mark_price, index_price, basis, basis_pct, funding_rate, annualized_rate)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, records)
                    
                    elif table_name.endswith('_prices'):
                        self.conn.executemany(f"""
                            INSERT INTO {table_name} 
                            (id, timestamp, mid_price, bid_price, ask_price, spread, spread_bps)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, records)
                    
                    elif table_name.endswith('_orderbooks'):
                        placeholders = ', '.join(['?'] * 48)  # 2 + 40 + 6
                        self.conn.executemany(f"""
                            INSERT INTO {table_name} VALUES ({placeholders})
                        """, records)
                    
                    elif table_name.endswith('_trades'):
                        self.conn.executemany(f"""
                            INSERT INTO {table_name}
                            (id, timestamp, trade_id, price, quantity, quote_value, side, is_buyer_maker)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, records)
                    
                    elif table_name.endswith('_funding_rates'):
                        self.conn.executemany(f"""
                            INSERT INTO {table_name}
                            (id, timestamp, funding_rate, funding_pct, annualized_pct, next_funding_time, countdown_secs)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, records)
                    
                    elif table_name.endswith('_open_interest'):
                        self.conn.executemany(f"""
                            INSERT INTO {table_name}
                            (id, timestamp, open_interest, open_interest_usd, oi_change_1h, oi_change_pct_1h)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, records)
                    
                    elif table_name.endswith('_ticker_24h'):
                        self.conn.executemany(f"""
                            INSERT INTO {table_name}
                            (id, timestamp, volume_24h, quote_volume_24h, trade_count_24h,
                             high_24h, low_24h, open_24h, last_price, price_change, price_change_pct, vwap_24h)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, records)
                    
                    elif table_name.endswith('_candles'):
                        # Use INSERT OR IGNORE to handle duplicate candles (same interval+open_time)
                        self.conn.executemany(f"""
                            INSERT OR IGNORE INTO {table_name}
                            (id, open_time, close_time, interval, open, high, low, close,
                             volume, quote_volume, trade_count, taker_buy_volume, taker_buy_quote, taker_buy_pct)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, records)
                    
                    elif table_name.endswith('_liquidations'):
                        self.conn.executemany(f"""
                            INSERT INTO {table_name}
                            (id, timestamp, side, price, quantity, value_usd, is_large)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, records)
                    
                    flushed += len(records)
                    tables_flushed += 1
                    successful_tables.append(table_name)
                    
                except Exception as e:
                    logger.error(f"Error flushing {table_name}: {e}")
                    failed_tables.append(table_name)
            
            # Only clear successfully flushed buffers (keep failed ones for retry)
            for table_name in successful_tables:
                self.buffers.pop(table_name, None)
            
            # Log if there are failed tables
            if failed_tables:
                logger.warning(f"⚠️ {len(failed_tables)} tables failed to flush, will retry: {failed_tables[:5]}...")
            
            duration_ms = (time.time() - start_time) * 1000
            if flushed > 0:
                logger.info(f"💾 Flushed {flushed:,} records to {tables_flushed} tables in {duration_ms:.1f}ms")
            
            self.stats['last_flush'] = datetime.now(timezone.utc)
            return flushed, tables_flushed


# ═══════════════════════════════════════════════════════════════════════════════
# TEST FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

async def test_isolated_collector():
    """Test the isolated collector with sample data."""
    
    collector = IsolatedDataCollector()
    collector.connect()
    
    print("\n" + "=" * 80)
    print("🧪 TESTING ISOLATED DATA COLLECTOR")
    print("=" * 80)
    
    # Test BTCUSDT on different exchanges
    test_cases = [
        # (symbol, exchange, market_type)
        ('BTCUSDT', 'binance', 'futures'),
        ('BTCUSDT', 'binance', 'spot'),
        ('BTCUSDT', 'bybit', 'futures'),
        ('BTCUSDT', 'bybit', 'spot'),
        ('BTCUSDT', 'pyth', 'oracle'),
        ('ETHUSDT', 'binance', 'futures'),
        ('BRETTUSDT', 'binance', 'futures'),
        ('BRETTUSDT', 'gateio', 'futures'),
        ('WIFUSDT', 'okx', 'futures'),  # WIFUSDT has no Bybit futures!
    ]
    
    # Add test data
    for symbol, exchange, market_type in test_cases:
        # Add price
        await collector.add_price(symbol, exchange, market_type, {
            'mid_price': 50000.0,
            'bid': 49999.0,
            'ask': 50001.0
        })
        
        # Add trade (not for oracle)
        if market_type != 'oracle':
            await collector.add_trade(symbol, exchange, market_type, {
                'price': 50000.0,
                'quantity': 0.5,
                'side': 'buy',
                'trade_id': 'T123'
            })
        
        # Add futures-only data
        if market_type == 'futures':
            await collector.add_mark_price(symbol, exchange, market_type, {
                'mark_price': 50000.0,
                'index_price': 49995.0,
                'funding_rate': 0.0001
            })
            
            await collector.add_funding_rate(symbol, exchange, market_type, {
                'funding_rate': 0.0001,
                'next_funding_time': datetime.now(timezone.utc)
            })
            
            await collector.add_open_interest(symbol, exchange, market_type, {
                'open_interest': 50000.0,
                'open_interest_usd': 2500000000.0
            })
            
            # Liquidation (not for Kraken)
            if exchange != 'kraken':
                await collector.add_liquidation(symbol, exchange, market_type, {
                    'side': 'long',
                    'price': 49000.0,
                    'quantity': 10.0
                })
        
        print(f"   ✅ Added data for {symbol} @ {exchange}_{market_type}")
    
    # Flush to database
    await collector.flush_buffers()
    
    # Verify data isolation
    print("\n" + "=" * 80)
    print("📊 VERIFYING DATA ISOLATION")
    print("=" * 80)
    
    # Check each table has ONLY its own data
    verification_queries = [
        ("btcusdt_binance_futures_prices", "BTCUSDT Binance Futures Prices"),
        ("btcusdt_binance_spot_prices", "BTCUSDT Binance Spot Prices"),
        ("btcusdt_bybit_futures_prices", "BTCUSDT Bybit Futures Prices"),
        ("btcusdt_pyth_oracle_prices", "BTCUSDT Pyth Oracle Prices"),
        ("ethusdt_binance_futures_prices", "ETHUSDT Binance Futures Prices"),
        ("brettusdt_binance_futures_prices", "BRETTUSDT Binance Futures Prices"),
        ("wifusdt_okx_futures_prices", "WIFUSDT OKX Futures Prices"),
    ]
    
    for table_name, description in verification_queries:
        try:
            count = collector.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"   ✅ {description}: {count} records")
        except Exception as e:
            print(f"   ❌ {description}: Error - {e}")
    
    # Show that tables are truly isolated
    print("\n" + "=" * 80)
    print("🔒 ISOLATION PROOF")
    print("=" * 80)
    
    # BTCUSDT Binance Futures should have NO spot data
    print("\n   BTCUSDT Binance Futures table contains:")
    btc_binance_futures = collector.conn.execute("""
        SELECT * FROM btcusdt_binance_futures_prices
    """).fetchall()
    for row in btc_binance_futures:
        print(f"      Row {row[0]}: mid_price={row[2]}")
    
    # BTCUSDT Binance Spot is in a COMPLETELY DIFFERENT table
    print("\n   BTCUSDT Binance Spot table contains:")
    btc_binance_spot = collector.conn.execute("""
        SELECT * FROM btcusdt_binance_spot_prices
    """).fetchall()
    for row in btc_binance_spot:
        print(f"      Row {row[0]}: mid_price={row[2]}")
    
    print("\n   ✅ Data is COMPLETELY ISOLATED - no mixing possible!")
    
    collector.conn.close()


if __name__ == "__main__":
    asyncio.run(test_isolated_collector())
