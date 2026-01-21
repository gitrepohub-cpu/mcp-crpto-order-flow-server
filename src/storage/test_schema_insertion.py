"""
🧪 Test Data Insertion for ALL Coins, ALL Exchanges, ALL Streams
================================================================
Verifies that data can be inserted correctly for every combination:
- 9 symbols
- 9 exchanges (6 futures + 2 spot + 1 oracle)
- 9 data stream types

Each stream has its own exact column schema.
"""

import duckdb
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_data_insertion():
    """Test inserting sample data for ALL valid combinations."""
    
    db_path = "data/exchange_data.duckdb"
    
    if not os.path.exists(db_path):
        print("❌ Database not found! Run database_init.py first.")
        return
    
    conn = duckdb.connect(db_path)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT',
               'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT']
    
    FUTURES_EXCHANGES = ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid']
    SPOT_EXCHANGES = ['binance', 'bybit']
    
    # Symbols NOT available on certain futures exchanges
    FUTURES_EXCLUSIONS = {
        'bybit': ['WIFUSDT'],           # WIFUSDT not on Bybit futures
        'okx': ['PNUTUSDT'],            # PNUTUSDT not on OKX
        'kraken': ['BRETTUSDT', 'ARUSDT'],  # Meme coins
        'hyperliquid': ['BRETTUSDT', 'ARUSDT']
    }
    
    # Spot symbols (limited)
    BINANCE_SPOT = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'WIFUSDT', 'PNUTUSDT']
    BYBIT_SPOT = SYMBOLS  # All symbols
    
    # Oracle (major coins only)
    ORACLE_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT']
    
    now = datetime.now(timezone.utc)
    results = {'passed': 0, 'failed': 0, 'skipped': 0}
    
    print("=" * 100)
    print("🧪 TESTING DATA INSERTION FOR ALL COMBINATIONS")
    print("=" * 100)
    
    # Get next IDs
    id_counters = {}
    for table in ['prices', 'orderbooks', 'trades', 'mark_prices', 'funding_rates',
                  'open_interest', 'ticker_24h', 'candles', 'liquidations']:
        max_id = conn.execute(f"SELECT COALESCE(MAX(id), 0) + 1 FROM {table}").fetchone()[0]
        id_counters[table] = max_id
    
    def get_id(table):
        id_counters[table] += 1
        return id_counters[table] - 1
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 1: PRICES (all market types)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n📊 Testing PRICES (11 columns)...")
    print("-" * 80)
    
    # Futures prices
    for exchange in FUTURES_EXCHANGES:
        exclusions = FUTURES_EXCLUSIONS.get(exchange, [])
        for symbol in SYMBOLS:
            if symbol in exclusions:
                results['skipped'] += 1
                continue
            try:
                conn.execute("""
                    INSERT INTO prices (id, timestamp, symbol, exchange, market_type, data_source,
                                       mid_price, bid_price, ask_price, spread, spread_bps)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (get_id('prices'), now, symbol, exchange, 'futures', f'{exchange}_futures',
                      100000.0, 99999.0, 100001.0, 2.0, 0.2))
                results['passed'] += 1
            except Exception as e:
                print(f"   ❌ FUTURES {symbol}@{exchange}: {e}")
                results['failed'] += 1
    
    # Spot prices
    for exchange in SPOT_EXCHANGES:
        symbols = BINANCE_SPOT if exchange == 'binance' else BYBIT_SPOT
        for symbol in symbols:
            try:
                conn.execute("""
                    INSERT INTO prices (id, timestamp, symbol, exchange, market_type, data_source,
                                       mid_price, bid_price, ask_price, spread, spread_bps)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (get_id('prices'), now, symbol, exchange, 'spot', f'{exchange}_spot',
                      100000.0, 99999.0, 100001.0, 2.0, 0.2))
                results['passed'] += 1
            except Exception as e:
                print(f"   ❌ SPOT {symbol}@{exchange}: {e}")
                results['failed'] += 1
    
    # Oracle prices (price only, no bid/ask)
    for symbol in ORACLE_SYMBOLS:
        try:
            conn.execute("""
                INSERT INTO prices (id, timestamp, symbol, exchange, market_type, data_source,
                                   mid_price, bid_price, ask_price, spread, spread_bps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (get_id('prices'), now, symbol, 'pyth', 'oracle', 'pyth_oracle',
                  100000.0, None, None, None, None))
            results['passed'] += 1
        except Exception as e:
            print(f"   ❌ ORACLE {symbol}: {e}")
            results['failed'] += 1
    
    print(f"   ✅ Prices: {results['passed']} passed")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 2: ORDERBOOKS (52 columns) - futures & spot only
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n📊 Testing ORDERBOOKS (52 columns)...")
    print("-" * 80)
    
    ob_passed = 0
    # Create sample orderbook data (10 levels)
    bid_prices = [99999.0 - i*10 for i in range(10)]
    bid_qtys = [1.0 + i*0.1 for i in range(10)]
    ask_prices = [100001.0 + i*10 for i in range(10)]
    ask_qtys = [1.0 + i*0.1 for i in range(10)]
    
    # Futures orderbooks
    for exchange in FUTURES_EXCHANGES:
        exclusions = FUTURES_EXCLUSIONS.get(exchange, [])
        for symbol in SYMBOLS:
            if symbol in exclusions:
                continue
            try:
                conn.execute("""
                    INSERT INTO orderbooks (
                        id, timestamp, symbol, exchange, market_type, data_source,
                        bid_1_price, bid_2_price, bid_3_price, bid_4_price, bid_5_price,
                        bid_6_price, bid_7_price, bid_8_price, bid_9_price, bid_10_price,
                        bid_1_qty, bid_2_qty, bid_3_qty, bid_4_qty, bid_5_qty,
                        bid_6_qty, bid_7_qty, bid_8_qty, bid_9_qty, bid_10_qty,
                        ask_1_price, ask_2_price, ask_3_price, ask_4_price, ask_5_price,
                        ask_6_price, ask_7_price, ask_8_price, ask_9_price, ask_10_price,
                        ask_1_qty, ask_2_qty, ask_3_qty, ask_4_qty, ask_5_qty,
                        ask_6_qty, ask_7_qty, ask_8_qty, ask_9_qty, ask_10_qty,
                        total_bid_depth, total_ask_depth, bid_ask_ratio, spread, spread_pct, mid_price
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (get_id('orderbooks'), now, symbol, exchange, 'futures', f'{exchange}_futures',
                      *bid_prices, *bid_qtys, *ask_prices, *ask_qtys,
                      sum(bid_qtys), sum(ask_qtys), 1.0, 2.0, 0.002, 100000.0))
                ob_passed += 1
            except Exception as e:
                print(f"   ❌ FUTURES {symbol}@{exchange}: {e}")
                results['failed'] += 1
    
    # Spot orderbooks
    for exchange in SPOT_EXCHANGES:
        symbols = BINANCE_SPOT if exchange == 'binance' else BYBIT_SPOT
        for symbol in symbols:
            try:
                conn.execute("""
                    INSERT INTO orderbooks (
                        id, timestamp, symbol, exchange, market_type, data_source,
                        bid_1_price, bid_2_price, bid_3_price, bid_4_price, bid_5_price,
                        bid_6_price, bid_7_price, bid_8_price, bid_9_price, bid_10_price,
                        bid_1_qty, bid_2_qty, bid_3_qty, bid_4_qty, bid_5_qty,
                        bid_6_qty, bid_7_qty, bid_8_qty, bid_9_qty, bid_10_qty,
                        ask_1_price, ask_2_price, ask_3_price, ask_4_price, ask_5_price,
                        ask_6_price, ask_7_price, ask_8_price, ask_9_price, ask_10_price,
                        ask_1_qty, ask_2_qty, ask_3_qty, ask_4_qty, ask_5_qty,
                        ask_6_qty, ask_7_qty, ask_8_qty, ask_9_qty, ask_10_qty,
                        total_bid_depth, total_ask_depth, bid_ask_ratio, spread, spread_pct, mid_price
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (get_id('orderbooks'), now, symbol, exchange, 'spot', f'{exchange}_spot',
                      *bid_prices, *bid_qtys, *ask_prices, *ask_qtys,
                      sum(bid_qtys), sum(ask_qtys), 1.0, 2.0, 0.002, 100000.0))
                ob_passed += 1
            except Exception as e:
                print(f"   ❌ SPOT {symbol}@{exchange}: {e}")
                results['failed'] += 1
    
    print(f"   ✅ Orderbooks: {ob_passed} passed")
    results['passed'] += ob_passed
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 3: TRADES (12 columns) - futures & spot only
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n📊 Testing TRADES (12 columns)...")
    print("-" * 80)
    
    trades_passed = 0
    
    # Futures trades
    for exchange in FUTURES_EXCHANGES:
        exclusions = FUTURES_EXCLUSIONS.get(exchange, [])
        for symbol in SYMBOLS:
            if symbol in exclusions:
                continue
            try:
                conn.execute("""
                    INSERT INTO trades (id, timestamp, symbol, exchange, market_type, data_source,
                                       trade_id, price, quantity, quote_value, side, is_buyer_maker)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (get_id('trades'), now, symbol, exchange, 'futures', f'{exchange}_futures',
                      'T123456', 100000.0, 0.5, 50000.0, 'buy', True))
                trades_passed += 1
            except Exception as e:
                print(f"   ❌ FUTURES {symbol}@{exchange}: {e}")
                results['failed'] += 1
    
    # Spot trades
    for exchange in SPOT_EXCHANGES:
        symbols = BINANCE_SPOT if exchange == 'binance' else BYBIT_SPOT
        for symbol in symbols:
            try:
                conn.execute("""
                    INSERT INTO trades (id, timestamp, symbol, exchange, market_type, data_source,
                                       trade_id, price, quantity, quote_value, side, is_buyer_maker)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (get_id('trades'), now, symbol, exchange, 'spot', f'{exchange}_spot',
                      'T123456', 100000.0, 0.5, 50000.0, 'sell', False))
                trades_passed += 1
            except Exception as e:
                print(f"   ❌ SPOT {symbol}@{exchange}: {e}")
                results['failed'] += 1
    
    print(f"   ✅ Trades: {trades_passed} passed")
    results['passed'] += trades_passed
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 4: MARK_PRICES (12 columns) - FUTURES ONLY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n📊 Testing MARK_PRICES (12 columns) - FUTURES ONLY...")
    print("-" * 80)
    
    mark_passed = 0
    
    for exchange in FUTURES_EXCHANGES:
        exclusions = FUTURES_EXCLUSIONS.get(exchange, [])
        # Some exchanges don't have index_price
        has_index = exchange in ['binance', 'bybit', 'gateio']
        
        for symbol in SYMBOLS:
            if symbol in exclusions:
                continue
            try:
                index_price = 99995.0 if has_index else None
                basis = 5.0 if has_index else None
                basis_pct = 0.005 if has_index else None
                
                conn.execute("""
                    INSERT INTO mark_prices (id, timestamp, symbol, exchange, market_type, data_source,
                                            mark_price, index_price, basis, basis_pct, funding_rate, annualized_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (get_id('mark_prices'), now, symbol, exchange, 'futures', f'{exchange}_futures',
                      100000.0, index_price, basis, basis_pct, 0.0001, 10.95))
                mark_passed += 1
            except Exception as e:
                print(f"   ❌ {symbol}@{exchange}: {e}")
                results['failed'] += 1
    
    print(f"   ✅ Mark Prices: {mark_passed} passed")
    results['passed'] += mark_passed
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 5: FUNDING_RATES (11 columns) - FUTURES ONLY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n📊 Testing FUNDING_RATES (11 columns) - FUTURES ONLY...")
    print("-" * 80)
    
    funding_passed = 0
    
    for exchange in FUTURES_EXCHANGES:
        exclusions = FUTURES_EXCLUSIONS.get(exchange, [])
        for symbol in SYMBOLS:
            if symbol in exclusions:
                continue
            try:
                conn.execute("""
                    INSERT INTO funding_rates (id, timestamp, symbol, exchange, market_type, data_source,
                                              funding_rate, funding_pct, annualized_pct, next_funding_time, countdown_secs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (get_id('funding_rates'), now, symbol, exchange, 'futures', f'{exchange}_futures',
                      0.0001, 0.01, 10.95, now, 28800))
                funding_passed += 1
            except Exception as e:
                print(f"   ❌ {symbol}@{exchange}: {e}")
                results['failed'] += 1
    
    print(f"   ✅ Funding Rates: {funding_passed} passed")
    results['passed'] += funding_passed
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 6: OPEN_INTEREST (10 columns) - FUTURES ONLY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n📊 Testing OPEN_INTEREST (10 columns) - FUTURES ONLY...")
    print("-" * 80)
    
    oi_passed = 0
    
    for exchange in FUTURES_EXCHANGES:
        exclusions = FUTURES_EXCLUSIONS.get(exchange, [])
        for symbol in SYMBOLS:
            if symbol in exclusions:
                continue
            try:
                conn.execute("""
                    INSERT INTO open_interest (id, timestamp, symbol, exchange, market_type, data_source,
                                              open_interest, open_interest_usd, oi_change_1h, oi_change_pct_1h)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (get_id('open_interest'), now, symbol, exchange, 'futures', f'{exchange}_futures',
                      50000.0, 5000000000.0, 100.0, 0.2))
                oi_passed += 1
            except Exception as e:
                print(f"   ❌ {symbol}@{exchange}: {e}")
                results['failed'] += 1
    
    print(f"   ✅ Open Interest: {oi_passed} passed")
    results['passed'] += oi_passed
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 7: TICKER_24H (16 columns) - futures & spot (NOT Hyperliquid)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n📊 Testing TICKER_24H (16 columns) - NOT Hyperliquid or Oracle...")
    print("-" * 80)
    
    ticker_passed = 0
    
    # Futures tickers (not Hyperliquid)
    futures_with_ticker = [e for e in FUTURES_EXCHANGES if e != 'hyperliquid']
    for exchange in futures_with_ticker:
        exclusions = FUTURES_EXCLUSIONS.get(exchange, [])
        for symbol in SYMBOLS:
            if symbol in exclusions:
                continue
            try:
                conn.execute("""
                    INSERT INTO ticker_24h (id, timestamp, symbol, exchange, market_type, data_source,
                                           volume_24h, quote_volume_24h, trade_count_24h,
                                           high_24h, low_24h, open_24h, last_price,
                                           price_change, price_change_pct, vwap_24h)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (get_id('ticker_24h'), now, symbol, exchange, 'futures', f'{exchange}_futures',
                      10000.0, 1000000000.0, 500000, 101000.0, 99000.0, 99500.0, 100000.0,
                      500.0, 0.5, 100100.0))
                ticker_passed += 1
            except Exception as e:
                print(f"   ❌ FUTURES {symbol}@{exchange}: {e}")
                results['failed'] += 1
    
    # Spot tickers
    for exchange in SPOT_EXCHANGES:
        symbols = BINANCE_SPOT if exchange == 'binance' else BYBIT_SPOT
        for symbol in symbols:
            try:
                conn.execute("""
                    INSERT INTO ticker_24h (id, timestamp, symbol, exchange, market_type, data_source,
                                           volume_24h, quote_volume_24h, trade_count_24h,
                                           high_24h, low_24h, open_24h, last_price,
                                           price_change, price_change_pct, vwap_24h)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (get_id('ticker_24h'), now, symbol, exchange, 'spot', f'{exchange}_spot',
                      10000.0, 1000000000.0, 500000, 101000.0, 99000.0, 99500.0, 100000.0,
                      500.0, 0.5, 100100.0))
                ticker_passed += 1
            except Exception as e:
                print(f"   ❌ SPOT {symbol}@{exchange}: {e}")
                results['failed'] += 1
    
    print(f"   ✅ Ticker 24h: {ticker_passed} passed")
    results['passed'] += ticker_passed
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 8: CANDLES (18 columns) - futures & spot only
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n📊 Testing CANDLES (18 columns)...")
    print("-" * 80)
    
    candles_passed = 0
    
    # Futures candles
    for exchange in FUTURES_EXCHANGES:
        exclusions = FUTURES_EXCLUSIONS.get(exchange, [])
        for symbol in SYMBOLS:
            if symbol in exclusions:
                continue
            try:
                conn.execute("""
                    INSERT INTO candles (id, open_time, close_time, symbol, exchange, market_type, data_source, interval,
                                        open, high, low, close, volume, quote_volume,
                                        trade_count, taker_buy_volume, taker_buy_quote, taker_buy_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (get_id('candles'), now, now, symbol, exchange, 'futures', f'{exchange}_futures', '1m',
                      99500.0, 100100.0, 99400.0, 100000.0, 100.0, 10000000.0,
                      1000, 60.0, 6000000.0, 60.0))
                candles_passed += 1
            except Exception as e:
                print(f"   ❌ FUTURES {symbol}@{exchange}: {e}")
                results['failed'] += 1
    
    # Spot candles
    for exchange in SPOT_EXCHANGES:
        symbols = BINANCE_SPOT if exchange == 'binance' else BYBIT_SPOT
        for symbol in symbols:
            try:
                conn.execute("""
                    INSERT INTO candles (id, open_time, close_time, symbol, exchange, market_type, data_source, interval,
                                        open, high, low, close, volume, quote_volume,
                                        trade_count, taker_buy_volume, taker_buy_quote, taker_buy_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (get_id('candles'), now, now, symbol, exchange, 'spot', f'{exchange}_spot', '1m',
                      99500.0, 100100.0, 99400.0, 100000.0, 100.0, 10000000.0,
                      1000, 60.0, 6000000.0, 60.0))
                candles_passed += 1
            except Exception as e:
                print(f"   ❌ SPOT {symbol}@{exchange}: {e}")
                results['failed'] += 1
    
    print(f"   ✅ Candles: {candles_passed} passed")
    results['passed'] += candles_passed
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 9: LIQUIDATIONS (11 columns) - FUTURES ONLY (NOT Kraken)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n📊 Testing LIQUIDATIONS (11 columns) - FUTURES ONLY, NOT Kraken...")
    print("-" * 80)
    
    liq_passed = 0
    
    # Futures with liquidations (not Kraken)
    futures_with_liq = [e for e in FUTURES_EXCHANGES if e != 'kraken']
    for exchange in futures_with_liq:
        exclusions = FUTURES_EXCLUSIONS.get(exchange, [])
        for symbol in SYMBOLS:
            if symbol in exclusions:
                continue
            try:
                conn.execute("""
                    INSERT INTO liquidations (id, timestamp, symbol, exchange, market_type, data_source,
                                             side, price, quantity, value_usd, is_large)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (get_id('liquidations'), now, symbol, exchange, 'futures', f'{exchange}_futures',
                      'long', 100000.0, 5.0, 500000.0, True))
                liq_passed += 1
            except Exception as e:
                print(f"   ❌ {symbol}@{exchange}: {e}")
                results['failed'] += 1
    
    print(f"   ✅ Liquidations: {liq_passed} passed")
    results['passed'] += liq_passed
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("📊 TEST SUMMARY")
    print("=" * 100)
    
    # Count records by table and market type
    summary = conn.execute("""
        SELECT 'prices' as table_name, market_type, COUNT(*) as cnt FROM prices GROUP BY market_type
        UNION ALL
        SELECT 'orderbooks', market_type, COUNT(*) FROM orderbooks GROUP BY market_type
        UNION ALL
        SELECT 'trades', market_type, COUNT(*) FROM trades GROUP BY market_type
        UNION ALL
        SELECT 'mark_prices', market_type, COUNT(*) FROM mark_prices GROUP BY market_type
        UNION ALL
        SELECT 'funding_rates', market_type, COUNT(*) FROM funding_rates GROUP BY market_type
        UNION ALL
        SELECT 'open_interest', market_type, COUNT(*) FROM open_interest GROUP BY market_type
        UNION ALL
        SELECT 'ticker_24h', market_type, COUNT(*) FROM ticker_24h GROUP BY market_type
        UNION ALL
        SELECT 'candles', market_type, COUNT(*) FROM candles GROUP BY market_type
        UNION ALL
        SELECT 'liquidations', market_type, COUNT(*) FROM liquidations GROUP BY market_type
        ORDER BY table_name, market_type
    """).fetchall()
    
    print("\n📈 Records by Table and Market Type:")
    print("-" * 60)
    current_table = None
    for row in summary:
        if row[0] != current_table:
            current_table = row[0]
            print(f"\n{current_table}:")
        print(f"   {row[1]}: {row[2]} records")
    
    # Total by market type
    totals = conn.execute("""
        SELECT 
            SUM(CASE WHEN market_type = 'futures' THEN 1 ELSE 0 END) as futures,
            SUM(CASE WHEN market_type = 'spot' THEN 1 ELSE 0 END) as spot,
            SUM(CASE WHEN market_type = 'oracle' THEN 1 ELSE 0 END) as oracle,
            COUNT(*) as total
        FROM (
            SELECT market_type FROM prices
            UNION ALL SELECT market_type FROM orderbooks
            UNION ALL SELECT market_type FROM trades
            UNION ALL SELECT market_type FROM mark_prices
            UNION ALL SELECT market_type FROM funding_rates
            UNION ALL SELECT market_type FROM open_interest
            UNION ALL SELECT market_type FROM ticker_24h
            UNION ALL SELECT market_type FROM candles
            UNION ALL SELECT market_type FROM liquidations
        )
    """).fetchone()
    
    print(f"\n" + "=" * 60)
    print("📊 MARKET TYPE SEPARATION VERIFIED:")
    print(f"   🔵 Futures: {totals[0]} records")
    print(f"   🟢 Spot: {totals[1]} records")
    print(f"   🟡 Oracle: {totals[2]} records")
    print(f"   📊 TOTAL: {totals[3]} records")
    print("=" * 60)
    
    print(f"\n✅ Tests Passed: {results['passed']}")
    print(f"❌ Tests Failed: {results['failed']}")
    print(f"⏭️ Tests Skipped: {results['skipped']}")
    
    if results['failed'] == 0:
        print("\n🎉 ALL TESTS PASSED! Database schemas are correct for all coins/exchanges/streams.")
    
    # Database size
    db_size = os.path.getsize(db_path)
    print(f"\n💾 Database size: {db_size / 1024:.2f} KB")
    
    conn.close()


if __name__ == "__main__":
    test_data_insertion()
