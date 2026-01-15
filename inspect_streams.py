"""
Stream Inspector - Shows all available data streams and their fields from each exchange.
Run this to see what data you can access and what columns are available.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment to use direct exchanges
os.environ["USE_DIRECT_EXCHANGES"] = "true"

from src.storage.direct_exchange_client import DirectExchangeClient


def print_separator(char="=", length=80):
    print(char * length)


def print_header(text):
    print_separator("=")
    print(f"  {text}")
    print_separator("=")


def print_subheader(text):
    print_separator("-")
    print(f"  {text}")
    print_separator("-")


def format_value(value):
    """Format value for display."""
    if isinstance(value, float):
        return f"{value:.8f}".rstrip('0').rstrip('.')
    elif isinstance(value, dict):
        return json.dumps(value, indent=2)
    elif isinstance(value, list):
        return f"[{len(value)} items]"
    return str(value)


async def inspect_streams():
    """Inspect all data streams from all exchanges."""
    print_header("CRYPTO EXCHANGE STREAM INSPECTOR")
    print(f"Started at: {datetime.utcnow().isoformat()}")
    print()
    
    # Create client
    client = DirectExchangeClient()
    
    print("Connecting to exchanges...")
    await client.connect()
    
    print("Waiting 10 seconds for data to stream in...")
    await asyncio.sleep(10)
    
    print("\n")
    
    # Inspect each data store
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    
    # ========================================================================
    # 1. PRICES
    # ========================================================================
    print_header("1. PRICES STREAM")
    print("Fields: price, timestamp")
    print()
    
    for symbol in symbols:
        if symbol in client.prices:
            print(f"📊 {symbol}:")
            for exchange, data in client.prices[symbol].items():
                if isinstance(data, dict):
                    price = data.get('price', 0)
                    ts = data.get('timestamp', 0)
                    print(f"   {exchange:25} → ${price:>12.2f}  (ts: {ts})")
                else:
                    print(f"   {exchange:25} → ${data:>12.2f}")
            print()
    
    # ========================================================================
    # 2. ORDERBOOKS
    # ========================================================================
    print_header("2. ORDERBOOK STREAM")
    print("Fields: bids[], asks[], spread, spread_pct, bid_depth, ask_depth, timestamp")
    print()
    
    sample_shown = False
    for symbol in symbols:
        if symbol in client.orderbooks and not sample_shown:
            print(f"📖 {symbol} - SAMPLE ORDERBOOK DATA:")
            for exchange, orderbook in list(client.orderbooks[symbol].items())[:2]:
                print(f"\n   Exchange: {exchange}")
                print(f"   Bids (top 3):")
                for i, bid in enumerate(orderbook.get('bids', [])[:3], 1):
                    print(f"      {i}. Price: ${bid['price']:,.2f}  Qty: {bid['quantity']:.4f}")
                print(f"   Asks (top 3):")
                for i, ask in enumerate(orderbook.get('asks', [])[:3], 1):
                    print(f"      {i}. Price: ${ask['price']:,.2f}  Qty: {ask['quantity']:.4f}")
                print(f"   Spread: ${orderbook.get('spread', 0):.2f} ({orderbook.get('spread_pct', 0):.4f}%)")
                print(f"   Bid Depth: {orderbook.get('bid_depth', 0):.4f}")
                print(f"   Ask Depth: {orderbook.get('ask_depth', 0):.4f}")
                sample_shown = True
                break
            break
    
    print("\n   Exchanges with orderbook data:")
    for symbol in symbols:
        if symbol in client.orderbooks:
            exchanges = list(client.orderbooks[symbol].keys())
            print(f"   {symbol}: {', '.join(exchanges)}")
    print()
    
    # ========================================================================
    # 3. TRADES
    # ========================================================================
    print_header("3. TRADES STREAM")
    print("Fields: price, quantity, side, timestamp, trade_id, value")
    print()
    
    total_trades = 0
    for symbol in symbols:
        if symbol in client.trades:
            print(f"💱 {symbol}:")
            for exchange, trade_list in client.trades[symbol].items():
                count = len(trade_list)
                total_trades += count
                if count > 0:
                    latest = trade_list[0]
                    print(f"   {exchange:25} → {count:>5} trades")
                    print(f"      Latest: ${latest.get('price', 0):,.2f}  "
                          f"Qty: {latest.get('quantity', 0):.4f}  "
                          f"Side: {latest.get('side', 'N/A'):>4}  "
                          f"Value: ${latest.get('value', 0):,.2f}")
            print()
    
    print(f"   Total trades captured: {total_trades}")
    
    if total_trades > 0:
        print("\n   SAMPLE TRADE FIELDS (first trade):")
        for symbol in symbols:
            if symbol in client.trades:
                for exchange, trade_list in client.trades[symbol].items():
                    if trade_list:
                        sample_trade = trade_list[0]
                        print(f"\n   {exchange} - {symbol}:")
                        for key, value in sample_trade.items():
                            print(f"      {key}: {format_value(value)}")
                        break
                break
    print()
    
    # ========================================================================
    # 4. FUNDING RATES
    # ========================================================================
    print_header("4. FUNDING RATES STREAM (Futures Only)")
    print("Fields: rate, rate_pct, annualized_rate, next_funding_time, timestamp")
    print()
    
    for symbol in symbols:
        if symbol in client.funding_rates:
            print(f"📈 {symbol}:")
            for exchange, data in client.funding_rates[symbol].items():
                rate_pct = data.get('rate_pct', 0)
                annualized = data.get('annualized_rate', 0)
                next_funding = data.get('next_funding_time', 'N/A')
                print(f"   {exchange:25} → {rate_pct:>8.4f}%  "
                      f"(Annualized: {annualized:>6.2f}%)  "
                      f"Next: {next_funding}")
            print()
    
    if not any(symbol in client.funding_rates for symbol in symbols):
        print("   No funding rate data available yet")
    print()
    
    # ========================================================================
    # 5. MARK PRICES
    # ========================================================================
    print_header("5. MARK PRICES STREAM (Futures Only)")
    print("Fields: mark_price, index_price, basis, basis_pct, timestamp")
    print()
    
    for symbol in symbols:
        if symbol in client.mark_prices:
            print(f"📍 {symbol}:")
            for exchange, data in client.mark_prices[symbol].items():
                mark = data.get('mark_price', 0)
                index = data.get('index_price', 0)
                basis = data.get('basis', 0)
                basis_pct = data.get('basis_pct', 0)
                print(f"   {exchange:25} → Mark: ${mark:,.2f}  "
                      f"Index: ${index:,.2f}  "
                      f"Basis: {basis_pct:.4f}%")
            print()
    
    if not any(symbol in client.mark_prices for symbol in symbols):
        print("   No mark price data available yet")
    print()
    
    # ========================================================================
    # 6. LIQUIDATIONS
    # ========================================================================
    print_header("6. LIQUIDATIONS STREAM")
    print("Fields: price, quantity, side, timestamp, value, exchange")
    print()
    
    total_liquidations = 0
    for symbol in symbols:
        if symbol in client.liquidations and client.liquidations[symbol]:
            print(f"⚠️  {symbol}:")
            for liq in client.liquidations[symbol][:5]:
                total_liquidations += 1
                print(f"   {liq.get('exchange', 'N/A'):25} → "
                      f"Side: {liq.get('side', 'N/A'):>5}  "
                      f"Price: ${liq.get('price', 0):,.2f}  "
                      f"Qty: {liq.get('quantity', 0):.4f}  "
                      f"Value: ${liq.get('value', 0):,.2f}")
            print()
    
    if total_liquidations == 0:
        print("   No recent liquidations detected (this is normal during calm markets)")
    else:
        print(f"   Total liquidations: {total_liquidations}")
    print()
    
    # ========================================================================
    # 7. OPEN INTEREST
    # ========================================================================
    print_header("7. OPEN INTEREST STREAM (Futures Only)")
    print("Fields: open_interest, open_interest_value, timestamp")
    print()
    
    for symbol in symbols:
        if symbol in client.open_interest:
            print(f"📊 {symbol}:")
            for exchange, data in client.open_interest[symbol].items():
                oi = data.get('open_interest', 0)
                oi_value = data.get('open_interest_value', 0)
                print(f"   {exchange:25} → OI: {oi:>15,.0f}  "
                      f"Value: ${oi_value:>15,.2f}")
            print()
    
    if not any(symbol in client.open_interest for symbol in symbols):
        print("   No open interest data available yet")
    print()
    
    # ========================================================================
    # 8. 24H TICKER
    # ========================================================================
    print_header("8. 24H TICKER STREAM")
    print("Fields: volume, quote_volume, high_24h, low_24h, price_change_pct, trades_count")
    print()
    
    for symbol in symbols:
        if symbol in client.ticker_24h:
            print(f"📈 {symbol}:")
            for exchange, data in client.ticker_24h[symbol].items():
                vol = data.get('volume', 0)
                quote_vol = data.get('quote_volume', 0)
                high = data.get('high_24h', 0)
                low = data.get('low_24h', 0)
                change = data.get('price_change_pct', 0)
                trades = data.get('trades_count', 0)
                
                print(f"   {exchange:25} →")
                print(f"      Volume: {vol:>15,.2f}  Quote: ${quote_vol:>15,.2f}")
                print(f"      High: ${high:>12,.2f}  Low: ${low:>12,.2f}  Change: {change:>6.2f}%")
                if trades > 0:
                    print(f"      Trades: {trades:,}")
            print()
    
    if not any(symbol in client.ticker_24h for symbol in symbols):
        print("   No 24h ticker data available yet")
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_header("STREAM SUMMARY")
    
    exchanges_with_data = set()
    for data_store in [client.prices, client.orderbooks, client.trades, 
                       client.funding_rates, client.mark_prices, 
                       client.open_interest, client.ticker_24h]:
        for symbol_data in data_store.values():
            if isinstance(symbol_data, dict):
                exchanges_with_data.update(symbol_data.keys())
    
    print(f"\n✅ Exchanges streaming data: {len(exchanges_with_data)}")
    for exchange in sorted(exchanges_with_data):
        print(f"   • {exchange}")
    
    print(f"\n📊 Data stores populated:")
    stores = {
        "Prices": len(client.prices),
        "Orderbooks": sum(len(v) for v in client.orderbooks.values()),
        "Trades": sum(len(t) for s in client.trades.values() for t in s.values()),
        "Funding Rates": len(client.funding_rates),
        "Mark Prices": len(client.mark_prices),
        "Liquidations": sum(len(v) for v in client.liquidations.values()),
        "Open Interest": len(client.open_interest),
        "24h Tickers": len(client.ticker_24h)
    }
    
    for store, count in stores.items():
        status = "✅" if count > 0 else "⏳"
        print(f"   {status} {store}: {count}")
    
    print()
    print_separator("=")
    print(f"Inspection completed at: {datetime.utcnow().isoformat()}")
    print_separator("=")
    
    # Cleanup
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(inspect_streams())
