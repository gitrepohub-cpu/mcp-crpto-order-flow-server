"""
Test script to verify data collection from all exchanges.
"""

import asyncio
import sys

async def test_exchanges():
    from src.storage.direct_exchange_client import DirectExchangeClient
    
    client = DirectExchangeClient()
    print('='*70)
    print('STARTING DIRECT EXCHANGE CLIENT - DATA COLLECTION TEST')
    print('='*70)
    
    # Start connections
    print('\nConnecting to exchanges...')
    success = await client.start()
    
    if not success:
        print('Failed to connect to any exchange!')
        return
    
    # Wait for data to accumulate
    print('Waiting 15 seconds for data to accumulate...\n')
    await asyncio.sleep(15)
    
    # Report on collected data
    print('='*70)
    print('DATA COLLECTION REPORT')
    print('='*70)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
    
    for symbol in symbols:
        print(f'\n### {symbol} ###')
        
        # Prices
        prices = client.prices.get(symbol, {})
        if prices:
            print(f'  PRICES ({len(prices)} exchanges):')
            for ex, data in prices.items():
                price = data.get("price", 0)
                bid = data.get("bid", 0)
                ask = data.get("ask", 0)
                print(f'    {ex}: ${price:,.2f} (bid: {bid:,.2f}, ask: {ask:,.2f})')
        
        # Mark Prices
        marks = client.mark_prices.get(symbol, {})
        if marks:
            print(f'  MARK PRICES ({len(marks)} exchanges):')
            for ex, data in marks.items():
                mark = data.get("mark_price", 0)
                funding = data.get("funding_rate", 0) * 100
                print(f'    {ex}: mark=${mark:,.2f}, funding={funding:.4f}%')
        
        # Index Prices
        indices = client.index_prices.get(symbol, {})
        if indices:
            print(f'  INDEX PRICES ({len(indices)} exchanges):')
            for ex, data in indices.items():
                idx = data.get("index_price", 0)
                print(f'    {ex}: ${idx:,.2f}')
        
        # Open Interest
        oi = client.open_interest.get(symbol, {})
        if oi:
            print(f'  OPEN INTEREST ({len(oi)} exchanges):')
            for ex, data in oi.items():
                oi_qty = data.get("open_interest", 0)
                oi_val = data.get("open_interest_value", 0)
                print(f'    {ex}: {oi_qty:,.2f} (${oi_val:,.0f})')
        
        # Orderbooks
        books = client.orderbooks.get(symbol, {})
        if books:
            print(f'  ORDERBOOKS ({len(books)} exchanges):')
            for ex, data in books.items():
                bids = data.get('bids', [])
                asks = data.get('asks', [])
                print(f'    {ex}: {len(bids)} bids, {len(asks)} asks')
        
        # Trades
        trades = client.trades.get(symbol, {})
        if trades:
            print(f'  TRADES ({len(trades)} exchanges):')
            for ex, trade_list in trades.items():
                if isinstance(trade_list, list):
                    print(f'    {ex}: {len(trade_list)} recent trades')
        
        # 24h Ticker
        tickers = client.ticker_24h.get(symbol, {})
        if tickers:
            print(f'  24H TICKER ({len(tickers)} exchanges):')
            for ex, data in tickers.items():
                vol = data.get('volume', 0)
                high = data.get('high_24h', 0)
                low = data.get('low_24h', 0)
                change = data.get('price_change_pct', 0)
                print(f'    {ex}: vol={vol:,.2f}, high=${high:,.2f}, low=${low:,.2f}, chg={change:.2f}%')
        
        # Candles
        candles = client.candles.get(symbol, {})
        if candles:
            print(f'  CANDLES ({len(candles)} exchanges):')
            for ex, candle_list in candles.items():
                if isinstance(candle_list, list):
                    print(f'    {ex}: {len(candle_list)} 1-min candles')
    
    # Liquidations
    print(f'\n### LIQUIDATIONS ###')
    total_liqs = 0
    for symbol in symbols:
        liqs = client.liquidations.get(symbol, [])
        if liqs:
            print(f'  {symbol}: {len(liqs)} liquidation events')
            total_liqs += len(liqs)
    if total_liqs == 0:
        print('  No liquidations captured yet (normal if market is calm)')
    
    # Connected exchanges summary
    print(f'\n### CONNECTED EXCHANGES ###')
    connected_count = 0
    for ex, connected in sorted(client.connected_exchanges.items()):
        status = '✓ CONNECTED' if connected else '✗ DISCONNECTED'
        print(f'  {ex}: {status}')
        if connected:
            connected_count += 1
    
    print(f'\nTotal: {connected_count}/{len(client.connected_exchanges)} exchanges connected')
    
    # Data coverage summary
    print(f'\n### DATA COVERAGE SUMMARY ###')
    data_types = ['prices', 'mark_prices', 'index_prices', 'open_interest', 
                  'orderbooks', 'trades', 'ticker_24h', 'candles']
    
    for dtype in data_types:
        store = getattr(client, dtype, {})
        exchanges_with_data = set()
        for symbol in symbols:
            if symbol in store:
                for ex in store[symbol].keys():
                    exchanges_with_data.add(ex)
        print(f'  {dtype}: {len(exchanges_with_data)} exchanges')
    
    # Stop client
    await client.stop()
    print('\n' + '='*70)
    print('TEST COMPLETE')
    print('='*70)

if __name__ == "__main__":
    asyncio.run(test_exchanges())
