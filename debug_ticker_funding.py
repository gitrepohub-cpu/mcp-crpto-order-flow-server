"""Debug 24h ticker and funding rate data"""
import asyncio
import json
import websockets

async def debug_binance_ticker():
    """Check raw Binance 24h ticker data format"""
    url = "wss://fstream.binance.com/stream?streams=btcusdt@ticker"
    
    print("=== BINANCE FUTURES 24H TICKER RAW DATA ===")
    async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
        for i in range(3):
            message = await ws.recv()
            data = json.loads(message)
            if "data" in data:
                payload = data["data"]
                print(f"\nMessage {i+1}:")
                print(f"  Event type: {payload.get('e')}")
                print(f"  Symbol: {payload.get('s')}")
                print(f"  Base Volume (v): {payload.get('v')}")
                print(f"  Quote Volume (q): {payload.get('q')}")
                print(f"  High (h): {payload.get('h')}")
                print(f"  Low (l): {payload.get('l')}")
                print(f"  Price Change %: {payload.get('P')}")
                print(f"  All keys: {list(payload.keys())}")

async def debug_binance_mark():
    """Check raw Binance mark price + funding data format"""
    url = "wss://fstream.binance.com/stream?streams=btcusdt@markPrice@1s"
    
    print("\n=== BINANCE FUTURES MARK PRICE RAW DATA ===")
    async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
        for i in range(3):
            message = await ws.recv()
            data = json.loads(message)
            if "data" in data:
                payload = data["data"]
                print(f"\nMessage {i+1}:")
                print(f"  Event type: {payload.get('e')}")
                print(f"  Mark Price (p): {payload.get('p')}")
                print(f"  Index Price (i): {payload.get('i')}")
                print(f"  Funding Rate (r): {payload.get('r')}")
                print(f"  Next Funding Time (T): {payload.get('T')}")
                print(f"  All keys: {list(payload.keys())}")

async def debug_bybit_ticker():
    """Check raw Bybit ticker data"""
    url = "wss://stream.bybit.com/v5/public/linear"
    
    print("\n=== BYBIT FUTURES TICKER RAW DATA ===")
    async with websockets.connect(url, ping_interval=20, ping_timeout=60) as ws:
        sub = {
            "op": "subscribe",
            "args": ["tickers.BTCUSDT"]
        }
        await ws.send(json.dumps(sub))
        
        for i in range(5):
            message = await ws.recv()
            data = json.loads(message)
            if "data" in data:
                payload = data["data"]
                print(f"\nMessage {i+1}:")
                print(f"  Topic: {data.get('topic')}")
                print(f"  Volume 24h: {payload.get('volume24h')}")
                print(f"  Turnover 24h: {payload.get('turnover24h')}")
                print(f"  High 24h: {payload.get('highPrice24h')}")
                print(f"  Low 24h: {payload.get('lowPrice24h')}")
                print(f"  Funding Rate: {payload.get('fundingRate')}")
                print(f"  Mark Price: {payload.get('markPrice')}")
                print(f"  All keys: {list(payload.keys())}")

async def main():
    await debug_binance_ticker()
    await debug_binance_mark()
    await debug_bybit_ticker()

if __name__ == "__main__":
    asyncio.run(main())
