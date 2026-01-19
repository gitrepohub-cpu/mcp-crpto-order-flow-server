"""Debug Kraken and OKX data formats"""
import asyncio
import json
import websockets

async def debug_kraken():
    """Check raw Kraken ticker data"""
    url = "wss://futures.kraken.com/ws/v1"
    
    print("=== KRAKEN FUTURES TICKER RAW DATA ===")
    async with websockets.connect(url, ping_interval=20, ping_timeout=60) as ws:
        sub = {
            "event": "subscribe",
            "feed": "ticker",
            "product_ids": ["PF_ETHUSD"]
        }
        await ws.send(json.dumps(sub))
        
        for i in range(5):
            message = await ws.recv()
            data = json.loads(message)
            print(f"\nMessage {i+1}:")
            print(f"  Feed: {data.get('feed')}")
            print(f"  Volume: {data.get('volume')}")
            print(f"  High: {data.get('high')}")
            print(f"  Low: {data.get('low')}")
            print(f"  Open: {data.get('open')}")
            print(f"  Change: {data.get('change')}")
            print(f"  Funding Rate: {data.get('fundingRate')}")
            if 'change' in data:
                print(f"  --> Change value type: {type(data.get('change'))}")
            print(f"  Keys: {list(data.keys())[:15]}...")

async def debug_okx_funding():
    """Check OKX funding rate channel"""
    url = "wss://ws.okx.com:8443/ws/v5/public"
    
    print("\n=== OKX FUNDING RATE RAW DATA ===")
    async with websockets.connect(url, ping_interval=20, ping_timeout=60) as ws:
        sub = {
            "op": "subscribe",
            "args": [{"channel": "funding-rate", "instId": "BTC-USDT-SWAP"}]
        }
        await ws.send(json.dumps(sub))
        
        for i in range(5):
            message = await ws.recv()
            data = json.loads(message)
            print(f"\nMessage {i+1}:")
            print(json.dumps(data, indent=2)[:500])

async def debug_gate_volume():
    """Check Gate.io volume fields"""
    url = "wss://fx-ws.gateio.ws/v4/ws/usdt"
    
    print("\n=== GATE.IO TICKER RAW DATA ===")
    async with websockets.connect(url, ping_interval=20, ping_timeout=60) as ws:
        sub = {
            "time": int(asyncio.get_event_loop().time()),
            "channel": "futures.tickers",
            "event": "subscribe",
            "payload": ["BTC_USDT"]
        }
        await ws.send(json.dumps(sub))
        
        for i in range(5):
            message = await ws.recv()
            data = json.loads(message)
            if "result" in data and data["result"]:
                result = data["result"]
                if isinstance(result, list):
                    result = result[0]
                print(f"\nMessage {i+1}:")
                print(f"  Volume 24h: {result.get('volume_24h')}")
                print(f"  Volume 24h USD: {result.get('volume_24h_usd')}")
                print(f"  Volume 24h BTC: {result.get('volume_24h_btc')}")
                print(f"  Volume 24h Quote: {result.get('volume_24h_quote')}")
                print(f"  Last: {result.get('last')}")
                print(f"  Change %: {result.get('change_percentage')}")
                print(f"  Keys: {list(result.keys())}")

async def main():
    await debug_kraken()
    await debug_okx_funding()
    await debug_gate_volume()

if __name__ == "__main__":
    asyncio.run(main())
