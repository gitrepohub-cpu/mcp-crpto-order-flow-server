"""
Simple WebSocket collector test
"""

import asyncio
import json
import logging
import websockets

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

running = True

async def connect_binance():
    """Connect to Binance Futures."""
    global running
    url = "wss://fstream.binance.com/ws/btcusdt@ticker"
    
    logger.info("Connecting to Binance...")
    
    try:
        async with websockets.connect(url, ping_interval=30) as ws:
            logger.info("✅ Connected to Binance!")
            count = 0
            
            while running:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10)
                    data = json.loads(msg)
                    price = data.get('c', 0)
                    count += 1
                    if count % 10 == 0:
                        logger.info(f"📊 Received {count} messages, BTC price: {price}")
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for message")
                except Exception as e:
                    logger.error(f"Error receiving: {e}")
                    break
                    
    except Exception as e:
        logger.error(f"Connection error: {e}")
        import traceback
        traceback.print_exc()
        
async def heartbeat():
    """Keep the loop alive."""
    global running
    while running:
        await asyncio.sleep(5)
        logger.info("💓 Heartbeat")
        
async def main():
    global running
    logger.info("Starting test collector...")
    
    try:
        await asyncio.gather(
            connect_binance(),
            heartbeat()
        )
    except KeyboardInterrupt:
        running = False
        logger.info("Stopped by user")
    except Exception as e:
        logger.error(f"Main error: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped")
