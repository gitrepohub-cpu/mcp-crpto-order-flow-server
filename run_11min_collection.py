"""
Standalone 11-minute data collection script.
Run with: python run_11min_collection.py
Or with: pythonw run_11min_collection.py (background)
"""

import asyncio
import sys
import logging
import signal
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from src.distributed_streaming.config import (
    ExchangeType,
    DataType,
    DistributedConfig,
)
from src.distributed_streaming.coordinator import StreamCoordinator


def main():
    # Ignore interrupt signals on Windows when running in background
    if sys.platform == 'win32':
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    
    # Configure logging - write to file instead of stdout for background operation
    log_file = Path(__file__).parent / "data" / "streaming.log"
    log_file.parent.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout) if sys.stdout else logging.NullHandler()
        ]
    )
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    
    # All 9 symbols
    symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ARUSDT",
        "BRETTUSDT", "POPCATUSDT", "WIFUSDT", "PNUTUSDT"
    ]
    
    # 5 exchanges that support all symbols
    exchanges = [
        ExchangeType.BINANCE_FUTURES,
        ExchangeType.BYBIT,
        ExchangeType.OKX,
        ExchangeType.HYPERLIQUID,
        ExchangeType.GATE,
    ]
    
    # Data types
    data_types = [
        DataType.TICKER,
        DataType.FUNDING_RATE,
        DataType.OPEN_INTEREST,
    ]
    
    config = DistributedConfig(
        symbols=symbols,
        exchanges=exchanges,
        data_types=data_types,
        db_path="data/distributed_streaming.duckdb",
        health_check_interval=30.0,
        stats_report_interval=60.0,
    )
    
    logger.info("=" * 60)
    logger.info("DISTRIBUTED STREAMING - 11 MINUTE COLLECTION")
    logger.info("=" * 60)
    logger.info(f"Symbols: {len(symbols)} - {', '.join(symbols)}")
    logger.info(f"Exchanges: {len(exchanges)} - {', '.join(e.value for e in exchanges)}")
    logger.info(f"Data types: {len(data_types)} - {', '.join(d.value for d in data_types)}")
    logger.info(f"Total workers: {config.total_workers}")
    logger.info("=" * 60)
    
    async def run():
        coordinator = StreamCoordinator(config)
        try:
            await coordinator.start(duration=11.0)  # 11 minutes
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            await coordinator.stop()
    
    asyncio.run(run())
    logger.info("Collection complete!")


if __name__ == "__main__":
    main()
