"""
⏰ Feature Calculation Scheduler
================================

Background service that runs feature calculations at regular intervals.
Writes computed features to the feature database for Streamlit consumption.

Phases:
- Phase 1: Price Features (every 1 second)
- Phase 2: Trade Features (every 1 second) 
- Phase 3: Flow Features (every 5 seconds)
- Phase 4: Funding/OI Features (every 5-60 seconds)
- Phase 5: Volatility/Momentum (every 5 seconds)
- Phase 6: Composite Signals (every 5 seconds)

Usage:
    # Run as standalone service
    python -m src.features.storage.feature_scheduler
    
    # Or import and run
    from src.features.storage.feature_scheduler import FeatureScheduler
    scheduler = FeatureScheduler()
    await scheduler.start()
"""

import asyncio
import logging
import signal
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import threading

try:
    import duckdb
except ImportError:
    duckdb = None

from .price_feature_calculator import PriceFeatureCalculator
from .trade_feature_calculator import TradeFeatureCalculator
from .flow_feature_calculator import FlowFeatureCalculator
from .feature_database_init import (
    FEATURE_SYMBOLS,
    SYMBOL_EXCHANGE_MAP,
    FEATURE_CATEGORIES,
    FEATURE_DB_PATH,
)

logger = logging.getLogger(__name__)


@dataclass
class CalculationStats:
    """Statistics for feature calculations."""
    total_calculations: int = 0
    successful_calculations: int = 0
    failed_calculations: int = 0
    last_calculation_time: Optional[datetime] = None
    avg_calculation_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""
    # Update frequencies in seconds
    price_features_interval: float = 1.0
    trade_features_interval: float = 1.0
    flow_features_interval: float = 5.0
    funding_features_interval: float = 60.0
    oi_features_interval: float = 5.0
    volatility_features_interval: float = 5.0
    momentum_features_interval: float = 5.0
    signals_interval: float = 5.0
    cross_exchange_interval: float = 10.0
    
    # Control flags
    enabled_categories: List[str] = field(default_factory=lambda: ['price_features', 'trade_features', 'flow_features'])
    max_errors_before_pause: int = 10
    error_pause_seconds: int = 30
    
    # Database paths
    raw_db_path: Optional[str] = None
    feature_db_path: Optional[str] = None


class FeatureScheduler:
    """
    Schedules and runs feature calculations at regular intervals.
    
    This scheduler:
    1. Runs different feature calculators at their specified intervals
    2. Tracks calculation statistics and errors
    3. Handles graceful shutdown
    4. Supports enabling/disabling specific feature categories
    
    Architecture:
        - Each feature category has its own async task
        - Tasks run independently at their specified intervals
        - Errors are tracked and can trigger pauses
        - Status can be queried for monitoring
    """
    
    def __init__(self, config: SchedulerConfig = None):
        """
        Initialize the scheduler.
        
        Args:
            config: Scheduler configuration (defaults to standard config)
        """
        self.config = config or SchedulerConfig()
        self._running = False
        self._tasks: Dict[str, asyncio.Task] = {}
        self._stats: Dict[str, CalculationStats] = {}
        self._calculators: Dict[str, Any] = {}
        
        # Initialize statistics for each category
        for category in FEATURE_CATEGORIES.keys():
            self._stats[category] = CalculationStats()
        self._stats['cross_exchange'] = CalculationStats()
        
        # Initialize calculators
        self._init_calculators()
    
    def _init_calculators(self):
        """Initialize feature calculators."""
        self._calculators['price_features'] = PriceFeatureCalculator(
            raw_db_path=self.config.raw_db_path,
            feature_db_path=self.config.feature_db_path
        )
        # Phase 2: Trade and Flow calculators
        self._calculators['trade_features'] = TradeFeatureCalculator(
            raw_db_path=self.config.raw_db_path,
            feature_db_path=self.config.feature_db_path
        )
        self._calculators['flow_features'] = FlowFeatureCalculator(
            raw_db_path=self.config.raw_db_path,
            feature_db_path=self.config.feature_db_path
        )
        # Other calculators will be added in later phases
        # self._calculators['funding_features'] = FundingFeatureCalculator(...)
        # self._calculators['oi_features'] = OIFeatureCalculator(...)
    
    async def _run_price_features(self):
        """Run price feature calculations at specified interval."""
        category = 'price_features'
        calculator = self._calculators.get(category)
        
        if not calculator:
            logger.error(f"No calculator found for {category}")
            return
        
        interval = self.config.price_features_interval
        
        while self._running:
            start_time = time.time()
            
            try:
                # Calculate for all symbol/exchange combinations
                results = calculator.calculate_all()
                
                # Update stats
                successful = sum(1 for v in results.values() if v)
                failed = sum(1 for v in results.values() if not v)
                
                self._stats[category].total_calculations += len(results)
                self._stats[category].successful_calculations += successful
                self._stats[category].failed_calculations += failed
                self._stats[category].last_calculation_time = datetime.utcnow()
                
                # Calculate average time
                elapsed_ms = (time.time() - start_time) * 1000
                self._stats[category].avg_calculation_time_ms = (
                    self._stats[category].avg_calculation_time_ms * 0.9 + elapsed_ms * 0.1
                )
                
                logger.debug(
                    f"Price features: {successful}/{len(results)} successful, "
                    f"{elapsed_ms:.1f}ms"
                )
                
            except Exception as e:
                logger.error(f"Error in price feature calculation: {e}")
                self._stats[category].errors.append(str(e))
                
                # Check if we should pause
                if len(self._stats[category].errors) >= self.config.max_errors_before_pause:
                    logger.warning(
                        f"Too many errors in {category}, pausing for "
                        f"{self.config.error_pause_seconds}s"
                    )
                    await asyncio.sleep(self.config.error_pause_seconds)
                    self._stats[category].errors.clear()
            
            # Wait for next interval
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)
    
    async def _run_placeholder_calculator(self, category: str, interval: float):
        """Placeholder for future calculators."""
        while self._running:
            logger.debug(f"[{category}] Not implemented yet - skipping")
            await asyncio.sleep(interval)
    
    async def _run_trade_features(self):
        """Run trade feature calculations at specified interval."""
        category = 'trade_features'
        calculator = self._calculators.get(category)
        
        if not calculator:
            logger.error(f"No calculator found for {category}")
            return
        
        interval = self.config.trade_features_interval
        
        while self._running:
            start_time = time.time()
            
            try:
                # Calculate for all symbol/exchange combinations
                results = calculator.calculate_all()
                
                # Update stats
                successful = sum(1 for v in results.values() if v)
                failed = sum(1 for v in results.values() if not v)
                
                self._stats[category].total_calculations += len(results)
                self._stats[category].successful_calculations += successful
                self._stats[category].failed_calculations += failed
                self._stats[category].last_calculation_time = datetime.utcnow()
                
                # Calculate average time
                elapsed_ms = (time.time() - start_time) * 1000
                self._stats[category].avg_calculation_time_ms = (
                    self._stats[category].avg_calculation_time_ms * 0.9 + elapsed_ms * 0.1
                )
                
                logger.debug(
                    f"Trade features: {successful}/{len(results)} successful, "
                    f"{elapsed_ms:.1f}ms"
                )
                
            except Exception as e:
                logger.error(f"Error in trade feature calculation: {e}")
                self._stats[category].errors.append(str(e))
                
                # Check if we should pause
                if len(self._stats[category].errors) >= self.config.max_errors_before_pause:
                    logger.warning(
                        f"Too many errors in {category}, pausing for "
                        f"{self.config.error_pause_seconds}s"
                    )
                    await asyncio.sleep(self.config.error_pause_seconds)
                    self._stats[category].errors.clear()
            
            # Wait for next interval
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)
    
    async def _run_flow_features(self):
        """Run flow feature calculations at specified interval."""
        category = 'flow_features'
        calculator = self._calculators.get(category)
        
        if not calculator:
            logger.error(f"No calculator found for {category}")
            return
        
        interval = self.config.flow_features_interval
        
        while self._running:
            start_time = time.time()
            
            try:
                # Calculate for all symbol/exchange combinations
                results = calculator.calculate_all()
                
                # Update stats
                successful = sum(1 for v in results.values() if v)
                failed = sum(1 for v in results.values() if not v)
                
                self._stats[category].total_calculations += len(results)
                self._stats[category].successful_calculations += successful
                self._stats[category].failed_calculations += failed
                self._stats[category].last_calculation_time = datetime.utcnow()
                
                # Calculate average time
                elapsed_ms = (time.time() - start_time) * 1000
                self._stats[category].avg_calculation_time_ms = (
                    self._stats[category].avg_calculation_time_ms * 0.9 + elapsed_ms * 0.1
                )
                
                logger.debug(
                    f"Flow features: {successful}/{len(results)} successful, "
                    f"{elapsed_ms:.1f}ms"
                )
                
            except Exception as e:
                logger.error(f"Error in flow feature calculation: {e}")
                self._stats[category].errors.append(str(e))
                
                # Check if we should pause
                if len(self._stats[category].errors) >= self.config.max_errors_before_pause:
                    logger.warning(
                        f"Too many errors in {category}, pausing for "
                        f"{self.config.error_pause_seconds}s"
                    )
                    await asyncio.sleep(self.config.error_pause_seconds)
                    self._stats[category].errors.clear()
            
            # Wait for next interval
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)
    
    async def start(self):
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        logger.info("🚀 Starting Feature Scheduler")
        logger.info(f"   Enabled categories: {self.config.enabled_categories}")
        
        # Start tasks for enabled categories
        if 'price_features' in self.config.enabled_categories:
            self._tasks['price_features'] = asyncio.create_task(
                self._run_price_features()
            )
            logger.info(f"   ✓ price_features (every {self.config.price_features_interval}s)")
        
        # Placeholders for future phases
        if 'trade_features' in self.config.enabled_categories:
            self._tasks['trade_features'] = asyncio.create_task(
                self._run_trade_features()
            )
            logger.info(f"   ✓ trade_features (every {self.config.trade_features_interval}s)")
        
        if 'flow_features' in self.config.enabled_categories:
            self._tasks['flow_features'] = asyncio.create_task(
                self._run_flow_features()
            )
            logger.info(f"   ✓ flow_features (every {self.config.flow_features_interval}s)")
        
        if 'funding_features' in self.config.enabled_categories:
            self._tasks['funding_features'] = asyncio.create_task(
                self._run_placeholder_calculator('funding_features', self.config.funding_features_interval)
            )
        
        if 'oi_features' in self.config.enabled_categories:
            self._tasks['oi_features'] = asyncio.create_task(
                self._run_placeholder_calculator('oi_features', self.config.oi_features_interval)
            )
        
        if 'volatility_features' in self.config.enabled_categories:
            self._tasks['volatility_features'] = asyncio.create_task(
                self._run_placeholder_calculator('volatility_features', self.config.volatility_features_interval)
            )
        
        if 'momentum_features' in self.config.enabled_categories:
            self._tasks['momentum_features'] = asyncio.create_task(
                self._run_placeholder_calculator('momentum_features', self.config.momentum_features_interval)
            )
        
        if 'signals' in self.config.enabled_categories:
            self._tasks['signals'] = asyncio.create_task(
                self._run_placeholder_calculator('signals', self.config.signals_interval)
            )
        
        logger.info(f"   Started {len(self._tasks)} calculation tasks")
    
    async def stop(self):
        """Stop the scheduler gracefully."""
        if not self._running:
            return
        
        logger.info("🛑 Stopping Feature Scheduler...")
        self._running = False
        
        # Cancel all tasks
        for name, task in self._tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info(f"   ✓ Stopped {name}")
        
        self._tasks.clear()
        logger.info("   Scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status and statistics."""
        return {
            'running': self._running,
            'enabled_categories': self.config.enabled_categories,
            'active_tasks': list(self._tasks.keys()),
            'statistics': {
                name: {
                    'total_calculations': stats.total_calculations,
                    'successful': stats.successful_calculations,
                    'failed': stats.failed_calculations,
                    'last_calculation': stats.last_calculation_time.isoformat() if stats.last_calculation_time else None,
                    'avg_time_ms': round(stats.avg_calculation_time_ms, 2),
                    'recent_errors': stats.errors[-5:],
                }
                for name, stats in self._stats.items()
            }
        }
    
    def enable_category(self, category: str):
        """Enable a feature category for calculation."""
        if category not in self.config.enabled_categories:
            self.config.enabled_categories.append(category)
            logger.info(f"Enabled category: {category}")
    
    def disable_category(self, category: str):
        """Disable a feature category."""
        if category in self.config.enabled_categories:
            self.config.enabled_categories.remove(category)
            # Cancel the task if running
            if category in self._tasks:
                self._tasks[category].cancel()
                del self._tasks[category]
            logger.info(f"Disabled category: {category}")


async def run_scheduler(categories: List[str] = None):
    """
    Run the feature scheduler as a standalone service.
    
    Args:
        categories: List of feature categories to enable
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    
    # Create config
    config = SchedulerConfig(
        enabled_categories=categories or ['price_features']
    )
    
    # Create scheduler
    scheduler = FeatureScheduler(config)
    
    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    
    def shutdown_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(scheduler.stop())
    
    # Register signal handlers (Unix only)
    try:
        loop.add_signal_handler(signal.SIGINT, shutdown_handler)
        loop.add_signal_handler(signal.SIGTERM, shutdown_handler)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        pass
    
    # Start scheduler
    await scheduler.start()
    
    # Keep running until stopped
    try:
        while scheduler._running:
            await asyncio.sleep(1)
            
            # Print status every 30 seconds
            if int(time.time()) % 30 == 0:
                status = scheduler.get_status()
                total = sum(s['total_calculations'] for s in status['statistics'].values())
                successful = sum(s['successful'] for s in status['statistics'].values())
                logger.info(f"📊 Status: {successful}/{total} calculations successful")
    except asyncio.CancelledError:
        pass
    finally:
        await scheduler.stop()


def main():
    """Entry point for running the scheduler."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature Calculation Scheduler')
    parser.add_argument(
        '--categories', '-c',
        nargs='+',
        default=['price_features', 'trade_features', 'flow_features'],
        help='Feature categories to enable'
    )
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        FEATURE CALCULATION SCHEDULER                          ║
║                                                                               ║
║  This service calculates features from raw data and stores them              ║
║  in the feature database for Streamlit consumption.                          ║
║                                                                               ║
║  Press Ctrl+C to stop                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    asyncio.run(run_scheduler(args.categories))


if __name__ == "__main__":
    main()
