"""
Stream Coordinator - Manages multiple StreamWorkers
==================================================
Handles worker lifecycle, health monitoring, auto-restart, and aggregated stats.
"""

import asyncio
import signal
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .config import DistributedConfig, WorkerConfig
from .worker import StreamWorker, WorkerStats

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorStats:
    """Aggregated stats across all workers."""
    total_workers: int = 0
    healthy_workers: int = 0
    total_requests: int = 0
    total_failures: int = 0
    total_rate_limits: int = 0
    total_records: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def uptime_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()
    
    @property
    def records_per_minute(self) -> float:
        mins = self.uptime_seconds / 60
        return self.total_records / mins if mins > 0 else 0


class StreamCoordinator:
    """
    Coordinates multiple StreamWorkers for distributed data collection.
    
    Features:
    - Spawns workers for each symbol-exchange pair
    - Health monitoring with auto-restart
    - Aggregated statistics
    - Graceful shutdown on signals
    """
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.workers: Dict[str, StreamWorker] = {}
        self.worker_tasks: Dict[str, asyncio.Task] = {}
        self.worker_restarts: Dict[str, int] = {}
        
        self._running = False
        self._stop_event: Optional[asyncio.Event] = None  # Created in start()
        self._health_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        
        self.stats = CoordinatorStats()
        
        logger.info(f"[COORDINATOR] Initialized with {config.total_workers} workers planned")
    
    def _create_workers(self):
        """Create StreamWorker instances from config."""
        worker_configs = self.config.generate_worker_configs()
        
        for wc in worker_configs:
            worker = StreamWorker(wc)
            self.workers[wc.worker_id] = worker
            self.worker_restarts[wc.worker_id] = 0
        
        self.stats.total_workers = len(self.workers)
        logger.info(f"[COORDINATOR] Created {len(self.workers)} workers")
    
    async def _run_worker(self, worker_id: str):
        """Run a single worker with auto-restart on failure."""
        worker = self.workers.get(worker_id)
        if not worker:
            return
        
        while self._running and not self._stop_event.is_set():
            try:
                await worker.start()
            except asyncio.CancelledError:
                logger.info(f"[COORDINATOR] Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"[COORDINATOR] Worker {worker_id} crashed: {e}")
                
                # Track restarts
                self.worker_restarts[worker_id] += 1
                if self.worker_restarts[worker_id] > self.config.max_worker_restarts:
                    logger.error(f"[COORDINATOR] Worker {worker_id} exceeded max restarts, stopping")
                    break
                
                # Exponential backoff before restart
                backoff = min(30, 2 ** self.worker_restarts[worker_id])
                logger.info(f"[COORDINATOR] Restarting {worker_id} in {backoff}s...")
                await asyncio.sleep(backoff)
                
                # Recreate worker
                wc = worker.config
                self.workers[worker_id] = StreamWorker(wc)
        
        logger.info(f"[COORDINATOR] Worker {worker_id} loop exited")
    
    async def _health_monitor(self):
        """Monitor worker health periodically."""
        while self._running and not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                healthy = 0
                unhealthy = []
                
                for worker_id, worker in self.workers.items():
                    if worker.is_healthy():
                        healthy += 1
                    else:
                        unhealthy.append(worker_id)
                
                self.stats.healthy_workers = healthy
                
                if unhealthy:
                    logger.warning(f"[HEALTH] Unhealthy workers: {unhealthy}")
                else:
                    logger.info(f"[HEALTH] All {healthy} workers healthy")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[HEALTH] Monitor error: {e}")
    
    async def _stats_reporter(self):
        """Report aggregated statistics periodically."""
        while self._running and not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.config.stats_report_interval)
                
                # Aggregate stats from all workers
                total_requests = 0
                total_failures = 0
                total_rate_limits = 0
                total_records = 0
                
                for worker in self.workers.values():
                    total_requests += worker.stats.requests_made
                    total_failures += worker.stats.requests_failed
                    total_rate_limits += worker.stats.rate_limit_hits
                    total_records += worker.stats.records_inserted
                
                self.stats.total_requests = total_requests
                self.stats.total_failures = total_failures
                self.stats.total_rate_limits = total_rate_limits
                self.stats.total_records = total_records
                
                # Log summary
                success_rate = ((total_requests - total_failures) / total_requests * 100) if total_requests > 0 else 0
                logger.info(
                    f"[STATS] "
                    f"Records: {total_records} ({self.stats.records_per_minute:.1f}/min) | "
                    f"Requests: {total_requests} ({success_rate:.1f}% success) | "
                    f"Rate limits: {total_rate_limits} | "
                    f"Workers: {self.stats.healthy_workers}/{self.stats.total_workers} healthy"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[STATS] Reporter error: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        # On Windows, signal handling is limited - just rely on KeyboardInterrupt
        # in the main() function for Ctrl+C handling
        pass
    
    async def start(self, duration: Optional[float] = None):
        """
        Start the coordinator and all workers.
        
        Args:
            duration: Optional runtime in minutes (None = run forever)
        """
        logger.info("[COORDINATOR] Starting distributed streaming...")
        
        self._running = True
        # Create the event inside the async context
        self._stop_event = asyncio.Event()
        
        # Create workers
        self._create_workers()
        
        # Setup signal handlers
        try:
            self._setup_signal_handlers()
        except Exception as e:
            logger.warning(f"[COORDINATOR] Could not setup signal handlers: {e}")
        
        # Start health monitor and stats reporter
        self._health_task = asyncio.create_task(self._health_monitor())
        self._stats_task = asyncio.create_task(self._stats_reporter())
        
        # Start all workers
        for worker_id in self.workers:
            task = asyncio.create_task(self._run_worker(worker_id))
            self.worker_tasks[worker_id] = task
        
        logger.info(f"[COORDINATOR] Started {len(self.worker_tasks)} worker tasks")
        
        # If duration specified, schedule shutdown
        if duration:
            duration_seconds = duration * 60
            logger.info(f"[COORDINATOR] Will run for {duration} minutes ({duration_seconds}s)")
            await asyncio.sleep(duration_seconds)
            await self.stop()
        else:
            # Run forever until stopped
            await self._stop_event.wait()
        
        logger.info("[COORDINATOR] Main loop exited")
    
    async def stop(self):
        """Stop all workers gracefully."""
        if not self._running:
            return
        
        logger.info("[COORDINATOR] Stopping all workers...")
        self._running = False
        self._stop_event.set()
        
        # Stop all workers
        stop_tasks = []
        for worker in self.workers.values():
            stop_tasks.append(worker.stop())
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Cancel worker tasks
        for task in self.worker_tasks.values():
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks.values(), return_exceptions=True)
        
        # Cancel monitor tasks
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
        if self._stats_task and not self._stats_task.done():
            self._stats_task.cancel()
        
        # Final stats
        logger.info(
            f"[COORDINATOR] Shutdown complete. "
            f"Total records: {self.stats.total_records}, "
            f"Total requests: {self.stats.total_requests}, "
            f"Rate limit hits: {self.stats.total_rate_limits}"
        )
    
    def get_worker_stats(self) -> List[Dict]:
        """Get stats for all workers."""
        return [w.stats.to_dict() for w in self.workers.values()]
    
    def get_summary(self) -> Dict:
        """Get coordinator summary."""
        return {
            "total_workers": self.stats.total_workers,
            "healthy_workers": self.stats.healthy_workers,
            "total_requests": self.stats.total_requests,
            "total_failures": self.stats.total_failures,
            "total_rate_limits": self.stats.total_rate_limits,
            "total_records": self.stats.total_records,
            "records_per_minute": f"{self.stats.records_per_minute:.1f}",
            "uptime_seconds": int(self.stats.uptime_seconds),
        }
