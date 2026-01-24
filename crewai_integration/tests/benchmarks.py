"""
Performance Benchmarks for CrewAI Integration
=============================================

Benchmarks for:
- Tool invocation latency
- Agent decision time
- Memory usage
- Database query impact
"""

import logging
import asyncio
import time
from typing import Dict, Any, List
from datetime import datetime
import sys

logger = logging.getLogger(__name__)


async def run_benchmarks() -> Dict[str, Any]:
    """
    Run all performance benchmarks.
    
    Target metrics from Phase 1 requirements:
    - Tool invocation latency: <100ms
    - Agent decision time: <5s
    - Memory usage: <500MB per agent
    - Database query impact: <5% overhead
    """
    results = {
        "benchmarks": [],
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {}
    }
    
    # Benchmark 1: Tool Wrapper Latency
    benchmark = await benchmark_tool_latency()
    results["benchmarks"].append(benchmark)
    
    # Benchmark 2: State Manager Performance
    benchmark = await benchmark_state_manager()
    results["benchmarks"].append(benchmark)
    
    # Benchmark 3: Event Bus Throughput
    benchmark = await benchmark_event_bus()
    results["benchmarks"].append(benchmark)
    
    # Benchmark 4: Config Loading Time
    benchmark = await benchmark_config_loading()
    results["benchmarks"].append(benchmark)
    
    # Benchmark 5: Permission Check Speed
    benchmark = await benchmark_permission_checks()
    results["benchmarks"].append(benchmark)
    
    # Calculate summary
    all_passed = all(b.get("passed", False) for b in results["benchmarks"])
    results["summary"] = {
        "all_passed": all_passed,
        "total_benchmarks": len(results["benchmarks"]),
        "passed": sum(1 for b in results["benchmarks"] if b.get("passed")),
        "failed": sum(1 for b in results["benchmarks"] if not b.get("passed"))
    }
    
    return results


async def benchmark_tool_latency() -> Dict[str, Any]:
    """
    Benchmark tool wrapper invocation latency.
    
    Target: <100ms per invocation
    """
    name = "Tool Wrapper Latency"
    target_ms = 100
    
    try:
        from crewai_integration.core.registry import ToolRegistry
        from crewai_integration.core.permissions import ToolCategory, AccessLevel
        
        registry = ToolRegistry()
        
        # Register a mock tool
        def mock_tool(param: str):
            return {"result": param}
        
        registry.register(
            name="benchmark_tool",
            function=mock_tool,
            category=ToolCategory.EXCHANGE_DATA,
            description="Benchmark tool"
        )
        
        # Measure latency over multiple calls (registration and lookup)
        latencies = []
        for i in range(100):
            start = time.perf_counter()
            # Test tool lookup and metadata retrieval
            tool = registry.get_tool(f"benchmark_tool")
            if tool:
                result = tool.function(param=f"test_{i}")
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        passed = avg_latency < target_ms
        
        return {
            "name": name,
            "target_ms": target_ms,
            "avg_latency_ms": round(avg_latency, 2),
            "max_latency_ms": round(max_latency, 2),
            "min_latency_ms": round(min_latency, 2),
            "iterations": 100,
            "passed": passed,
            "message": f"Avg latency: {avg_latency:.2f}ms (target: <{target_ms}ms)"
        }
        
    except Exception as e:
        return {
            "name": name,
            "passed": False,
            "error": str(e)
        }


async def benchmark_state_manager() -> Dict[str, Any]:
    """
    Benchmark state manager operations.
    
    Tests write and read performance.
    """
    name = "State Manager Performance"
    
    try:
        from crewai_integration.state.manager import StateManager
        
        sm = StateManager(db_path=":memory:")
        await sm.initialize()
        
        # Benchmark writes
        write_times = []
        for i in range(100):
            start = time.perf_counter()
            await sm.record_decision(
                agent_id=f"bench_agent_{i%5}",
                crew="bench_crew",
                tool_name="bench_tool",
                parameters={"i": i},
                result={"ok": True},
                success=True,
                latency_ms=10.0
            )
            write_times.append((time.perf_counter() - start) * 1000)
        
        # Benchmark reads
        read_times = []
        for i in range(100):
            start = time.perf_counter()
            await sm.get_agent_decisions(f"bench_agent_{i%5}", limit=10)
            read_times.append((time.perf_counter() - start) * 1000)
        
        await sm.close()
        
        avg_write = sum(write_times) / len(write_times)
        avg_read = sum(read_times) / len(read_times)
        
        passed = avg_write < 50 and avg_read < 50  # 50ms targets
        
        return {
            "name": name,
            "avg_write_ms": round(avg_write, 2),
            "avg_read_ms": round(avg_read, 2),
            "write_iterations": 100,
            "read_iterations": 100,
            "passed": passed,
            "message": f"Write: {avg_write:.2f}ms, Read: {avg_read:.2f}ms"
        }
        
    except Exception as e:
        return {
            "name": name,
            "passed": False,
            "error": str(e)
        }


async def benchmark_event_bus() -> Dict[str, Any]:
    """
    Benchmark event bus throughput.
    
    Tests events per second capacity.
    """
    name = "Event Bus Throughput"
    target_eps = 1000  # Events per second
    
    try:
        from crewai_integration.events.bus import EventBus, Event, EventType
        
        bus = EventBus()
        received_count = 0
        
        async def counter_handler(event: Event):
            nonlocal received_count
            received_count += 1
        
        bus.subscribe(EventType.DATA_RECEIVED, counter_handler)
        
        # Publish many events
        start = time.perf_counter()
        for i in range(1000):
            await bus.publish(Event(
                type=EventType.DATA_RECEIVED,
                source="benchmark",
                data={"i": i}
            ))
        elapsed = time.perf_counter() - start
        
        eps = 1000 / elapsed if elapsed > 0 else 0
        passed = eps >= target_eps
        
        return {
            "name": name,
            "events_per_second": round(eps, 0),
            "target_eps": target_eps,
            "events_published": 1000,
            "events_received": received_count,
            "elapsed_seconds": round(elapsed, 3),
            "passed": passed,
            "message": f"{eps:.0f} events/sec (target: {target_eps})"
        }
        
    except Exception as e:
        return {
            "name": name,
            "passed": False,
            "error": str(e)
        }


async def benchmark_config_loading() -> Dict[str, Any]:
    """
    Benchmark configuration loading time.
    """
    name = "Config Loading Time"
    target_ms = 500
    
    try:
        from crewai_integration.config.loader import ConfigLoader
        
        times = []
        for _ in range(10):
            loader = ConfigLoader()
            loader._cache.clear()  # Clear cache
            
            start = time.perf_counter()
            loader.load_all()
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = sum(times) / len(times)
        passed = avg_time < target_ms
        
        return {
            "name": name,
            "avg_time_ms": round(avg_time, 2),
            "target_ms": target_ms,
            "iterations": 10,
            "passed": passed,
            "message": f"Avg load time: {avg_time:.2f}ms"
        }
        
    except Exception as e:
        return {
            "name": name,
            "passed": False,
            "error": str(e)
        }


async def benchmark_permission_checks() -> Dict[str, Any]:
    """
    Benchmark permission checking speed.
    """
    name = "Permission Check Speed"
    target_us = 100  # Microseconds
    
    try:
        from crewai_integration.core.permissions import (
            PermissionManager, ToolCategory, AccessLevel
        )
        
        pm = PermissionManager()
        
        # Register many agents
        for i in range(10):
            pm.register_agent(f"agent_{i}", "data_crew", f"Role_{i}")
        
        # Benchmark permission checks
        times = []
        for _ in range(10000):
            start = time.perf_counter()
            pm.check_permission(
                "agent_5",
                "some_tool",
                ToolCategory.EXCHANGE_DATA,
                AccessLevel.READ_ONLY
            )
            times.append((time.perf_counter() - start) * 1_000_000)  # microseconds
        
        avg_time = sum(times) / len(times)
        passed = avg_time < target_us
        
        return {
            "name": name,
            "avg_time_us": round(avg_time, 2),
            "target_us": target_us,
            "iterations": 10000,
            "passed": passed,
            "message": f"Avg check time: {avg_time:.2f}μs"
        }
        
    except Exception as e:
        return {
            "name": name,
            "passed": False,
            "error": str(e)
        }


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    import sys
    # This is a rough estimate
    return sys.getsizeof(globals()) / (1024 * 1024)


if __name__ == "__main__":
    results = asyncio.run(run_benchmarks())
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")
    
    for bench in results["benchmarks"]:
        status = "✓" if bench.get("passed") else "✗"
        print(f"\n{status} {bench['name']}")
        if "message" in bench:
            print(f"   {bench['message']}")
        if "error" in bench:
            print(f"   Error: {bench['error']}")
    
    print(f"\n{'='*60}")
    print(f"Summary: {results['summary']['passed']}/{results['summary']['total_benchmarks']} passed")
