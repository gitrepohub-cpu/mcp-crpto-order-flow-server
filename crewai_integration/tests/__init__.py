"""
Testing Framework for CrewAI Integration
========================================

Provides comprehensive testing capabilities:
- Unit tests for tool wrappers
- Integration tests for crews
- Simulation and shadow modes
- Performance benchmarks
"""

from .unit_tests import run_wrapper_tests, run_permission_tests
from .integration_tests import run_integration_tests
from .simulation import SimulationRunner, ShadowModeRunner
from .benchmarks import run_benchmarks

__all__ = [
    "run_wrapper_tests",
    "run_permission_tests",
    "run_integration_tests",
    "SimulationRunner",
    "ShadowModeRunner",
    "run_benchmarks",
]
