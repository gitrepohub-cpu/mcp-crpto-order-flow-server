"""
CrewAI Integration Verification Script
=======================================

Run this script to verify Phase 1 installation is complete.

Usage:
    python verify_crewai_integration.py
"""

import sys
import os

def check_imports():
    """Check all modules can be imported."""
    print("\n📦 Checking imports...")
    errors = []
    
    modules = [
        ("crewai_integration", "Main module"),
        ("crewai_integration.core.permissions", "Permission system"),
        ("crewai_integration.core.registry", "Tool registry"),
        ("crewai_integration.core.controller", "Controller"),
        ("crewai_integration.tools.base", "Tool base"),
        ("crewai_integration.tools.wrappers", "Tool wrappers"),
        ("crewai_integration.state.manager", "State manager"),
        ("crewai_integration.state.schemas", "State schemas"),
        ("crewai_integration.config.loader", "Config loader"),
        ("crewai_integration.config.schemas", "Config schemas"),
        ("crewai_integration.events.bus", "Event bus"),
        ("crewai_integration.tests.unit_tests", "Unit tests"),
        ("crewai_integration.tests.integration_tests", "Integration tests"),
        ("crewai_integration.tests.simulation", "Simulation"),
        ("crewai_integration.tests.benchmarks", "Benchmarks"),
    ]
    
    for module, desc in modules:
        try:
            __import__(module)
            print(f"  ✓ {desc}")
        except Exception as e:
            print(f"  ✗ {desc}: {e}")
            errors.append((module, str(e)))
    
    return len(errors) == 0, errors


def check_files():
    """Check all required files exist."""
    print("\n📂 Checking files...")
    
    base = os.path.dirname(os.path.abspath(__file__))
    files = [
        "crewai_integration/__init__.py",
        "crewai_integration/core/__init__.py",
        "crewai_integration/core/permissions.py",
        "crewai_integration/core/registry.py",
        "crewai_integration/core/controller.py",
        "crewai_integration/tools/__init__.py",
        "crewai_integration/tools/base.py",
        "crewai_integration/tools/wrappers.py",
        "crewai_integration/state/__init__.py",
        "crewai_integration/state/manager.py",
        "crewai_integration/state/schemas.py",
        "crewai_integration/config/__init__.py",
        "crewai_integration/config/loader.py",
        "crewai_integration/config/schemas.py",
        "crewai_integration/config/system.yaml",
        "crewai_integration/config/agents.yaml",
        "crewai_integration/config/tasks.yaml",
        "crewai_integration/config/crews.yaml",
        "crewai_integration/events/__init__.py",
        "crewai_integration/events/bus.py",
        "crewai_integration/tests/__init__.py",
        "crewai_integration/tests/unit_tests.py",
        "crewai_integration/tests/integration_tests.py",
        "crewai_integration/tests/simulation.py",
        "crewai_integration/tests/benchmarks.py",
        "crewai_integration/docs/README.md",
        "crewai_integration/docs/TOOL_WRAPPER_REFERENCE.md",
        "crewai_integration/docs/STATE_MANAGEMENT_GUIDE.md",
    ]
    
    missing = []
    for f in files:
        path = os.path.join(base, f)
        if os.path.exists(path):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f}")
            missing.append(f)
    
    return len(missing) == 0, missing


def check_dependencies():
    """Check required dependencies."""
    print("\n📚 Checking dependencies...")
    
    deps = [
        ("duckdb", "DuckDB"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("aiohttp", "aiohttp"),
    ]
    
    missing = []
    for module, name in deps:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (not installed)")
            missing.append(module)
    
    # Check optional deps
    optional = [
        ("crewai", "CrewAI"),
        ("yaml", "PyYAML"),
    ]
    
    print("\n  Optional dependencies:")
    for module, name in optional:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name} (not installed - needed for full functionality)")
    
    return len(missing) == 0, missing


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("CrewAI Integration - Phase 1 Verification")
    print("=" * 60)
    
    results = []
    
    # Check files
    files_ok, missing_files = check_files()
    results.append(("Files", files_ok))
    
    # Check dependencies
    deps_ok, missing_deps = check_dependencies()
    results.append(("Dependencies", deps_ok))
    
    # Check imports
    imports_ok, import_errors = check_imports()
    results.append(("Imports", imports_ok))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ Phase 1 verification PASSED!")
        print("\nNext steps:")
        print("  1. Install CrewAI: pip install crewai pyyaml")
        print("  2. Run tests: python -m crewai_integration.tests.unit_tests")
        print("  3. Run benchmarks: python -m crewai_integration.tests.benchmarks")
        print("  4. Start in shadow mode for 48-hour validation")
        return 0
    else:
        print("✗ Phase 1 verification FAILED")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
