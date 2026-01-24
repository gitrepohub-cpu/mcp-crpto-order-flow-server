#!/usr/bin/env python3
"""
Wrapper Diagnostic Tool

Command-line tool for testing and diagnosing individual tool wrappers.

Usage:
    python wrapper_diagnostic.py                     # Test all wrappers
    python wrapper_diagnostic.py --wrapper exchange  # Test specific wrapper
    python wrapper_diagnostic.py --tool ticker       # Test specific tool
    python wrapper_diagnostic.py --invoke            # Test actual invocation
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("wrapper_diagnostic")


WRAPPER_CLASSES = {
    "exchange": ("ExchangeDataTools", "crewai_integration.tools.wrappers"),
    "forecasting": ("ForecastingTools", "crewai_integration.tools.wrappers"),
    "analytics": ("AnalyticsTools", "crewai_integration.tools.wrappers"),
    "streaming": ("StreamingTools", "crewai_integration.tools.wrappers"),
    "features": ("FeatureTools", "crewai_integration.tools.wrappers"),
    "visualization": ("VisualizationTools", "crewai_integration.tools.wrappers"),
}


def get_wrapper_class(name: str):
    """Dynamically import and return a wrapper class."""
    if name not in WRAPPER_CLASSES:
        raise ValueError(f"Unknown wrapper: {name}. Available: {list(WRAPPER_CLASSES.keys())}")
    
    class_name, module_name = WRAPPER_CLASSES[name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def diagnose_wrapper(wrapper_name: str, verbose: bool = False) -> Dict[str, Any]:
    """Diagnose a single wrapper."""
    results = {
        "wrapper": wrapper_name,
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    try:
        # Import check
        wrapper_class = get_wrapper_class(wrapper_name)
        results["checks"]["import"] = {"status": "OK", "class": wrapper_class.__name__}
        
        # Shadow mode initialization
        try:
            wrapper = wrapper_class(shadow_mode=True)
            results["checks"]["shadow_init"] = {"status": "OK"}
        except Exception as e:
            results["checks"]["shadow_init"] = {"status": "FAIL", "error": str(e)}
            return results
        
        # Normal mode initialization
        try:
            wrapper_normal = wrapper_class(shadow_mode=False)
            results["checks"]["normal_init"] = {"status": "OK"}
        except Exception as e:
            results["checks"]["normal_init"] = {"status": "FAIL", "error": str(e)}
        
        # Health check
        try:
            health = wrapper.health_check()
            results["checks"]["health_check"] = {
                "status": "OK",
                "operational": health.get("operational", False),
                "tools_count": health.get("tools_count", 0),
                "shadow_mode": health.get("shadow_mode", None)
            }
        except Exception as e:
            results["checks"]["health_check"] = {"status": "FAIL", "error": str(e)}
        
        # Tool listing
        try:
            tools = wrapper.list_tools()
            results["checks"]["list_tools"] = {
                "status": "OK",
                "count": len(tools),
                "sample": tools[:5] if len(tools) > 5 else tools
            }
        except Exception as e:
            results["checks"]["list_tools"] = {"status": "FAIL", "error": str(e)}
        
        # Statistics
        try:
            stats = wrapper.get_statistics()
            results["checks"]["statistics"] = {
                "status": "OK",
                "success_count": stats.get("success_count", 0),
                "error_count": stats.get("error_count", 0)
            }
        except Exception as e:
            results["checks"]["statistics"] = {"status": "FAIL", "error": str(e)}
        
        # is_shadow_mode method
        try:
            is_shadow = wrapper.is_shadow_mode()
            results["checks"]["is_shadow_mode"] = {
                "status": "OK",
                "value": is_shadow
            }
        except Exception as e:
            results["checks"]["is_shadow_mode"] = {"status": "FAIL", "error": str(e)}
        
        # Calculate overall status
        all_ok = all(
            check.get("status") == "OK" 
            for check in results["checks"].values()
        )
        results["overall_status"] = "PASS" if all_ok else "PARTIAL"
        
    except Exception as e:
        results["checks"]["import"] = {"status": "FAIL", "error": str(e)}
        results["overall_status"] = "FAIL"
    
    return results


async def test_tool_invocation(
    wrapper_name: str,
    tool_name: str,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Test invoking a specific tool."""
    results = {
        "wrapper": wrapper_name,
        "tool": tool_name,
        "params": params or {},
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        wrapper_class = get_wrapper_class(wrapper_name)
        wrapper = wrapper_class(shadow_mode=True)
        
        # Check if tool exists
        if tool_name not in wrapper.list_tools():
            results["status"] = "FAIL"
            results["error"] = f"Tool '{tool_name}' not found in {wrapper_name}"
            return results
        
        # Invoke tool
        result = await wrapper.invoke(
            tool_name=tool_name,
            **(params or {})
        )
        
        results["status"] = "OK" if result.get("success") else "FAIL"
        results["result"] = result
        
    except Exception as e:
        results["status"] = "FAIL"
        results["error"] = str(e)
    
    return results


def print_diagnostic_report(results: Dict[str, Any]):
    """Print formatted diagnostic report."""
    print("\n" + "=" * 60)
    print(f"Wrapper Diagnostic Report: {results['wrapper']}")
    print(f"Time: {results['timestamp']}")
    print("=" * 60)
    
    for check_name, check_result in results.get("checks", {}).items():
        status = check_result.get("status", "UNKNOWN")
        status_symbol = "[OK]" if status == "OK" else "[FAIL]"
        print(f"\n{status_symbol} {check_name}:")
        
        for key, value in check_result.items():
            if key != "status":
                print(f"    {key}: {value}")
    
    print("\n" + "-" * 60)
    overall = results.get("overall_status", "UNKNOWN")
    print(f"Overall Status: {overall}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Wrapper Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python wrapper_diagnostic.py                     # Test all wrappers
    python wrapper_diagnostic.py --wrapper exchange  # Test specific wrapper
    python wrapper_diagnostic.py --list              # List all available wrappers
    python wrapper_diagnostic.py --tools exchange    # List tools in wrapper
    python wrapper_diagnostic.py --invoke exchange binance_get_ticker
        """
    )
    
    parser.add_argument(
        "--wrapper", "-w",
        help="Specific wrapper to diagnose"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available wrappers"
    )
    parser.add_argument(
        "--tools", "-t",
        help="List all tools in a specific wrapper"
    )
    parser.add_argument(
        "--invoke", "-i",
        nargs=2,
        metavar=("WRAPPER", "TOOL"),
        help="Test invoking a specific tool"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    # List wrappers
    if args.list:
        print("\nAvailable Wrappers:")
        print("-" * 40)
        for name, (class_name, module) in WRAPPER_CLASSES.items():
            try:
                wrapper_class = get_wrapper_class(name)
                wrapper = wrapper_class(shadow_mode=True)
                tools_count = len(wrapper.list_tools())
                print(f"  {name:15} - {class_name} ({tools_count} tools)")
            except Exception as e:
                print(f"  {name:15} - ERROR: {e}")
        return
    
    # List tools in wrapper
    if args.tools:
        try:
            wrapper_class = get_wrapper_class(args.tools)
            wrapper = wrapper_class(shadow_mode=True)
            tools = wrapper.list_tools()
            print(f"\nTools in {args.tools} ({len(tools)} total):")
            print("-" * 40)
            for tool in tools:
                print(f"  - {tool}")
        except Exception as e:
            print(f"Error: {e}")
        return
    
    # Test invocation
    if args.invoke:
        wrapper_name, tool_name = args.invoke
        results = asyncio.run(test_tool_invocation(wrapper_name, tool_name))
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\nInvocation Test: {wrapper_name}/{tool_name}")
            print("-" * 40)
            print(f"Status: {results['status']}")
            if "error" in results:
                print(f"Error: {results['error']}")
            if "result" in results:
                print(f"Result: {json.dumps(results['result'], indent=2)}")
        return
    
    # Diagnose specific or all wrappers
    wrappers_to_test = [args.wrapper] if args.wrapper else list(WRAPPER_CLASSES.keys())
    
    all_results = []
    for wrapper_name in wrappers_to_test:
        results = diagnose_wrapper(wrapper_name, verbose=args.verbose)
        all_results.append(results)
        
        if not args.json:
            print_diagnostic_report(results)
    
    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        for result in all_results:
            status = result.get("overall_status", "UNKNOWN")
            symbol = "[PASS]" if status == "PASS" else "[FAIL]"
            wrapper = result.get("wrapper", "unknown")
            
            # Get tool count
            health = result.get("checks", {}).get("health_check", {})
            tools = health.get("tools_count", "?")
            
            print(f"{symbol} {wrapper:15} - {tools} tools")
        
        # Overall
        passed = sum(1 for r in all_results if r.get("overall_status") == "PASS")
        total = len(all_results)
        print("-" * 60)
        print(f"Total: {passed}/{total} wrappers passing")


if __name__ == "__main__":
    main()
