#!/usr/bin/env python3
"""Quick test to verify wrapper initialization fixes."""

from crewai_integration.tools.wrappers import (
    ExchangeDataTools,
    ForecastingTools,
    AnalyticsTools,
    StreamingTools,
    FeatureTools,
    VisualizationTools
)

def main():
    print("=" * 60)
    print("Testing Wrapper Initialization with shadow_mode=True")
    print("=" * 60)
    
    wrappers = [
        ('ExchangeDataTools', ExchangeDataTools),
        ('ForecastingTools', ForecastingTools),
        ('AnalyticsTools', AnalyticsTools),
        ('StreamingTools', StreamingTools),
        ('FeatureTools', FeatureTools),
        ('VisualizationTools', VisualizationTools)
    ]
    
    all_passed = True
    for name, cls in wrappers:
        try:
            w = cls(shadow_mode=True)
            h = w.health_check()
            
            # Verify shadow mode
            if not w.is_shadow_mode():
                raise ValueError("shadow_mode not set correctly")
            
            print(f"[OK] {name}: {h['tools_count']} tools, shadow_mode={h['shadow_mode']}")
            
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("SUCCESS: All 6 wrappers initialized correctly!")
    else:
        print("FAILED: Some wrappers had issues")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
