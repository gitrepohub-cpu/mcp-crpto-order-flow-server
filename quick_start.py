"""
🚀 QUICK START - Production Data Collection
===========================================

This script:
1. Checks current database status
2. Guides you through starting the collector
3. Provides next steps

Run this FIRST before anything else.
"""

import duckdb
import sys
from pathlib import Path

def check_database():
    """Check database status."""
    db_path = "data/isolated_exchange_data.duckdb"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return None
    
    try:
        conn = duckdb.connect(db_path, read_only=True)
        
        # Count tables
        tables = conn.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = 'main' 
            AND table_name NOT LIKE '%_registry'
        """).fetchone()[0]
        
        # Count tables with data
        all_tables = conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main' 
            AND table_name NOT LIKE '%_registry'
        """).fetchall()
        
        with_data = 0
        for (table_name,) in all_tables:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                if count > 0:
                    with_data += 1
            except:
                pass
        
        conn.close()
        
        return {
            'total_tables': tables,
            'tables_with_data': with_data,
            'coverage': (with_data / tables * 100) if tables > 0 else 0
        }
    except Exception as e:
        print(f"❌ Error accessing database: {e}")
        return None

def main():
    """Main quick start flow."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 PRODUCTION COLLECTOR - QUICK START                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check database
    print("📊 Checking current database status...")
    stats = check_database()
    
    if not stats:
        print("\n❌ Cannot access database. Please check file path.")
        return
    
    print(f"\n✅ Database found!")
    print(f"   Total tables: {stats['total_tables']}")
    print(f"   Tables with data: {stats['tables_with_data']}")
    print(f"   Coverage: {stats['coverage']:.1f}%")
    
    # Determine status
    if stats['coverage'] < 50:
        status = "🔴 CRITICAL"
        message = "Very low coverage. Collector likely not running or misconfigured."
    elif stats['coverage'] < 80:
        status = "🟡 WARNING"
        message = "Moderate coverage. Some exchanges missing or disconnected."
    elif stats['coverage'] < 95:
        status = "🟢 GOOD"
        message = "Good coverage. Some optional data types missing (normal)."
    else:
        status = "✅ EXCELLENT"
        message = "Excellent coverage. System working optimally."
    
    print(f"\n{status}")
    print(f"   {message}")
    
    # Recommendations
    print(f"\n" + "=" * 80)
    print("📋 NEXT STEPS:")
    print("=" * 80)
    
    if stats['coverage'] < 95:
        print("""
1️⃣ START PRODUCTION COLLECTOR (ALL 9 EXCHANGES):
   
   Double-click: start_production_collector.bat
   
   Or run in PowerShell:
   > .\\start_production_collector.bat

2️⃣ WAIT 2-3 MINUTES
   
   Let the collector establish connections and start streaming data.

3️⃣ MONITOR IN REAL-TIME:
   
   Open a NEW PowerShell window and run:
   > python monitor_collection.py
   
   This shows live updates every 10 seconds.

4️⃣ VERIFY AFTER 5 MINUTES:
   
   > python comprehensive_exchange_audit.py
   
   Should show 500+ tables with data (99%+ coverage).

5️⃣ CHECK FEATURES:
   
   > python debug_phase1_phase2.py
   
   Verify Phase 1 & 2 features are calculating.
        """)
    else:
        print("""
✅ Your system is already collecting data well!

MAINTENANCE:
   - Monitor: python monitor_collection.py
   - Check status: python comprehensive_exchange_audit.py
   - View features: python debug_phase1_phase2.py

If you see coverage drop below 90%, restart the collector.
        """)
    
    print("\n" + "=" * 80)
    print("📖 For detailed instructions, see: PRODUCTION_SETUP_GUIDE.md")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
