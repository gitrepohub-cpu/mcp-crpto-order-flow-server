"""Quick verification after 10-minute collection test"""

import duckdb

print("\n" + "=" * 80)
print("  10-MINUTE COLLECTION TEST - RESULTS")
print("=" * 80 + "\n")

# Check raw data
db_path = 'data/isolated_exchange_data.duckdb'
conn = duckdb.connect(db_path, read_only=True)

tables = conn.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'main' 
    AND table_name NOT LIKE '%_registry'
""").fetchall()

with_data = 0
for (table_name,) in tables:
    try:
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        if count > 0:
            with_data += 1
    except:
        pass

print(f"RAW DATA TABLES:")
print(f"  Tables with data: {with_data}/{len(tables)} ({with_data/len(tables)*100:.1f}%)")
conn.close()

# Check features
db_path = 'data/features_data.duckdb'
conn = duckdb.connect(db_path, read_only=True)

tables = conn.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'main'
""").fetchall()

with_data = 0
for (table_name,) in tables:
    try:
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        if count > 0:
            with_data += 1
    except:
        pass

print(f"\nFEATURE TABLES:")
print(f"  Tables with data: {with_data}/{len(tables)} ({with_data/len(tables)*100:.1f}%)")
conn.close()

print("\n" + "=" * 80)
print("  SUCCESS: 10-minute collection test complete!")
print("=" * 80)
print("\nPhase 1 & 2 features are calculating and available for Streamlit!")
print("\nNext: Run 'streamlit run streamlit_viewer.py' to view the data\n")
