@echo off
echo ========================================
echo Starting Production Collector
echo Connecting to ALL 9 exchanges...
echo ========================================
echo.
python src/storage/production_isolated_collector.py
pause
