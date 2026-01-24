@echo off
echo Starting 11-minute distributed streaming collection...
echo.
echo Process will run in background. Check data/distributed_streaming.duckdb for results.
echo.
start /B pythonw run_11min_collection.py > nul 2>&1
echo Started! Process ID will show in Task Manager as pythonw.exe
echo.
echo To check status, run: python check_streaming_status.py
echo To stop early: taskkill /IM pythonw.exe /F
