"""Integration test for demo_pipeline.py"""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_demo_pipeline_runs():
    """Test that demo pipeline can start and stop cleanly."""
    # Get path to demo script
    demo_path = Path(__file__).parent.parent / "demo_pipeline.py"
    assert demo_path.exists(), f"Demo script not found at {demo_path}"
    
    # Start demo with short duration
    process = subprocess.Popen(
        [sys.executable, str(demo_path), "--duration", "3"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    try:
        # Wait for process to complete
        stdout, stderr = process.communicate(timeout=10)
        
        # Check exit code
        assert process.returncode == 0, f"Demo exited with code {process.returncode}\nSTDERR: {stderr}"
        
        # Check for expected log messages
        assert "Starting demo pipeline" in stdout
        assert "Started BinanceWSCollector" in stdout
        assert "Pipeline shutdown complete" in stdout
        
        # Check no errors
        assert "ERROR" not in stdout
        assert "EXCEPTION" not in stdout
        
    except subprocess.TimeoutExpired:
        process.kill()
        pytest.fail("Demo script timed out")


@pytest.mark.integration
def test_demo_pipeline_help():
    """Test that demo pipeline shows help."""
    demo_path = Path(__file__).parent.parent / "demo_pipeline.py"
    
    result = subprocess.run(
        [sys.executable, str(demo_path), "--help"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "Demo pipeline for Binance WebSocket to TimescaleDB" in result.stdout
    assert "--database-url" in result.stdout
    assert "--duration" in result.stdout
    assert "--batch-size" in result.stdout


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="Integration test requires TEST_DATABASE_URL"
)
async def test_demo_pipeline_with_database():
    """Test demo pipeline with real database connection."""
    import asyncpg
    
    database_url = os.getenv("TEST_DATABASE_URL")
    demo_path = Path(__file__).parent.parent / "demo_pipeline.py"
    
    # Clear any existing test data
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute("DROP TABLE IF EXISTS trades")
    finally:
        await conn.close()
    
    # Run demo for 5 seconds
    process = subprocess.Popen(
        [
            sys.executable, 
            str(demo_path), 
            "--database-url", database_url,
            "--duration", "5",
            "--batch-size", "10",  # Small batch for testing
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    stdout, stderr = process.communicate()
    
    # Check successful completion
    assert process.returncode == 0, f"Demo failed: {stderr}"
    assert "Connected to TimescaleDB" in stdout
    assert "Pipeline statistics" in stdout
    
    # Verify data was written
    conn = await asyncpg.connect(database_url)
    try:
        count = await conn.fetchval("SELECT COUNT(*) FROM trades")
        assert count > 0, "No trades were written to database"
        
        # Check data integrity
        row = await conn.fetchrow("""
            SELECT symbol, price, quantity 
            FROM trades 
            LIMIT 1
        """)
        assert row["symbol"] == "BTCUSDT"
        assert float(row["price"]) > 0
        assert float(row["quantity"]) > 0
        
    finally:
        await conn.close()