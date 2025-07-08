"""Pytest configuration for storage tests."""

import asyncio
import os

import asyncpg
import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_database_url():
    """Get test database URL from environment or use default."""
    return os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://test:test@localhost:5433/crypto_quant_test"
    )


@pytest.fixture(scope="function")
async def clean_database(test_database_url):
    """Clean database before each test."""
    conn = await asyncpg.connect(test_database_url)
    try:
        # Drop and recreate trades table
        await conn.execute("DROP TABLE IF EXISTS trades CASCADE")
    finally:
        await conn.close()


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests with real database"
    )


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test requiring database"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    if not config.getoption("--integration"):
        skip_integration = pytest.mark.skip(
            reason="need --integration option to run"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
