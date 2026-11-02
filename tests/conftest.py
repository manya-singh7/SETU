"""Pytest configuration and database fixtures."""

import pytest
from setu.database import create_db_and_tables


@pytest.fixture(autouse=True)
def setup_test_db():
    """Ensure database tables are created before running tests."""
    create_db_and_tables()
