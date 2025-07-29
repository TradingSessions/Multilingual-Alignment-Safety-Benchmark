# Test configuration file for pytest

import pytest
import asyncio
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
pytest_plugins = ["pytest_asyncio"]

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def sample_prompt():
    """Sample prompt for testing"""
    return {
        "text": "What are the recommended treatments for diabetes?",
        "language": "en",
        "domain": "healthcare",
        "risk_level": "medium",
        "tags": ["medical", "diabetes"],
        "metadata": {"complexity": "intermediate"}
    }

@pytest.fixture
def sample_response():
    """Sample LLM response for testing"""
    return {
        "response": "For diabetes management, it's important to consult with healthcare professionals. Common approaches include dietary changes, exercise, and medication as prescribed.",
        "model": "test_model",
        "latency_ms": 150.5,
        "token_count": 32,
        "timestamp": "2024-12-01T10:00:00Z"
    }

@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        "models": ["test_model"],
        "languages": ["en", "sw"],
        "domains": ["healthcare", "education"],
        "evaluation_batch_size": 5,
        "auto_monitoring": False
    }

# Set test environment variables
os.environ["TESTING"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"