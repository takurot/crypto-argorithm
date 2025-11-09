"""Tests for configuration loader."""

import pytest
import tempfile
from pathlib import Path
import yaml

from src.utils.config import Config


@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        
        # Create a test config file
        test_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {
                    "username": "admin",
                    "password": "secret"
                }
            },
            "api": {
                "timeout": 30,
                "max_retries": 3
            }
        }
        
        with open(config_dir / "test.yaml", "w") as f:
            yaml.dump(test_config, f)
        
        yield config_dir


def test_config_load(temp_config_dir):
    """Test loading a config file."""
    config = Config(config_dir=str(temp_config_dir))
    loaded = config.load("test")
    
    assert "database" in loaded
    assert "api" in loaded
    assert loaded["database"]["host"] == "localhost"


def test_config_load_nonexistent():
    """Test loading a non-existent config file."""
    config = Config(config_dir="nonexistent")
    
    with pytest.raises(FileNotFoundError):
        config.load("missing")


def test_config_get_simple_key(temp_config_dir):
    """Test getting a simple config value."""
    config = Config(config_dir=str(temp_config_dir))
    
    timeout = config.get("test", "api.timeout")
    assert timeout == 30


def test_config_get_nested_key(temp_config_dir):
    """Test getting a nested config value."""
    config = Config(config_dir=str(temp_config_dir))
    
    username = config.get("test", "database.credentials.username")
    assert username == "admin"


def test_config_get_with_default(temp_config_dir):
    """Test getting a non-existent key with default value."""
    config = Config(config_dir=str(temp_config_dir))
    
    value = config.get("test", "nonexistent.key", default="default_value")
    assert value == "default_value"


def test_config_caching(temp_config_dir):
    """Test that configs are cached after first load."""
    config = Config(config_dir=str(temp_config_dir))
    
    # Load twice
    first = config.load("test")
    second = config.load("test")
    
    # Should be the same object (cached)
    assert first is second

