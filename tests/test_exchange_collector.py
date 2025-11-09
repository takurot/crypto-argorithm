"""Tests for exchange data collector."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.data.collectors.exchange_collector import ExchangeCollector
from src.utils.config import Config


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock configuration."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    data_sources = {
        "exchanges": {
            "min_required": 3,
            "outlier_threshold_pct": 3.0,
            "apis": {
                "binance": {
                    "url": "https://api.binance.com/api/v3/ticker/price",
                    "symbol_format": "{base}{quote}",
                    "rate_limit": 1200,
                },
                "coinbase": {
                    "url": "https://api.coinbase.com/v2/prices/{symbol}/spot",
                    "symbol_format": "{base}-{quote}",
                    "rate_limit": 10,
                },
                "kraken": {
                    "url": "https://api.kraken.com/0/public/Ticker",
                    "symbol_format": "X{base}{quote}",
                    "rate_limit": 15,
                },
                "bitstamp": {
                    "url": "https://www.bitstamp.net/api/v2/ticker/{symbol}",
                    "symbol_format": "{base}{quote}",
                    "rate_limit": 8000,
                },
            }
        }
    }
    
    with open(config_dir / "data_sources.yaml", "w") as f:
        import yaml
        yaml.dump(data_sources, f)
    
    return Config(config_dir=str(config_dir))


@pytest.fixture
def collector(mock_config, tmp_path):
    """Create an ExchangeCollector instance."""
    cache_dir = tmp_path / "cache"
    return ExchangeCollector(
        symbols=["BTC"],
        cache_dir=cache_dir,
        config=mock_config,
    )


def test_collector_initialization(collector):
    """Test collector initialization."""
    assert collector.name == "exchange"
    assert collector.symbols == ["BTC"]
    assert collector.min_required == 3
    assert collector.outlier_threshold == 0.03


def test_extract_price_binance(collector):
    """Test price extraction from Binance format."""
    data = {"price": "50000.50"}
    price = collector._extract_price("binance", data)
    assert price == 50000.50


def test_extract_price_coinbase(collector):
    """Test price extraction from Coinbase format."""
    data = {"data": {"amount": "50000.75"}}
    price = collector._extract_price("coinbase", data)
    assert price == 50000.75


def test_extract_price_kraken(collector):
    """Test price extraction from Kraken format."""
    data = {
        "result": {
            "XXBTZUSD": {
                "c": ["50000.25", "1.0"]
            }
        }
    }
    price = collector._extract_price("kraken", data)
    assert price == 50000.25


def test_extract_price_invalid(collector):
    """Test price extraction with invalid data."""
    data = {"invalid": "data"}
    price = collector._extract_price("binance", data)
    assert price is None


@patch('src.data.collectors.exchange_collector.ExchangeCollector.fetch_with_retry')
def test_fetch_single_exchange_success(mock_fetch, collector):
    """Test successful fetch from a single exchange."""
    mock_fetch.return_value = {"price": "50000.00"}
    
    result = collector._fetch_single_exchange("binance", "BTC", "USD")
    
    assert result is not None
    assert result["exchange"] == "binance"
    assert result["symbol"] == "BTC"
    assert result["price"] == 50000.00


@patch('src.data.collectors.exchange_collector.ExchangeCollector.fetch_with_retry')
def test_fetch_single_exchange_failure(mock_fetch, collector):
    """Test fetch failure from a single exchange."""
    mock_fetch.side_effect = Exception("Network error")
    
    result = collector._fetch_single_exchange("binance", "BTC", "USD")
    
    assert result is None


@patch('src.data.collectors.exchange_collector.ExchangeCollector._fetch_single_exchange')
def test_collect_symbol_success(mock_fetch, collector):
    """Test successful collection from multiple exchanges."""
    # Mock responses from 4 exchanges
    mock_fetch.side_effect = [
        {"exchange": "binance", "symbol": "BTC", "price": 50000.00, "timestamp": "2025-01-01T00:00:00"},
        {"exchange": "coinbase", "symbol": "BTC", "price": 50100.00, "timestamp": "2025-01-01T00:00:00"},
        {"exchange": "kraken", "symbol": "BTC", "price": 50050.00, "timestamp": "2025-01-01T00:00:00"},
        {"exchange": "bitstamp", "symbol": "BTC", "price": 50075.00, "timestamp": "2025-01-01T00:00:00"},
    ]
    
    result = collector.collect_symbol("BTC")
    
    assert result["status"] == "success"
    assert result["num_exchanges"] == 4
    assert result["prices"]["median"] > 0
    assert result["prices"]["mean"] > 0


@patch('src.data.collectors.exchange_collector.ExchangeCollector._fetch_single_exchange')
def test_collect_symbol_insufficient_data(mock_fetch, collector):
    """Test collection with insufficient exchanges."""
    # Mock only 2 responses (below min_required=3)
    mock_fetch.side_effect = [
        {"exchange": "binance", "symbol": "BTC", "price": 50000.00, "timestamp": "2025-01-01T00:00:00"},
        {"exchange": "coinbase", "symbol": "BTC", "price": 50100.00, "timestamp": "2025-01-01T00:00:00"},
        None,
        None,
    ]
    
    result = collector.collect_symbol("BTC")
    
    assert result["status"] == "insufficient_data"
    assert result["num_exchanges"] == 2


@patch('src.data.collectors.exchange_collector.ExchangeCollector._fetch_single_exchange')
def test_collect_symbol_with_outliers(mock_fetch, collector):
    """Test collection with outlier detection."""
    # One price is 10% higher (outlier)
    mock_fetch.side_effect = [
        {"exchange": "binance", "symbol": "BTC", "price": 50000.00, "timestamp": "2025-01-01T00:00:00"},
        {"exchange": "coinbase", "symbol": "BTC", "price": 50100.00, "timestamp": "2025-01-01T00:00:00"},
        {"exchange": "kraken", "symbol": "BTC", "price": 50050.00, "timestamp": "2025-01-01T00:00:00"},
        {"exchange": "bitstamp", "symbol": "BTC", "price": 55000.00, "timestamp": "2025-01-01T00:00:00"},  # Outlier
    ]
    
    result = collector.collect_symbol("BTC")
    
    assert result["status"] == "success"
    assert result["num_outliers"] == 1
    assert result["num_exchanges"] == 3  # Outlier removed


@patch('src.data.collectors.exchange_collector.ExchangeCollector.collect_symbol')
def test_collect_multiple_symbols(mock_collect_symbol, collector, tmp_path):
    """Test collection of multiple symbols."""
    mock_collect_symbol.return_value = {
        "symbol": "BTC",
        "status": "success",
        "prices": {"median": 50000.00},
    }
    
    collector.symbols = ["BTC", "ETH"]
    results = collector.collect(save=True)
    
    assert "BTC" in results
    assert "ETH" in results
    assert mock_collect_symbol.call_count == 2
    
    # Check that cache file was created
    cache_files = list(tmp_path.glob("cache/**/*.json"))
    assert len(cache_files) > 0

