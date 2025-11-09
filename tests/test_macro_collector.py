"""Tests for macro economic data collector."""

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from src.data.collectors.macro_collector import MacroCollector
from src.utils.config import Config


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock configuration."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    data_sources = {
        "macro": {
            "fred": {
                "base_url": "https://api.stlouisfed.org/fred",
                "api_key_env": "FRED_API_KEY",
                "series": {
                    "us10y": "DGS10",
                    "m2_supply": "WM2NS",
                },
                "max_missing_days_daily": 7,
                "max_missing_days_monthly": 30,
            },
            "yahoo_finance": {
                "symbols": {
                    "nasdaq": "^IXIC",
                    "gold": "GC=F",
                }
            }
        }
    }
    
    with open(config_dir / "data_sources.yaml", "w") as f:
        import yaml
        yaml.dump(data_sources, f)
    
    return Config(config_dir=str(config_dir))


@pytest.fixture
def collector(mock_config, tmp_path, monkeypatch):
    """Create a MacroCollector instance with mock API key."""
    monkeypatch.setenv("FRED_API_KEY", "test_api_key")
    
    cache_dir = tmp_path / "cache"
    return MacroCollector(
        cache_dir=cache_dir,
        config=mock_config,
    )


def test_collector_initialization(collector):
    """Test collector initialization."""
    assert collector.name == "macro"
    assert collector.fred_api_key == "test_api_key"


def test_collector_initialization_without_api_key(mock_config, tmp_path, monkeypatch):
    """Test collector initialization without API key."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    
    collector = MacroCollector(cache_dir=tmp_path / "cache", config=mock_config)
    assert collector.fred_api_key is None


@patch('src.data.collectors.macro_collector.MacroCollector.fetch_with_retry')
def test_fetch_fred_series_success(mock_fetch, collector):
    """Test successful FRED series fetch."""
    mock_fetch.return_value = {
        "observations": [
            {"date": "2025-01-01", "value": "4.5"},
            {"date": "2025-01-02", "value": "4.6"},
            {"date": "2025-01-03", "value": "."},  # Missing value
        ]
    }
    
    start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 1, 3, tzinfo=timezone.utc)
    
    df = collector._fetch_fred_series("DGS10", start_date, end_date)
    
    assert df is not None
    assert len(df) == 2  # Missing value excluded
    assert "DGS10" in df.columns


@patch('src.data.collectors.macro_collector.MacroCollector.fetch_with_retry')
def test_fetch_fred_series_empty(mock_fetch, collector):
    """Test FRED series fetch with empty response."""
    mock_fetch.return_value = {"observations": []}
    
    start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 1, 3, tzinfo=timezone.utc)
    
    df = collector._fetch_fred_series("DGS10", start_date, end_date)
    
    assert df is None


@patch('src.data.collectors.macro_collector.MacroCollector.fetch_with_retry')
def test_fetch_fred_series_error(mock_fetch, collector):
    """Test FRED series fetch with error."""
    mock_fetch.side_effect = Exception("API error")
    
    start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 1, 3, tzinfo=timezone.utc)
    
    df = collector._fetch_fred_series("DGS10", start_date, end_date)
    
    assert df is None


def test_fetch_fred_series_without_api_key(mock_config, tmp_path, monkeypatch):
    """Test FRED fetch without API key."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    
    collector = MacroCollector(cache_dir=tmp_path / "cache", config=mock_config)
    
    start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 1, 3, tzinfo=timezone.utc)
    
    df = collector._fetch_fred_series("DGS10", start_date, end_date)
    
    assert df is None


@patch('yfinance.Ticker')
def test_fetch_yahoo_symbol_success(mock_ticker_class, collector):
    """Test successful Yahoo Finance fetch."""
    # Create mock ticker instance
    mock_ticker = Mock()
    mock_ticker_class.return_value = mock_ticker
    
    # Create mock history DataFrame
    dates = pd.date_range('2025-01-01', periods=3, freq='D')
    mock_df = pd.DataFrame({
        'Close': [15000.0, 15100.0, 15200.0]
    }, index=dates)
    mock_ticker.history.return_value = mock_df
    
    start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 1, 3, tzinfo=timezone.utc)
    
    df = collector._fetch_yahoo_symbol("^IXIC", start_date, end_date)
    
    assert df is not None
    assert len(df) == 3
    assert "^IXIC" in df.columns


@patch('yfinance.Ticker')
def test_fetch_yahoo_symbol_empty(mock_ticker_class, collector):
    """Test Yahoo Finance fetch with empty response."""
    mock_ticker = Mock()
    mock_ticker_class.return_value = mock_ticker
    mock_ticker.history.return_value = pd.DataFrame()
    
    start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 1, 3, tzinfo=timezone.utc)
    
    df = collector._fetch_yahoo_symbol("^IXIC", start_date, end_date)
    
    assert df is None


@patch('src.data.collectors.macro_collector.MacroCollector._fetch_fred_series')
@patch('src.data.collectors.macro_collector.MacroCollector._fetch_yahoo_symbol')
def test_collect_all_data(mock_yahoo, mock_fred, collector, tmp_path):
    """Test collecting all macro data."""
    # Mock FRED responses
    dates = pd.date_range('2025-01-01', periods=3, freq='D')
    mock_fred.return_value = pd.DataFrame({'DGS10': [4.5, 4.6, 4.7]}, index=dates)
    
    # Mock Yahoo responses
    mock_yahoo.return_value = pd.DataFrame({'^IXIC': [15000, 15100, 15200]}, index=dates)
    
    start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 1, 3, tzinfo=timezone.utc)
    
    results = collector.collect(start_date=start_date, end_date=end_date, save=True)
    
    assert "fred" in results
    assert "yahoo" in results
    assert "metadata" in results
    assert results["metadata"]["fred_series_count"] == 2
    assert results["metadata"]["yahoo_symbols_count"] == 2
    
    # Check that cache files were created
    cache_files = list(tmp_path.glob("cache/**/*.csv"))
    assert len(cache_files) > 0


@patch('src.data.collectors.macro_collector.MacroCollector._fetch_fred_series')
@patch('src.data.collectors.macro_collector.MacroCollector._fetch_yahoo_symbol')
def test_collect_with_defaults(mock_yahoo, mock_fred, collector):
    """Test collecting with default date range."""
    dates = pd.date_range('2025-01-01', periods=3, freq='D')
    mock_fred.return_value = pd.DataFrame({'DGS10': [4.5, 4.6, 4.7]}, index=dates)
    mock_yahoo.return_value = pd.DataFrame({'^IXIC': [15000, 15100, 15200]}, index=dates)
    
    results = collector.collect(save=False)
    
    assert results is not None
    assert "metadata" in results
    
    # Check that default date range is approximately 2 years
    start = datetime.fromisoformat(results["metadata"]["start_date"])
    end = datetime.fromisoformat(results["metadata"]["end_date"])
    delta = (end - start).days
    assert 700 < delta < 800  # Approximately 2 years

