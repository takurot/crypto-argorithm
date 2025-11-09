"""Tests for feature engineering pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import feature_engineering


@pytest.fixture
def sample_ohlcv_csv(tmp_path):
    """Create sample OHLCV CSV file."""
    dates = pd.date_range('2025-01-01', periods=150, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(100, 110, 150),
        'high': np.random.uniform(110, 120, 150),
        'low': np.random.uniform(90, 100, 150),
        'close': np.random.uniform(100, 110, 150),
        'volume': np.random.uniform(1000, 2000, 150),
    })
    
    df = df.set_index('date')
    
    # Save to CSV
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path)
    
    return csv_path


@pytest.fixture
def sample_ohlcv_parquet(tmp_path):
    """Create sample OHLCV Parquet file."""
    dates = pd.date_range('2025-01-01', periods=150, freq='D')
    
    df = pd.DataFrame({
        'open': np.random.uniform(100, 110, 150),
        'high': np.random.uniform(110, 120, 150),
        'low': np.random.uniform(90, 100, 150),
        'close': np.random.uniform(100, 110, 150),
        'volume': np.random.uniform(1000, 2000, 150),
    }, index=dates)
    df.index.name = 'date'
    
    # Save to Parquet
    parquet_path = tmp_path / "test_data.parquet"
    df.to_parquet(parquet_path)
    
    return parquet_path


def test_load_data_csv(sample_ohlcv_csv):
    """Test loading data from CSV."""
    df = feature_engineering.load_data(str(sample_ohlcv_csv))
    
    assert len(df) == 150
    assert 'open' in df.columns
    assert 'close' in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)


def test_load_data_parquet(sample_ohlcv_parquet):
    """Test loading data from Parquet."""
    df = feature_engineering.load_data(str(sample_ohlcv_parquet))
    
    assert len(df) == 150
    assert 'open' in df.columns
    assert 'close' in df.columns


def test_load_data_nonexistent():
    """Test loading non-existent file."""
    with pytest.raises(FileNotFoundError):
        feature_engineering.load_data("nonexistent.csv")


def test_validate_data_valid():
    """Test validation of valid data."""
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [100, 101, 102],
        'volume': [1000, 1100, 1200],
    })
    
    # Too short for actual validation to pass
    result = feature_engineering.validate_data(df)
    assert result is False  # Less than 100 rows


def test_validate_data_sufficient():
    """Test validation with sufficient data."""
    df = pd.DataFrame({
        'open': np.random.uniform(100, 110, 120),
        'high': np.random.uniform(110, 120, 120),
        'low': np.random.uniform(90, 100, 120),
        'close': np.random.uniform(100, 110, 120),
        'volume': np.random.uniform(1000, 2000, 120),
    })
    
    result = feature_engineering.validate_data(df)
    assert result is True


def test_validate_data_missing_columns():
    """Test validation with missing columns."""
    df = pd.DataFrame({
        'open': [100, 101, 102],
        'close': [100, 101, 102],
        # Missing high, low, volume
    })
    
    result = feature_engineering.validate_data(df)
    assert result is False


def test_generate_features_all():
    """Test generating all features."""
    df = pd.DataFrame({
        'open': np.random.uniform(100, 110, 120),
        'high': np.random.uniform(110, 120, 120),
        'low': np.random.uniform(90, 100, 120),
        'close': np.random.uniform(100, 110, 120),
        'volume': np.random.uniform(1000, 2000, 120),
    })
    
    result = feature_engineering.generate_features(df)
    
    # Should have more columns than input
    assert len(result.columns) > len(df.columns)
    
    # Should have lag features
    assert 'close_lag1' in result.columns
    
    # Should have technical indicators
    assert 'sma7' in result.columns
    assert 'rsi' in result.columns
    
    # Should have targets
    assert 'y_1d' in result.columns
    assert 'y_dir' in result.columns


def test_generate_features_skip_lags():
    """Test generating features without lags."""
    df = pd.DataFrame({
        'open': np.random.uniform(100, 110, 120),
        'high': np.random.uniform(110, 120, 120),
        'low': np.random.uniform(90, 100, 120),
        'close': np.random.uniform(100, 110, 120),
        'volume': np.random.uniform(1000, 2000, 120),
    })
    
    result = feature_engineering.generate_features(df, skip_lags=True)
    
    # Should not have lag features
    assert 'close_lag1' not in result.columns
    
    # Should still have technical indicators
    assert 'sma7' in result.columns


def test_generate_features_skip_technical():
    """Test generating features without technical indicators."""
    df = pd.DataFrame({
        'open': np.random.uniform(100, 110, 120),
        'high': np.random.uniform(110, 120, 120),
        'low': np.random.uniform(90, 100, 120),
        'close': np.random.uniform(100, 110, 120),
        'volume': np.random.uniform(1000, 2000, 120),
    })
    
    result = feature_engineering.generate_features(df, skip_technical=True)
    
    # Should have lag features
    assert 'close_lag1' in result.columns
    
    # Should not have technical indicators
    assert 'sma7' not in result.columns


def test_generate_features_skip_targets():
    """Test generating features without targets."""
    df = pd.DataFrame({
        'open': np.random.uniform(100, 110, 120),
        'high': np.random.uniform(110, 120, 120),
        'low': np.random.uniform(90, 100, 120),
        'close': np.random.uniform(100, 110, 120),
        'volume': np.random.uniform(1000, 2000, 120),
    })
    
    result = feature_engineering.generate_features(df, skip_targets=True)
    
    # Should have features
    assert 'close_lag1' in result.columns
    assert 'sma7' in result.columns
    
    # Should not have targets
    assert 'y_1d' not in result.columns
    assert 'y_dir' not in result.columns


def test_calculate_metadata():
    """Test metadata calculation."""
    df = pd.DataFrame({
        'close': [100.0, 101.0, np.nan, 103.0],
        'volume': [1000, 1100, 1200, 1300],
    })
    
    metadata = feature_engineering.calculate_metadata(df)
    
    # Check structure
    assert 'shape' in metadata
    assert 'date_range' in metadata
    assert 'columns' in metadata
    
    # Check shape
    assert metadata['shape']['rows'] == 4
    assert metadata['shape']['columns'] == 2
    
    # Check column metadata
    assert 'close' in metadata['columns']
    assert metadata['columns']['close']['missing_count'] == 1
    assert metadata['columns']['close']['missing_rate'] == 0.25
    
    # Check numeric statistics
    assert 'mean' in metadata['columns']['close']
    assert 'std' in metadata['columns']['close']


def test_save_features_csv(tmp_path):
    """Test saving features to CSV."""
    df = pd.DataFrame({
        'close': [100, 101, 102],
        'volume': [1000, 1100, 1200],
    })
    
    output_path = feature_engineering.save_features(
        df, str(tmp_path), output_format="csv", symbol="BTC"
    )
    
    assert output_path.exists()
    assert output_path.suffix == '.csv'
    
    # Load and verify
    loaded = pd.read_csv(output_path, index_col=0)
    assert len(loaded) == 3


def test_save_features_parquet(tmp_path):
    """Test saving features to Parquet."""
    df = pd.DataFrame({
        'close': [100, 101, 102],
        'volume': [1000, 1100, 1200],
    })
    
    output_path = feature_engineering.save_features(
        df, str(tmp_path), output_format="parquet", symbol="BTC"
    )
    
    assert output_path.exists()
    assert output_path.suffix == '.parquet'
    
    # Load and verify
    loaded = pd.read_parquet(output_path)
    assert len(loaded) == 3


def test_save_metadata(tmp_path):
    """Test saving metadata."""
    metadata = {
        "shape": {"rows": 100, "columns": 10},
        "columns": {
            "close": {"dtype": "float64", "missing_count": 0}
        }
    }
    
    output_path = feature_engineering.save_metadata(
        metadata, str(tmp_path), symbol="BTC"
    )
    
    assert output_path.exists()
    assert output_path.suffix == '.json'
    
    # Load and verify
    with open(output_path, 'r') as f:
        loaded = json.load(f)
    
    assert loaded['shape']['rows'] == 100
    assert 'close' in loaded['columns']


def test_end_to_end_csv(sample_ohlcv_csv, tmp_path):
    """Test end-to-end pipeline with CSV input."""
    # Load data
    df = feature_engineering.load_data(str(sample_ohlcv_csv))
    
    # Validate
    assert feature_engineering.validate_data(df)
    
    # Generate features
    df_features = feature_engineering.generate_features(df)
    
    # Calculate metadata
    metadata = feature_engineering.calculate_metadata(df_features)
    
    # Save features
    features_path = feature_engineering.save_features(
        df_features, str(tmp_path), output_format="parquet"
    )
    
    # Save metadata
    metadata_path = feature_engineering.save_metadata(metadata, str(tmp_path))
    
    # Verify
    assert features_path.exists()
    assert metadata_path.exists()
    
    # Load and check
    loaded_features = pd.read_parquet(features_path)
    assert len(loaded_features) > 0
    assert len(loaded_features.columns) > 5  # Should have generated features


def test_end_to_end_parquet(sample_ohlcv_parquet, tmp_path):
    """Test end-to-end pipeline with Parquet input."""
    # Load data
    df = feature_engineering.load_data(str(sample_ohlcv_parquet))
    
    # Validate
    assert feature_engineering.validate_data(df)
    
    # Generate features
    df_features = feature_engineering.generate_features(df)
    
    # Calculate metadata
    metadata = feature_engineering.calculate_metadata(df_features)
    
    # Save
    features_path = feature_engineering.save_features(
        df_features, str(tmp_path), output_format="csv"
    )
    metadata_path = feature_engineering.save_metadata(metadata, str(tmp_path))
    
    # Verify
    assert features_path.exists()
    assert metadata_path.exists()

