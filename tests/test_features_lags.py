"""Tests for lag feature generation."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.lags import LagFeatureGenerator


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2025-01-01', periods=20, freq='D')
    
    data = {
        'date': dates,
        'open': np.random.uniform(100, 110, 20),
        'high': np.random.uniform(110, 120, 20),
        'low': np.random.uniform(90, 100, 20),
        'close': np.random.uniform(100, 110, 20),
        'volume': np.random.uniform(1000, 2000, 20),
    }
    
    df = pd.DataFrame(data)
    df = df.set_index('date')
    
    return df


def test_generator_initialization():
    """Test lag feature generator initialization."""
    generator = LagFeatureGenerator()
    
    assert generator.price_lags == [1, 2, 3, 5, 7]
    assert generator.volume_lags == [1, 2, 3, 5, 7]
    assert generator.cross_asset_lags == [1, 2, 3]
    assert generator.macro_lags == [1, 2, 3]


def test_generator_custom_lags():
    """Test generator with custom lag periods."""
    generator = LagFeatureGenerator(
        price_lags=[1, 5],
        volume_lags=[1, 3],
    )
    
    assert generator.price_lags == [1, 5]
    assert generator.volume_lags == [1, 3]


def test_generate_price_lags(sample_ohlcv_data):
    """Test price lag feature generation."""
    generator = LagFeatureGenerator(price_lags=[1, 2, 3])
    
    result = generator.generate_price_lags(sample_ohlcv_data)
    
    # Check that lag columns were created
    assert 'close_lag1' in result.columns
    assert 'close_lag2' in result.columns
    assert 'close_lag3' in result.columns
    assert 'open_lag1' in result.columns
    
    # Check lag values
    assert result['close_lag1'].iloc[1] == sample_ohlcv_data['close'].iloc[0]
    assert result['close_lag2'].iloc[2] == sample_ohlcv_data['close'].iloc[0]
    
    # Check NaN values
    assert pd.isna(result['close_lag1'].iloc[0])
    assert pd.isna(result['close_lag3'].iloc[0])


def test_generate_volume_lags(sample_ohlcv_data):
    """Test volume lag feature generation."""
    generator = LagFeatureGenerator(volume_lags=[1, 2, 3])
    
    result = generator.generate_volume_lags(sample_ohlcv_data)
    
    # Check that lag columns were created
    assert 'volume_lag1' in result.columns
    assert 'volume_lag2' in result.columns
    assert 'volume_lag3' in result.columns
    
    # Check lag values
    assert result['volume_lag1'].iloc[1] == sample_ohlcv_data['volume'].iloc[0]
    
    # Check NaN values
    assert pd.isna(result['volume_lag1'].iloc[0])


def test_generate_log_returns(sample_ohlcv_data):
    """Test log return feature generation."""
    generator = LagFeatureGenerator(price_lags=[1, 2, 3])
    
    result = generator.generate_log_returns(sample_ohlcv_data, 'close')
    
    # Check that log return column was created
    assert 'log_return' in result.columns
    assert 'log_return_lag2' in result.columns
    assert 'log_return_lag3' in result.columns
    
    # Check log return calculation
    expected_return = np.log(
        sample_ohlcv_data['close'].iloc[1] / sample_ohlcv_data['close'].iloc[0]
    )
    assert abs(result['log_return'].iloc[1] - expected_return) < 1e-10
    
    # Check first value is NaN
    assert pd.isna(result['log_return'].iloc[0])


def test_generate_price_changes(sample_ohlcv_data):
    """Test price change feature generation."""
    generator = LagFeatureGenerator()
    
    result = generator.generate_price_changes(sample_ohlcv_data, 'close', periods=[1, 3])
    
    # Check that change columns were created
    assert 'close_change1' in result.columns
    assert 'close_pct_change1' in result.columns
    assert 'close_change3' in result.columns
    assert 'close_pct_change3' in result.columns
    
    # Check absolute change calculation
    expected_change = sample_ohlcv_data['close'].iloc[1] - sample_ohlcv_data['close'].iloc[0]
    assert abs(result['close_change1'].iloc[1] - expected_change) < 1e-10
    
    # Check percentage change calculation
    expected_pct = (sample_ohlcv_data['close'].iloc[1] / sample_ohlcv_data['close'].iloc[0]) - 1
    assert abs(result['close_pct_change1'].iloc[1] - expected_pct) < 1e-10


def test_generate_volume_changes(sample_ohlcv_data):
    """Test volume change feature generation."""
    generator = LagFeatureGenerator()
    
    result = generator.generate_volume_changes(sample_ohlcv_data, 'volume', periods=[1, 3])
    
    # Check that change columns were created
    assert 'volume_change1' in result.columns
    assert 'volume_pct_change1' in result.columns
    assert 'volume_change3' in result.columns
    assert 'volume_pct_change3' in result.columns
    
    # Check absolute change calculation
    expected_change = sample_ohlcv_data['volume'].iloc[1] - sample_ohlcv_data['volume'].iloc[0]
    assert abs(result['volume_change1'].iloc[1] - expected_change) < 1e-10


def test_generate_all_lag_features(sample_ohlcv_data):
    """Test generation of all lag features."""
    generator = LagFeatureGenerator(
        price_lags=[1, 2],
        volume_lags=[1, 2],
    )
    
    result = generator.generate_all_lag_features(sample_ohlcv_data)
    
    # Check that multiple feature types were created
    assert 'close_lag1' in result.columns
    assert 'volume_lag1' in result.columns
    assert 'log_return' in result.columns
    assert 'close_change1' in result.columns
    assert 'volume_pct_change1' in result.columns
    
    # Check that features were added
    original_cols = len(sample_ohlcv_data.columns)
    new_cols = len(result.columns)
    assert new_cols > original_cols


def test_missing_column_handling():
    """Test handling of missing columns."""
    generator = LagFeatureGenerator()
    
    # Create data without expected columns
    df = pd.DataFrame({
        'price': [100, 101, 102],
        'vol': [1000, 1100, 1200],
    })
    
    # Should not raise error, just log warning
    result = generator.generate_price_lags(df, ['close'])
    assert len(result.columns) == len(df.columns)  # No new columns added
    
    result = generator.generate_volume_lags(df, 'volume')
    assert len(result.columns) == len(df.columns)  # No new columns added


def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    generator = LagFeatureGenerator()
    
    df = pd.DataFrame()
    result = generator.generate_all_lag_features(df)
    
    assert len(result) == 0


def test_single_row_dataframe():
    """Test handling of single row DataFrame."""
    generator = LagFeatureGenerator(price_lags=[1])
    
    df = pd.DataFrame({
        'close': [100],
        'volume': [1000],
    })
    
    result = generator.generate_price_lags(df)
    
    # Lag features should be NaN for single row
    assert pd.isna(result['close_lag1'].iloc[0])


def test_feature_count():
    """Test that correct number of features are generated."""
    generator = LagFeatureGenerator(
        price_lags=[1, 2, 3],  # 3 lags
        volume_lags=[1, 2, 3],  # 3 lags
    )
    
    df = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [100, 101, 102, 103, 104],
        'volume': [1000, 1100, 1200, 1300, 1400],
    })
    
    result = generator.generate_all_lag_features(df)
    
    # Price lags: 4 columns × 3 lags = 12
    # Volume lags: 1 column × 3 lags = 3
    # Log returns: 1 + 2 lagged = 3
    # Price changes: 2 periods × 2 types = 4
    # Volume changes: 2 periods × 2 types = 4
    # Total new features: 12 + 3 + 3 + 4 + 4 = 26
    expected_new_features = 26
    actual_new_features = len(result.columns) - len(df.columns)
    
    assert actual_new_features == expected_new_features

