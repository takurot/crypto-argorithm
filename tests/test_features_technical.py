"""Tests for technical indicator features."""

import pytest
import pandas as pd
import numpy as np

from src.features.technical import TechnicalIndicatorGenerator


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range('2025-01-01', periods=100, freq='D')
    
    # Create realistic price data with trend
    base_price = 100
    trend = np.linspace(0, 20, 100)
    noise = np.random.normal(0, 2, 100)
    close = base_price + trend + noise
    
    data = {
        'date': dates,
        'open': close + np.random.uniform(-1, 1, 100),
        'high': close + np.random.uniform(1, 3, 100),
        'low': close - np.random.uniform(1, 3, 100),
        'close': close,
        'volume': np.random.uniform(1000, 2000, 100),
    }
    
    df = pd.DataFrame(data)
    df = df.set_index('date')
    
    return df


def test_generator_initialization():
    """Test technical indicator generator initialization."""
    generator = TechnicalIndicatorGenerator()
    
    assert generator.sma_windows == [7, 21, 50]
    assert generator.ema_windows == [9, 26, 50]
    assert generator.rsi_window == 14
    assert generator.macd_windows == (12, 26, 9)


def test_generate_sma(sample_ohlcv_data):
    """Test SMA generation."""
    generator = TechnicalIndicatorGenerator(sma_windows=[7, 21])
    
    result = generator.generate_sma(sample_ohlcv_data, 'close')
    
    # Check that SMA columns were created
    assert 'sma7' in result.columns
    assert 'sma21' in result.columns
    assert 'price_sma7_dev' in result.columns
    assert 'price_sma21_dev' in result.columns
    
    # Check SMA calculation (manual verification for window 7)
    expected_sma7_at_10 = sample_ohlcv_data['close'].iloc[4:11].mean()
    assert abs(result['sma7'].iloc[10] - expected_sma7_at_10) < 1e-10
    
    # Check that first values are NaN
    assert pd.isna(result['sma7'].iloc[0])
    assert pd.isna(result['sma21'].iloc[0])


def test_generate_ema(sample_ohlcv_data):
    """Test EMA generation."""
    generator = TechnicalIndicatorGenerator(ema_windows=[9, 26])
    
    result = generator.generate_ema(sample_ohlcv_data, 'close')
    
    # Check that EMA columns were created
    assert 'ema9' in result.columns
    assert 'ema26' in result.columns
    assert 'price_ema9_dev' in result.columns
    
    # EMA should be smooth and follow the trend
    assert not pd.isna(result['ema9'].iloc[20])
    
    # EMA should be between min and max of close prices
    assert result['ema9'].min() >= sample_ohlcv_data['close'].min() - 5
    assert result['ema9'].max() <= sample_ohlcv_data['close'].max() + 5


def test_generate_rsi(sample_ohlcv_data):
    """Test RSI generation."""
    generator = TechnicalIndicatorGenerator(rsi_window=14)
    
    result = generator.generate_rsi(sample_ohlcv_data, 'close')
    
    # Check that RSI column was created
    assert 'rsi' in result.columns
    
    # RSI should be between 0 and 100
    valid_rsi = result['rsi'].dropna()
    assert (valid_rsi >= 0).all()
    assert (valid_rsi <= 100).all()
    
    # First values should be NaN
    assert pd.isna(result['rsi'].iloc[0])


def test_generate_macd(sample_ohlcv_data):
    """Test MACD generation."""
    generator = TechnicalIndicatorGenerator(macd_windows=(12, 26, 9))
    
    result = generator.generate_macd(sample_ohlcv_data, 'close')
    
    # Check that MACD columns were created
    assert 'macd' in result.columns
    assert 'macd_signal' in result.columns
    assert 'macd_hist' in result.columns
    
    # MACD histogram should be MACD - Signal
    valid_idx = result['macd_hist'].first_valid_index()
    if valid_idx is not None:
        expected_hist = result['macd'].loc[valid_idx] - result['macd_signal'].loc[valid_idx]
        assert abs(result['macd_hist'].loc[valid_idx] - expected_hist) < 1e-10


def test_generate_bollinger_bands(sample_ohlcv_data):
    """Test Bollinger Bands generation."""
    generator = TechnicalIndicatorGenerator(bollinger_window=20, bollinger_std=2)
    
    result = generator.generate_bollinger_bands(sample_ohlcv_data, 'close')
    
    # Check that Bollinger columns were created
    assert 'bb_middle' in result.columns
    assert 'bb_upper' in result.columns
    assert 'bb_lower' in result.columns
    assert 'bb_width' in result.columns
    assert 'bb_pct' in result.columns
    
    # Upper band should be above middle, middle above lower
    valid_data = result.dropna()
    assert (valid_data['bb_upper'] >= valid_data['bb_middle']).all()
    assert (valid_data['bb_middle'] >= valid_data['bb_lower']).all()
    
    # %B should generally be between 0 and 1
    bb_pct_valid = result['bb_pct'].dropna()
    # Allow some outliers
    within_range = ((bb_pct_valid >= -0.5) & (bb_pct_valid <= 1.5)).sum()
    assert within_range / len(bb_pct_valid) > 0.90


def test_generate_atr(sample_ohlcv_data):
    """Test ATR generation."""
    generator = TechnicalIndicatorGenerator(atr_window=14)
    
    result = generator.generate_atr(sample_ohlcv_data)
    
    # Check that ATR columns were created
    assert 'atr' in result.columns
    assert 'atr_pct' in result.columns
    
    # ATR should be positive
    valid_atr = result['atr'].dropna()
    assert (valid_atr > 0).all()
    
    # ATR percentage should be reasonable (< 50% typically)
    valid_atr_pct = result['atr_pct'].dropna()
    assert (valid_atr_pct < 0.5).all()


def test_generate_momentum(sample_ohlcv_data):
    """Test Momentum generation."""
    generator = TechnicalIndicatorGenerator()
    
    result = generator.generate_momentum(sample_ohlcv_data, 'close', window=10)
    
    # Check that momentum columns were created
    assert 'momentum' in result.columns
    assert 'momentum_pct' in result.columns
    
    # Momentum should equal price difference
    valid_idx = 15  # After window + some buffer
    expected_momentum = sample_ohlcv_data['close'].iloc[valid_idx] - sample_ohlcv_data['close'].iloc[valid_idx - 10]
    assert abs(result['momentum'].iloc[valid_idx] - expected_momentum) < 1e-10


def test_generate_all_technical_indicators(sample_ohlcv_data):
    """Test generation of all technical indicators."""
    generator = TechnicalIndicatorGenerator(
        sma_windows=[7, 21],
        ema_windows=[9, 26],
    )
    
    result = generator.generate_all_technical_indicators(sample_ohlcv_data, 'close')
    
    # Check that multiple indicator types were created
    assert 'sma7' in result.columns
    assert 'ema9' in result.columns
    assert 'rsi' in result.columns
    assert 'macd' in result.columns
    assert 'bb_middle' in result.columns
    assert 'atr' in result.columns
    assert 'momentum' in result.columns
    
    # Check that features were added
    original_cols = len(sample_ohlcv_data.columns)
    new_cols = len(result.columns)
    assert new_cols > original_cols


def test_missing_column_handling():
    """Test handling of missing columns."""
    generator = TechnicalIndicatorGenerator()
    
    # Create data without expected columns
    df = pd.DataFrame({
        'price': [100, 101, 102, 103, 104],
        'vol': [1000, 1100, 1200, 1300, 1400],
    })
    
    # Should not raise error, just log warning
    result = generator.generate_sma(df, 'close')
    assert len(result.columns) == len(df.columns)  # No new columns added


def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    generator = TechnicalIndicatorGenerator()
    
    df = pd.DataFrame()
    result = generator.generate_all_technical_indicators(df)
    
    assert len(result) == 0


def test_short_dataframe():
    """Test handling of DataFrame shorter than window."""
    generator = TechnicalIndicatorGenerator(sma_windows=[7])
    
    df = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],  # Only 5 rows
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'volume': [1000, 1100, 1200, 1300, 1400],
    })
    
    result = generator.generate_sma(df, 'close')
    
    # SMA should be all NaN since data is shorter than window
    assert pd.isna(result['sma7']).all()


def test_feature_count():
    """Test that correct number of features are generated."""
    generator = TechnicalIndicatorGenerator(
        sma_windows=[7, 21],  # 2 SMAs + 2 deviations = 4
        ema_windows=[9, 26],  # 2 EMAs + 2 deviations = 4
    )
    
    df = pd.DataFrame({
        'open': np.random.uniform(100, 110, 50),
        'high': np.random.uniform(110, 120, 50),
        'low': np.random.uniform(90, 100, 50),
        'close': np.random.uniform(100, 110, 50),
        'volume': np.random.uniform(1000, 2000, 50),
    })
    
    result = generator.generate_all_technical_indicators(df, 'close')
    
    # SMA: 2×2 = 4, EMA: 2×2 = 4
    # RSI: 1, MACD: 3, Bollinger: 5, ATR: 2, Momentum: 2
    # Total new features: 4 + 4 + 1 + 3 + 5 + 2 + 2 = 21
    expected_new_features = 21
    actual_new_features = len(result.columns) - len(df.columns)
    
    assert actual_new_features == expected_new_features


def test_rsi_extreme_values():
    """Test RSI with extreme price movements."""
    # Create data with strong uptrend
    df = pd.DataFrame({
        'close': [100 + i * 2 for i in range(30)]  # Strong uptrend
    })
    
    generator = TechnicalIndicatorGenerator(rsi_window=14)
    result = generator.generate_rsi(df, 'close')
    
    # RSI should be high (near 100) in strong uptrend
    rsi_values = result['rsi'].iloc[20:].dropna()
    assert (rsi_values > 50).all()  # Should be above 50 in uptrend


def test_bollinger_squeeze():
    """Test Bollinger Bands during low volatility."""
    # Create data with low volatility
    df = pd.DataFrame({
        'close': [100 + np.random.uniform(-0.1, 0.1) for _ in range(50)]
    })
    
    generator = TechnicalIndicatorGenerator(bollinger_window=20)
    result = generator.generate_bollinger_bands(df, 'close')
    
    # Bollinger width should be small during low volatility
    bb_width_valid = result['bb_width'].dropna()
    assert bb_width_valid.mean() < 0.1  # Should be narrow

