"""Tests for target variable generation."""

import pytest
import pandas as pd
import numpy as np

from src.features.targets import TargetGenerator


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range('2025-01-01', periods=20, freq='D')
    
    # Simple price series: 100, 101, 102, ..., 119
    prices = [100 + i for i in range(20)]
    
    data = {
        'date': dates,
        'close': prices,
    }
    
    df = pd.DataFrame(data)
    df = df.set_index('date')
    
    return df


def test_generator_initialization():
    """Test target generator initialization."""
    generator = TargetGenerator()
    assert generator is not None


def test_generate_log_return_target(sample_price_data):
    """Test log return target generation."""
    generator = TargetGenerator()
    
    result = generator.generate_log_return_target(
        sample_price_data, 'close', horizon=1, target_name='y_1d'
    )
    
    # Check that target column was created
    assert 'y_1d' in result.columns
    
    # Calculate expected log return for first row
    # log(101 / 100) = log(1.01) ≈ 0.00995
    expected = np.log(101 / 100)
    assert abs(result['y_1d'].iloc[0] - expected) < 1e-10
    
    # Last row should be NaN (no future price)
    assert pd.isna(result['y_1d'].iloc[-1])


def test_generate_log_return_target_horizon3(sample_price_data):
    """Test log return target with horizon=3."""
    generator = TargetGenerator()
    
    result = generator.generate_log_return_target(
        sample_price_data, 'close', horizon=3
    )
    
    # Check that target column was created
    assert 'y_3d' in result.columns
    
    # Calculate expected log return for first row
    # log(103 / 100) = log(1.03)
    expected = np.log(103 / 100)
    assert abs(result['y_3d'].iloc[0] - expected) < 1e-10
    
    # Last 3 rows should be NaN
    assert pd.isna(result['y_3d'].iloc[-3:]).all()


def test_generate_average_log_return_target(sample_price_data):
    """Test average log return target generation."""
    generator = TargetGenerator()
    
    result = generator.generate_average_log_return_target(
        sample_price_data, 'close', horizon=3, target_name='y_3d_avg'
    )
    
    # Check that target column was created
    assert 'y_3d_avg' in result.columns
    
    # For first row, should be average of:
    # log(101/100), log(102/101), log(103/102)
    ret1 = np.log(101 / 100)
    ret2 = np.log(102 / 101)
    ret3 = np.log(103 / 102)
    expected = (ret1 + ret2 + ret3) / 3
    
    assert abs(result['y_3d_avg'].iloc[0] - expected) < 1e-10
    
    # Only the very last row should be NaN (can't calculate any future return)
    assert pd.isna(result['y_3d_avg'].iloc[-1])


def test_generate_direction_target(sample_price_data):
    """Test direction target generation."""
    generator = TargetGenerator()
    
    result = generator.generate_direction_target(
        sample_price_data, 'close', horizon=1, target_name='y_dir'
    )
    
    # Check that target column was created
    assert 'y_dir' in result.columns
    
    # Since prices are increasing, most directions should be 1 (up)
    # But last row has no future price, so it will be NaN in log return
    # which becomes 0 (down) in binary classification
    valid_directions = result['y_dir'].iloc[:-1]
    assert (valid_directions == 1).sum() >= len(valid_directions) - 1


def test_generate_direction_target_with_threshold():
    """Test direction target with custom threshold."""
    # Create data with small price movements
    df = pd.DataFrame({
        'close': [100, 100.1, 99.9, 100.2, 99.8]
    })
    
    generator = TargetGenerator()
    
    # With threshold=0.002 (0.2%), some movements won't qualify as "up"
    result = generator.generate_direction_target(
        df, 'close', horizon=1, target_name='y_dir', threshold=0.002
    )
    
    # Check that target was created
    assert 'y_dir' in result.columns
    
    # First movement: 100 -> 100.1, log return ≈ 0.001 < 0.002, so y_dir=0
    assert result['y_dir'].iloc[0] == 0


def test_generate_direction_target_downtrend():
    """Test direction target with downtrend."""
    # Create decreasing prices
    df = pd.DataFrame({
        'close': [100, 99, 98, 97, 96]
    })
    
    generator = TargetGenerator()
    
    result = generator.generate_direction_target(df, 'close', horizon=1)
    
    # All directions should be 0 (down)
    valid_directions = result['y_dir'].iloc[:-1].dropna()
    assert (valid_directions == 0).all()


def test_generate_all_targets(sample_price_data):
    """Test generation of all target variables."""
    generator = TargetGenerator()
    
    result = generator.generate_all_targets(sample_price_data, 'close')
    
    # Check that all target columns were created
    assert 'y_1d' in result.columns
    assert 'y_3d_avg' in result.columns
    assert 'y_7d_avg' in result.columns
    assert 'y_dir' in result.columns
    
    # Check that y_dir is binary
    valid_dir = result['y_dir'].dropna()
    assert set(valid_dir.unique()).issubset({0, 1})


def test_remove_future_leakage(sample_price_data):
    """Test removal of future leakage rows."""
    generator = TargetGenerator()
    
    # Generate targets first
    df_with_targets = generator.generate_all_targets(sample_price_data, 'close')
    
    # Remove future leakage
    result = generator.remove_future_leakage(df_with_targets, max_horizon=7)
    
    # Should remove last 7 rows
    assert len(result) == len(sample_price_data) - 7
    
    # y_7d_avg should not have NaN in the remaining rows
    assert not result['y_7d_avg'].isna().any()


def test_get_target_statistics():
    """Test target statistics calculation."""
    # Create data with known properties
    df = pd.DataFrame({
        'close': [100 + i * 0.5 for i in range(50)],
    })
    
    generator = TargetGenerator()
    df_with_targets = generator.generate_all_targets(df, 'close')
    
    stats = generator.get_target_statistics(df_with_targets)
    
    # Check that statistics were calculated for all targets
    assert 'y_1d' in stats
    assert 'y_3d_avg' in stats
    assert 'y_7d_avg' in stats
    assert 'y_dir' in stats
    
    # Check regression target statistics
    assert stats['y_1d']['type'] == 'regression'
    assert 'mean' in stats['y_1d']
    assert 'std' in stats['y_1d']
    assert 'positive_ratio' in stats['y_1d']
    
    # Check classification target statistics
    assert stats['y_dir']['type'] == 'classification'
    assert 'positive_class_ratio' in stats['y_dir']
    assert 'positive_count' in stats['y_dir']
    
    # Since prices are increasing, y_dir should be mostly 1
    assert stats['y_dir']['positive_class_ratio'] > 0.9


def test_missing_column_handling():
    """Test handling of missing columns."""
    generator = TargetGenerator()
    
    # Create data without expected column
    df = pd.DataFrame({
        'price': [100, 101, 102],
    })
    
    # Should not raise error, just log warning
    result = generator.generate_log_return_target(df, 'close')
    assert 'y_1d' not in result.columns


def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    generator = TargetGenerator()
    
    df = pd.DataFrame()
    result = generator.generate_all_targets(df)
    
    assert len(result) == 0


def test_short_dataframe():
    """Test handling of very short DataFrame."""
    generator = TargetGenerator()
    
    df = pd.DataFrame({
        'close': [100, 101, 102]
    })
    
    result = generator.generate_all_targets(df, 'close')
    
    # Should create targets, but most will be NaN
    assert 'y_1d' in result.columns
    assert 'y_7d_avg' in result.columns


def test_log_return_properties():
    """Test mathematical properties of log returns."""
    # Create data with known returns
    df = pd.DataFrame({
        'close': [100, 110, 121]  # 10% increase each period
    })
    
    generator = TargetGenerator()
    result = generator.generate_log_return_target(df, 'close', horizon=1)
    
    # Log return for 10% increase should be log(1.1) ≈ 0.0953
    expected_log_return = np.log(1.1)
    assert abs(result['y_1d'].iloc[0] - expected_log_return) < 1e-10
    assert abs(result['y_1d'].iloc[1] - expected_log_return) < 1e-10


def test_average_log_return_vs_single():
    """Test that average log return is different from single period."""
    df = pd.DataFrame({
        'close': [100, 101, 103, 106]  # Varying returns
    })
    
    generator = TargetGenerator()
    
    # Single 3-day return
    single = generator.generate_log_return_target(df, 'close', horizon=3)
    
    # Average 3-day return
    average = generator.generate_average_log_return_target(df, 'close', horizon=3)
    
    # They should be different (average smooths)
    assert single['y_3d'].iloc[0] != average['y_3d_avg'].iloc[0]


def test_direction_consistency_with_log_return():
    """Test that direction target is consistent with log return."""
    df = pd.DataFrame({
        'close': [100, 102, 98, 105, 95]
    })
    
    generator = TargetGenerator()
    
    df_with_targets = generator.generate_log_return_target(df, 'close', horizon=1)
    df_with_targets = generator.generate_direction_target(df_with_targets, 'close', horizon=1)
    
    # Where y_1d > 0, y_dir should be 1
    # Where y_1d < 0, y_dir should be 0
    for idx in range(len(df) - 1):
        if not pd.isna(df_with_targets['y_1d'].iloc[idx]):
            expected_dir = 1 if df_with_targets['y_1d'].iloc[idx] > 0 else 0
            assert df_with_targets['y_dir'].iloc[idx] == expected_dir


def test_target_count():
    """Test that correct number of targets are generated."""
    df = pd.DataFrame({
        'close': [100 + i for i in range(20)]
    })
    
    generator = TargetGenerator()
    result = generator.generate_all_targets(df, 'close')
    
    # Should add 4 target columns
    expected_new_columns = 4
    actual_new_columns = len(result.columns) - len(df.columns)
    
    assert actual_new_columns == expected_new_columns

