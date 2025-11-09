"""Integration test for feature engineering pipeline."""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.lags import LagFeatureGenerator
from src.features.technical import TechnicalIndicatorGenerator
from src.features.targets import TargetGenerator
from src.utils.logging import setup_logging, get_logger

# Setup
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def create_sample_data(n_days: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing.
    
    Args:
        n_days: Number of days of data
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)
    
    dates = pd.date_range('2025-01-01', periods=n_days, freq='D')
    
    # Create realistic price data with trend and volatility
    base_price = 100
    trend = np.linspace(0, 20, n_days)
    volatility = np.random.normal(0, 2, n_days)
    close = base_price + trend + volatility
    
    data = {
        'date': dates,
        'open': close + np.random.uniform(-1, 1, n_days),
        'high': close + np.random.uniform(1, 3, n_days),
        'low': close - np.random.uniform(1, 3, n_days),
        'close': close,
        'volume': np.random.uniform(1000, 2000, n_days),
    }
    
    df = pd.DataFrame(data)
    df = df.set_index('date')
    
    return df


def test_lag_features():
    """Test lag feature generation."""
    logger.info("=== Testing Lag Feature Generation ===")
    
    df = create_sample_data(100)
    
    generator = LagFeatureGenerator(
        price_lags=[1, 2, 3, 5, 7],
        volume_lags=[1, 2, 3, 5, 7],
    )
    
    result = generator.generate_all_lag_features(df)
    
    logger.info(
        "lag_features_result",
        original_columns=len(df.columns),
        new_columns=len(result.columns),
        features_added=len(result.columns) - len(df.columns),
    )
    
    # Verify some features
    assert 'close_lag1' in result.columns
    assert 'volume_lag1' in result.columns
    assert 'log_return' in result.columns
    assert 'close_change1' in result.columns
    
    logger.info("lag_features_test_passed")
    
    return result


def test_technical_indicators():
    """Test technical indicator generation."""
    logger.info("=== Testing Technical Indicator Generation ===")
    
    df = create_sample_data(100)
    
    generator = TechnicalIndicatorGenerator(
        sma_windows=[7, 21, 50],
        ema_windows=[9, 26, 50],
    )
    
    result = generator.generate_all_technical_indicators(df)
    
    logger.info(
        "technical_indicators_result",
        original_columns=len(df.columns),
        new_columns=len(result.columns),
        features_added=len(result.columns) - len(df.columns),
    )
    
    # Verify some indicators
    assert 'sma7' in result.columns
    assert 'ema9' in result.columns
    assert 'rsi' in result.columns
    assert 'macd' in result.columns
    assert 'bb_middle' in result.columns
    assert 'atr' in result.columns
    
    logger.info("technical_indicators_test_passed")
    
    return result


def test_target_generation():
    """Test target variable generation."""
    logger.info("=== Testing Target Variable Generation ===")
    
    df = create_sample_data(100)
    
    generator = TargetGenerator()
    
    result = generator.generate_all_targets(df)
    
    logger.info(
        "targets_result",
        original_columns=len(df.columns),
        new_columns=len(result.columns),
        features_added=len(result.columns) - len(df.columns),
    )
    
    # Verify targets
    assert 'y_1d' in result.columns
    assert 'y_3d_avg' in result.columns
    assert 'y_7d_avg' in result.columns
    assert 'y_dir' in result.columns
    
    # Get statistics
    stats = generator.get_target_statistics(result)
    
    for target_name, target_stats in stats.items():
        logger.info(
            "target_statistics",
            target=target_name,
            type=target_stats['type'],
            count=target_stats['count'],
            stats={k: v for k, v in target_stats.items() if k not in ['type', 'count']},
        )
    
    logger.info("targets_test_passed")
    
    return result, stats


def test_full_pipeline():
    """Test full feature engineering pipeline."""
    logger.info("=== Testing Full Feature Engineering Pipeline ===")
    
    df = create_sample_data(100)
    
    logger.info("original_data", rows=len(df), columns=len(df.columns))
    
    # Step 1: Lag features
    lag_gen = LagFeatureGenerator()
    df_with_lags = lag_gen.generate_all_lag_features(df)
    logger.info("after_lags", rows=len(df_with_lags), columns=len(df_with_lags.columns))
    
    # Step 2: Technical indicators
    tech_gen = TechnicalIndicatorGenerator()
    df_with_tech = tech_gen.generate_all_technical_indicators(df_with_lags)
    logger.info("after_technical", rows=len(df_with_tech), columns=len(df_with_tech.columns))
    
    # Step 3: Target variables
    target_gen = TargetGenerator()
    df_with_targets = target_gen.generate_all_targets(df_with_tech)
    logger.info("after_targets", rows=len(df_with_targets), columns=len(df_with_targets.columns))
    
    # Step 4: Remove future leakage
    df_clean = target_gen.remove_future_leakage(df_with_targets, max_horizon=7)
    logger.info("after_cleanup", rows=len(df_clean), columns=len(df_clean.columns))
    
    # Count NaN values
    nan_counts = df_clean.isna().sum()
    columns_with_nans = (nan_counts > 0).sum()
    
    logger.info(
        "nan_analysis",
        columns_with_nans=columns_with_nans,
        total_columns=len(df_clean.columns),
        max_nans=nan_counts.max(),
    )
    
    # Summary
    logger.info(
        "pipeline_summary",
        original_features=len(df.columns),
        final_features=len(df_clean.columns),
        features_added=len(df_clean.columns) - len(df.columns),
        original_rows=len(df),
        final_rows=len(df_clean),
        rows_removed=len(df) - len(df_clean),
    )
    
    logger.info("full_pipeline_test_passed")
    
    return df_clean


def test_feature_quality():
    """Test quality of generated features."""
    logger.info("=== Testing Feature Quality ===")
    
    df = create_sample_data(100)
    
    # Generate all features
    lag_gen = LagFeatureGenerator()
    tech_gen = TechnicalIndicatorGenerator()
    target_gen = TargetGenerator()
    
    df_features = lag_gen.generate_all_lag_features(df)
    df_features = tech_gen.generate_all_technical_indicators(df_features)
    df_features = target_gen.generate_all_targets(df_features)
    df_features = target_gen.remove_future_leakage(df_features, max_horizon=7)
    
    # Check for infinite values
    inf_counts = np.isinf(df_features.select_dtypes(include=[np.number])).sum()
    total_infs = inf_counts.sum()
    
    logger.info("infinite_values_check", total_infinites=total_infs)
    assert total_infs == 0, f"Found {total_infs} infinite values"
    
    # Check RSI range
    if 'rsi' in df_features.columns:
        rsi_valid = df_features['rsi'].dropna()
        rsi_in_range = ((rsi_valid >= 0) & (rsi_valid <= 100)).all()
        logger.info(
            "rsi_range_check",
            in_range=rsi_in_range,
            min=rsi_valid.min(),
            max=rsi_valid.max(),
        )
        assert rsi_in_range, "RSI values outside [0, 100] range"
    
    # Check y_dir is binary
    if 'y_dir' in df_features.columns:
        y_dir_valid = df_features['y_dir'].dropna()
        unique_values = set(y_dir_valid.unique())
        is_binary = unique_values.issubset({0, 1})
        logger.info(
            "y_dir_binary_check",
            is_binary=is_binary,
            unique_values=sorted(unique_values),
        )
        assert is_binary, f"y_dir not binary, found values: {unique_values}"
    
    # Check correlation between features (should have variety)
    numeric_features = df_features.select_dtypes(include=[np.number]).dropna()
    if len(numeric_features.columns) > 1:
        corr_matrix = numeric_features.corr().abs()
        # Exclude diagonal
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr = corr_matrix.max().max()
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        logger.info(
            "correlation_analysis",
            max_correlation=float(max_corr),
            avg_correlation=float(avg_corr),
        )
    
    logger.info("feature_quality_test_passed")
    
    return True


def main():
    """Run all integration tests."""
    logger.info("=== Starting Phase 2 Feature Engineering Integration Tests ===")
    
    try:
        # Individual component tests
        test_lag_features()
        test_technical_indicators()
        test_target_generation()
        
        # Full pipeline test
        df_result = test_full_pipeline()
        
        # Quality checks
        test_feature_quality()
        
        # Final summary
        logger.info("=" * 60)
        logger.info("=== All Integration Tests Passed ✅ ===")
        logger.info("=" * 60)
        logger.info(
            "summary",
            total_tests=5,
            passed=5,
            failed=0,
        )
        
        return True
        
    except Exception as e:
        logger.error("integration_test_failed", error=str(e), exc_info=True)
        logger.error("=== Integration Tests Failed ❌ ===")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

