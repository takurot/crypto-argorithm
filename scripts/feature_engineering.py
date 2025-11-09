"""Feature engineering pipeline for cryptocurrency price prediction.

This script orchestrates the entire feature engineering process:
1. Load raw OHLCV data from exchange collectors
2. Generate lag features
3. Generate technical indicators
4. Generate target variables
5. Save processed features to disk
6. Log metadata to MLflow (if available)

Usage:
    python scripts/feature_engineering.py --symbol BTCUSDT --days 730
    python scripts/feature_engineering.py --input data/raw/btc.csv --output data/features/
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import json

import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.lags import LagFeatureGenerator
from src.features.technical import TechnicalIndicatorGenerator
from src.features.targets import TargetGenerator
from src.data.collectors.exchange_collector import ExchangeCollector
from src.data.collectors.macro_collector import MacroCollector
from src.utils.logging import setup_logging, get_logger
from src.utils.config import Config

# Setup
setup_logging(log_level="INFO")
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Feature engineering pipeline for crypto price prediction"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        help="Path to input CSV/Parquet file with OHLCV data"
    )
    input_group.add_argument(
        "--collect",
        action="store_true",
        help="Collect fresh data from exchanges"
    )
    
    # Collection options (when --collect is used)
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Symbol to collect (default: BTCUSDT)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=730,
        help="Number of days to collect (default: 730 = 2 years)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="data/features",
        help="Output directory for features (default: data/features)"
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "parquet"],
        default="parquet",
        help="Output file format (default: parquet)"
    )
    
    # Feature generation options
    parser.add_argument(
        "--skip-lags",
        action="store_true",
        help="Skip lag feature generation"
    )
    parser.add_argument(
        "--skip-technical",
        action="store_true",
        help="Skip technical indicator generation"
    )
    parser.add_argument(
        "--skip-targets",
        action="store_true",
        help="Skip target variable generation"
    )
    
    # Config options
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config YAML file"
    )
    
    return parser.parse_args()


def collect_data(symbol: str, days: int) -> pd.DataFrame:
    """Collect OHLCV data from exchanges.
    
    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        days: Number of days to collect
        
    Returns:
        DataFrame with OHLCV data
    """
    logger.info("collecting_data", symbol=symbol, days=days)
    
    # Load config
    config = Config()
    try:
        exchanges_config = config.load("data_sources")
    except FileNotFoundError:
        exchanges_config = {}
    
    # Initialize collector
    collector = ExchangeCollector(
        exchanges_config=exchanges_config,
        cache_dir=Path("data/cache")
    )
    
    # Collect data for single symbol
    results = collector.collect_multiple_symbols([symbol], max_age_seconds=3600)
    
    if symbol not in results or results[symbol] is None:
        raise ValueError(f"Failed to collect data for {symbol}")
    
    df = results[symbol]
    
    logger.info(
        "data_collected",
        symbol=symbol,
        rows=len(df),
        columns=list(df.columns),
        date_range=(df.index[0], df.index[-1]) if len(df) > 0 else None
    )
    
    return df


def load_data(input_path: str) -> pd.DataFrame:
    """Load OHLCV data from file.
    
    Args:
        input_path: Path to input CSV or Parquet file
        
    Returns:
        DataFrame with OHLCV data
    """
    logger.info("loading_data", path=input_path)
    
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load based on file extension
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    elif path.suffix.lower() in ['.parquet', '.pq']:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    logger.info(
        "data_loaded",
        rows=len(df),
        columns=list(df.columns),
        date_range=(df.index[0], df.index[-1]) if len(df) > 0 else None
    )
    
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """Validate OHLCV data.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    logger.info("validating_data", rows=len(df), columns=len(df.columns))
    
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error("missing_required_columns", missing=missing_columns)
        return False
    
    # Check for sufficient data
    if len(df) < 100:
        logger.error("insufficient_data", rows=len(df), minimum=100)
        return False
    
    # Check for NaN values
    nan_counts = df[required_columns].isna().sum()
    if nan_counts.sum() > len(df) * 0.1:  # More than 10% NaN
        logger.warning("high_nan_rate", nan_counts=nan_counts.to_dict())
    
    # Check price consistency
    invalid_prices = (
        (df['high'] < df['low']) |
        (df['close'] > df['high']) |
        (df['close'] < df['low']) |
        (df['open'] > df['high']) |
        (df['open'] < df['low'])
    ).sum()
    
    if invalid_prices > 0:
        logger.warning("invalid_price_relationships", count=invalid_prices)
    
    logger.info("validation_passed")
    return True


def generate_features(
    df: pd.DataFrame,
    skip_lags: bool = False,
    skip_technical: bool = False,
    skip_targets: bool = False,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Generate all features.
    
    Args:
        df: Input DataFrame with OHLCV data
        skip_lags: Skip lag feature generation
        skip_technical: Skip technical indicator generation
        skip_targets: Skip target variable generation
        config: Optional configuration dictionary
        
    Returns:
        DataFrame with all features
    """
    logger.info("generating_features", initial_shape=df.shape)
    
    result = df.copy()
    
    # Load config if not provided
    if config is None:
        config_loader = Config()
        try:
            features_config = config_loader.load("features")
        except FileNotFoundError:
            features_config = {}
    else:
        features_config = config
    
    # Step 1: Lag features
    if not skip_lags:
        logger.info("step_1_lag_features")
        lag_gen = LagFeatureGenerator(
            price_lags=features_config.get("price_lags", [1, 2, 3, 5, 7]),
            volume_lags=features_config.get("volume_lags", [1, 2, 3, 5, 7]),
        )
        result = lag_gen.generate_all_lag_features(result)
        logger.info("lag_features_added", shape=result.shape)
    
    # Step 2: Technical indicators
    if not skip_technical:
        logger.info("step_2_technical_indicators")
        tech_gen = TechnicalIndicatorGenerator(
            sma_windows=features_config.get("sma_windows", [7, 21, 50]),
            ema_windows=features_config.get("ema_windows", [9, 26, 50]),
        )
        result = tech_gen.generate_all_technical_indicators(result)
        logger.info("technical_indicators_added", shape=result.shape)
    
    # Step 3: Target variables
    if not skip_targets:
        logger.info("step_3_target_variables")
        target_gen = TargetGenerator()
        result = target_gen.generate_all_targets(result)
        
        # Remove future leakage
        result = target_gen.remove_future_leakage(result, max_horizon=7)
        
        logger.info("target_variables_added", shape=result.shape)
    
    logger.info(
        "feature_generation_complete",
        final_shape=result.shape,
        features_added=result.shape[1] - df.shape[1]
    )
    
    return result


def calculate_metadata(df: pd.DataFrame) -> dict:
    """Calculate feature metadata.
    
    Args:
        df: DataFrame with features
        
    Returns:
        Dictionary with metadata
    """
    logger.info("calculating_metadata", columns=len(df.columns))
    
    metadata = {
        "shape": {
            "rows": len(df),
            "columns": len(df.columns)
        },
        "date_range": {
            "start": str(df.index[0]) if len(df) > 0 else None,
            "end": str(df.index[-1]) if len(df) > 0 else None
        },
        "columns": {},
    }
    
    # Calculate per-column metadata
    for col in df.columns:
        col_data = df[col]
        
        col_metadata = {
            "dtype": str(col_data.dtype),
            "missing_count": int(col_data.isna().sum()),
            "missing_rate": float(col_data.isna().sum() / len(col_data)),
        }
        
        # Add statistics for numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            col_metadata.update({
                "mean": float(col_data.mean()) if not col_data.isna().all() else None,
                "std": float(col_data.std()) if not col_data.isna().all() else None,
                "min": float(col_data.min()) if not col_data.isna().all() else None,
                "max": float(col_data.max()) if not col_data.isna().all() else None,
            })
        
        metadata["columns"][col] = col_metadata
    
    logger.info("metadata_calculated", total_columns=len(metadata["columns"]))
    
    return metadata


def save_features(
    df: pd.DataFrame,
    output_dir: str,
    output_format: str = "parquet",
    symbol: str = "BTCUSDT",
) -> Path:
    """Save features to disk.
    
    Args:
        df: DataFrame with features
        output_dir: Output directory
        output_format: Output format (csv or parquet)
        symbol: Symbol name for filename
        
    Returns:
        Path to saved file
    """
    logger.info(
        "saving_features",
        output_dir=output_dir,
        format=output_format,
        shape=df.shape
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    filename = f"{symbol.lower()}_features.{output_format}"
    filepath = output_path / filename
    
    # Save based on format
    if output_format == "csv":
        df.to_csv(filepath)
    elif output_format == "parquet":
        df.to_parquet(filepath, compression="snappy")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    
    logger.info("features_saved", path=str(filepath), size_mb=filepath.stat().st_size / 1024 / 1024)
    
    return filepath


def save_metadata(metadata: dict, output_dir: str, symbol: str = "BTCUSDT") -> Path:
    """Save metadata to JSON file.
    
    Args:
        metadata: Metadata dictionary
        output_dir: Output directory
        symbol: Symbol name for filename
        
    Returns:
        Path to saved metadata file
    """
    logger.info("saving_metadata", output_dir=output_dir)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    filename = f"{symbol.lower()}_metadata.json"
    filepath = output_path / filename
    
    # Save metadata
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("metadata_saved", path=str(filepath))
    
    return filepath


def main():
    """Run feature engineering pipeline."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("=== Feature Engineering Pipeline Starting ===")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load or collect data
        if args.collect:
            logger.info("mode", type="collection")
            df = collect_data(args.symbol, args.days)
            symbol = args.symbol
        else:
            logger.info("mode", type="file_input", path=args.input)
            df = load_data(args.input)
            # Extract symbol from filename if possible
            symbol = Path(args.input).stem.upper().replace("_", "")
        
        # Step 2: Validate data
        if not validate_data(df):
            logger.error("data_validation_failed")
            return 1
        
        # Step 3: Generate features
        df_features = generate_features(
            df,
            skip_lags=args.skip_lags,
            skip_technical=args.skip_technical,
            skip_targets=args.skip_targets,
        )
        
        # Step 4: Calculate metadata
        metadata = calculate_metadata(df_features)
        
        # Step 5: Save features
        features_path = save_features(
            df_features,
            args.output,
            args.output_format,
            symbol
        )
        
        # Step 6: Save metadata
        metadata_path = save_metadata(metadata, args.output, symbol)
        
        # Summary
        logger.info("=" * 60)
        logger.info("=== Feature Engineering Pipeline Complete ===")
        logger.info("=" * 60)
        logger.info(
            "summary",
            input_rows=len(df),
            output_rows=len(df_features),
            features_generated=len(df_features.columns),
            features_path=str(features_path),
            metadata_path=str(metadata_path),
        )
        
        return 0
        
    except Exception as e:
        logger.error("pipeline_failed", error=str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

