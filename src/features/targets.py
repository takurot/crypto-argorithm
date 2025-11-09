"""Target variable generation for price prediction."""

from typing import Optional

import pandas as pd
import numpy as np

from src.utils.logging import get_logger


logger = get_logger(__name__)


class TargetGenerator:
    """Generate target variables for price prediction models."""

    def __init__(self):
        """Initialize target generator."""
        logger.info("target_generator_initialized")

    def generate_log_return_target(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
        horizon: int = 1,
        target_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate log return target for regression.
        
        Args:
            df: DataFrame with price data
            price_column: Name of price column
            horizon: Number of periods ahead (default: 1)
            target_name: Custom name for target (default: y_{horizon}d)
            
        Returns:
            DataFrame with target column added
        """
        result = df.copy()
        
        if price_column not in df.columns:
            logger.warning("price_column_not_found", column=price_column)
            return result
        
        target_name = target_name or f"y_{horizon}d"
        
        # Future price
        future_price = df[price_column].shift(-horizon)
        
        # Log return: log(P_t+h / P_t)
        result[target_name] = np.log(future_price / df[price_column])
        
        logger.info(
            "log_return_target_generated",
            target_name=target_name,
            horizon=horizon,
        )
        
        return result

    def generate_average_log_return_target(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
        horizon: int = 3,
        target_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate average log return target over multiple periods.
        
        Args:
            df: DataFrame with price data
            price_column: Name of price column
            horizon: Number of periods to average (default: 3)
            target_name: Custom name for target (default: y_{horizon}d_avg)
            
        Returns:
            DataFrame with target column added
        """
        result = df.copy()
        
        if price_column not in df.columns:
            logger.warning("price_column_not_found", column=price_column)
            return result
        
        target_name = target_name or f"y_{horizon}d_avg"
        
        # Calculate log returns for each future period
        log_returns = []
        for h in range(1, horizon + 1):
            future_price = df[price_column].shift(-h)
            current_price = df[price_column].shift(-h + 1) if h > 1 else df[price_column]
            log_ret = np.log(future_price / current_price)
            log_returns.append(log_ret)
        
        # Average of log returns
        result[target_name] = pd.concat(log_returns, axis=1).mean(axis=1)
        
        logger.info(
            "average_log_return_target_generated",
            target_name=target_name,
            horizon=horizon,
        )
        
        return result

    def generate_direction_target(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
        horizon: int = 1,
        target_name: Optional[str] = None,
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        """Generate binary direction target for classification.
        
        Args:
            df: DataFrame with price data
            price_column: Name of price column
            horizon: Number of periods ahead (default: 1)
            target_name: Custom name for target (default: y_dir)
            threshold: Threshold for direction (default: 0.0, meaning any positive return)
            
        Returns:
            DataFrame with target column added
        """
        result = df.copy()
        
        if price_column not in df.columns:
            logger.warning("price_column_not_found", column=price_column)
            return result
        
        target_name = target_name or "y_dir"
        
        # Future price
        future_price = df[price_column].shift(-horizon)
        
        # Log return
        log_return = np.log(future_price / df[price_column])
        
        # Binary direction (1 if up, 0 if down)
        result[target_name] = (log_return > threshold).astype(int)
        
        logger.info(
            "direction_target_generated",
            target_name=target_name,
            horizon=horizon,
            threshold=threshold,
        )
        
        return result

    def generate_all_targets(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
    ) -> pd.DataFrame:
        """Generate all target variables as per spec2.md.
        
        Args:
            df: DataFrame with price data
            price_column: Name of price column
            
        Returns:
            DataFrame with all target columns added
        """
        result = df.copy()
        
        # y_1d: 1-day log return (primary regression target)
        result = self.generate_log_return_target(
            result, price_column, horizon=1, target_name="y_1d"
        )
        
        # y_3d_avg: 3-day average log return
        result = self.generate_average_log_return_target(
            result, price_column, horizon=3, target_name="y_3d_avg"
        )
        
        # y_7d_avg: 7-day average log return
        result = self.generate_average_log_return_target(
            result, price_column, horizon=7, target_name="y_7d_avg"
        )
        
        # y_dir: 1-day direction (primary classification target)
        result = self.generate_direction_target(
            result, price_column, horizon=1, target_name="y_dir"
        )
        
        # Count targets
        targets_added = 4
        
        logger.info(
            "all_targets_generated",
            targets_added=targets_added,
            target_names=["y_1d", "y_3d_avg", "y_7d_avg", "y_dir"],
        )
        
        return result

    def remove_future_leakage(
        self,
        df: pd.DataFrame,
        max_horizon: int = 7,
    ) -> pd.DataFrame:
        """Remove rows with future leakage (last max_horizon rows).
        
        Args:
            df: DataFrame with target variables
            max_horizon: Maximum prediction horizon used
            
        Returns:
            DataFrame with future leakage removed
        """
        # Remove last max_horizon rows where targets are NaN
        result = df.iloc[:-max_horizon].copy()
        
        logger.info(
            "future_leakage_removed",
            original_rows=len(df),
            cleaned_rows=len(result),
            removed_rows=max_horizon,
        )
        
        return result

    def get_target_statistics(
        self,
        df: pd.DataFrame,
        target_columns: Optional[list] = None,
    ) -> dict:
        """Calculate statistics for target variables.
        
        Args:
            df: DataFrame with target variables
            target_columns: List of target column names (default: auto-detect)
            
        Returns:
            Dictionary with statistics for each target
        """
        if target_columns is None:
            # Auto-detect target columns (start with 'y_')
            target_columns = [col for col in df.columns if col.startswith('y_')]
        
        stats = {}
        
        for col in target_columns:
            if col not in df.columns:
                continue
            
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            if col == 'y_dir':
                # Binary classification target
                stats[col] = {
                    'type': 'classification',
                    'count': len(col_data),
                    'positive_class_ratio': col_data.mean(),
                    'positive_count': col_data.sum(),
                    'negative_count': (1 - col_data).sum(),
                }
            else:
                # Regression target
                stats[col] = {
                    'type': 'regression',
                    'count': len(col_data),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'median': col_data.median(),
                    'positive_ratio': (col_data > 0).mean(),
                }
        
        logger.info("target_statistics_calculated", targets=len(stats))
        
        return stats

