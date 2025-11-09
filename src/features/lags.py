"""Lag features generation for time series data."""

from typing import List, Optional

import pandas as pd
import numpy as np

from src.utils.logging import get_logger


logger = get_logger(__name__)


class LagFeatureGenerator:
    """Generate lag features for time series data."""

    def __init__(
        self,
        price_lags: Optional[List[int]] = None,
        volume_lags: Optional[List[int]] = None,
        cross_asset_lags: Optional[List[int]] = None,
        macro_lags: Optional[List[int]] = None,
    ):
        """Initialize lag feature generator.
        
        Args:
            price_lags: List of lag periods for price (default: [1,2,3,5,7])
            volume_lags: List of lag periods for volume (default: [1,2,3,5,7])
            cross_asset_lags: List of lag periods for cross-asset (default: [1,2,3])
            macro_lags: List of lag periods for macro indicators (default: [1,2,3])
        """
        self.price_lags = price_lags or [1, 2, 3, 5, 7]
        self.volume_lags = volume_lags or [1, 2, 3, 5, 7]
        self.cross_asset_lags = cross_asset_lags or [1, 2, 3]
        self.macro_lags = macro_lags or [1, 2, 3]
        
        logger.info(
            "lag_feature_generator_initialized",
            price_lags=self.price_lags,
            volume_lags=self.volume_lags,
            cross_asset_lags=self.cross_asset_lags,
            macro_lags=self.macro_lags,
        )

    def generate_price_lags(
        self,
        df: pd.DataFrame,
        price_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate lag features for price columns.
        
        Args:
            df: DataFrame with price data
            price_columns: List of price column names (default: ['close', 'high', 'low', 'open'])
            
        Returns:
            DataFrame with lag features added
        """
        price_columns = price_columns or ['close', 'high', 'low', 'open']
        result = df.copy()
        
        for col in price_columns:
            if col not in df.columns:
                logger.warning("column_not_found", column=col)
                continue
            
            for lag in self.price_lags:
                lag_col = f"{col}_lag{lag}"
                result[lag_col] = df[col].shift(lag)
                
        logger.info(
            "price_lags_generated",
            columns=len(price_columns),
            lags=len(self.price_lags),
            total_features=len(price_columns) * len(self.price_lags),
        )
        
        return result

    def generate_volume_lags(
        self,
        df: pd.DataFrame,
        volume_column: str = 'volume',
    ) -> pd.DataFrame:
        """Generate lag features for volume.
        
        Args:
            df: DataFrame with volume data
            volume_column: Name of volume column
            
        Returns:
            DataFrame with lag features added
        """
        result = df.copy()
        
        if volume_column not in df.columns:
            logger.warning("volume_column_not_found", column=volume_column)
            return result
        
        for lag in self.volume_lags:
            lag_col = f"{volume_column}_lag{lag}"
            result[lag_col] = df[volume_column].shift(lag)
        
        logger.info(
            "volume_lags_generated",
            lags=len(self.volume_lags),
        )
        
        return result

    def generate_log_returns(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
        prefix: str = 'log_return',
    ) -> pd.DataFrame:
        """Generate log returns from price.
        
        Args:
            df: DataFrame with price data
            price_column: Name of price column
            prefix: Prefix for log return column
            
        Returns:
            DataFrame with log return feature added
        """
        result = df.copy()
        
        if price_column not in df.columns:
            logger.warning("price_column_not_found", column=price_column)
            return result
        
        # Calculate log return
        result[prefix] = np.log(df[price_column] / df[price_column].shift(1))
        
        # Generate lags of log return
        for lag in self.price_lags:
            if lag == 1:
                continue  # Already have current log return
            lag_col = f"{prefix}_lag{lag}"
            result[lag_col] = result[prefix].shift(lag - 1)
        
        logger.info("log_returns_generated", column=price_column)
        
        return result

    def generate_price_changes(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
        periods: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Generate price change features.
        
        Args:
            df: DataFrame with price data
            price_column: Name of price column
            periods: List of periods for change calculation (default: [1, 3])
            
        Returns:
            DataFrame with price change features added
        """
        periods = periods or [1, 3]
        result = df.copy()
        
        if price_column not in df.columns:
            logger.warning("price_column_not_found", column=price_column)
            return result
        
        for period in periods:
            # Absolute change
            change_col = f"{price_column}_change{period}"
            result[change_col] = df[price_column].diff(period)
            
            # Percentage change
            pct_col = f"{price_column}_pct_change{period}"
            result[pct_col] = df[price_column].pct_change(period)
        
        logger.info(
            "price_changes_generated",
            periods=periods,
            features=len(periods) * 2,
        )
        
        return result

    def generate_volume_changes(
        self,
        df: pd.DataFrame,
        volume_column: str = 'volume',
        periods: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Generate volume change features.
        
        Args:
            df: DataFrame with volume data
            volume_column: Name of volume column
            periods: List of periods for change calculation (default: [1, 3])
            
        Returns:
            DataFrame with volume change features added
        """
        periods = periods or [1, 3]
        result = df.copy()
        
        if volume_column not in df.columns:
            logger.warning("volume_column_not_found", column=volume_column)
            return result
        
        for period in periods:
            # Absolute change
            change_col = f"{volume_column}_change{period}"
            result[change_col] = df[volume_column].diff(period)
            
            # Percentage change
            pct_col = f"{volume_column}_pct_change{period}"
            result[pct_col] = df[volume_column].pct_change(period)
        
        logger.info(
            "volume_changes_generated",
            periods=periods,
            features=len(periods) * 2,
        )
        
        return result

    def generate_all_lag_features(
        self,
        df: pd.DataFrame,
        price_columns: Optional[List[str]] = None,
        volume_column: str = 'volume',
    ) -> pd.DataFrame:
        """Generate all lag-based features.
        
        Args:
            df: DataFrame with OHLCV data
            price_columns: List of price column names
            volume_column: Name of volume column
            
        Returns:
            DataFrame with all lag features added
        """
        result = df.copy()
        
        # Price lags
        result = self.generate_price_lags(result, price_columns)
        
        # Volume lags
        result = self.generate_volume_lags(result, volume_column)
        
        # Log returns
        result = self.generate_log_returns(result, 'close')
        
        # Price changes
        result = self.generate_price_changes(result, 'close')
        
        # Volume changes
        result = self.generate_volume_changes(result, volume_column)
        
        # Count features
        original_cols = len(df.columns)
        new_cols = len(result.columns)
        features_added = new_cols - original_cols
        
        logger.info(
            "all_lag_features_generated",
            original_columns=original_cols,
            new_columns=new_cols,
            features_added=features_added,
        )
        
        return result

