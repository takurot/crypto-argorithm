"""Technical indicator features for cryptocurrency trading."""

from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

from src.utils.logging import get_logger


logger = get_logger(__name__)


class TechnicalIndicatorGenerator:
    """Generate technical indicator features."""

    def __init__(
        self,
        sma_windows: Optional[List[int]] = None,
        ema_windows: Optional[List[int]] = None,
        rsi_window: int = 14,
        macd_windows: Optional[Tuple[int, int, int]] = None,
        bollinger_window: int = 20,
        bollinger_std: int = 2,
        atr_window: int = 14,
    ):
        """Initialize technical indicator generator.
        
        Args:
            sma_windows: Windows for Simple Moving Average (default: [7,21,50])
            ema_windows: Windows for Exponential Moving Average (default: [9,26,50])
            rsi_window: Window for RSI calculation (default: 14)
            macd_windows: (fast, slow, signal) for MACD (default: (12,26,9))
            bollinger_window: Window for Bollinger Bands (default: 20)
            bollinger_std: Number of standard deviations for Bollinger (default: 2)
            atr_window: Window for ATR calculation (default: 14)
        """
        self.sma_windows = sma_windows or [7, 21, 50]
        self.ema_windows = ema_windows or [9, 26, 50]
        self.rsi_window = rsi_window
        self.macd_windows = macd_windows or (12, 26, 9)
        self.bollinger_window = bollinger_window
        self.bollinger_std = bollinger_std
        self.atr_window = atr_window
        
        logger.info(
            "technical_indicator_generator_initialized",
            sma_windows=self.sma_windows,
            ema_windows=self.ema_windows,
            rsi_window=self.rsi_window,
        )

    def generate_sma(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
    ) -> pd.DataFrame:
        """Generate Simple Moving Average features.
        
        Args:
            df: DataFrame with price data
            price_column: Name of price column
            
        Returns:
            DataFrame with SMA features added
        """
        result = df.copy()
        
        if price_column not in df.columns:
            logger.warning("price_column_not_found", column=price_column)
            return result
        
        for window in self.sma_windows:
            col_name = f"sma{window}"
            result[col_name] = df[price_column].rolling(window=window).mean()
            
            # Price deviation from SMA
            dev_col = f"price_sma{window}_dev"
            result[dev_col] = (df[price_column] - result[col_name]) / result[col_name]
        
        logger.info("sma_generated", windows=self.sma_windows)
        
        return result

    def generate_ema(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
    ) -> pd.DataFrame:
        """Generate Exponential Moving Average features.
        
        Args:
            df: DataFrame with price data
            price_column: Name of price column
            
        Returns:
            DataFrame with EMA features added
        """
        result = df.copy()
        
        if price_column not in df.columns:
            logger.warning("price_column_not_found", column=price_column)
            return result
        
        for window in self.ema_windows:
            col_name = f"ema{window}"
            result[col_name] = df[price_column].ewm(span=window, adjust=False).mean()
            
            # Price deviation from EMA
            dev_col = f"price_ema{window}_dev"
            result[dev_col] = (df[price_column] - result[col_name]) / result[col_name]
        
        logger.info("ema_generated", windows=self.ema_windows)
        
        return result

    def generate_rsi(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
    ) -> pd.DataFrame:
        """Generate Relative Strength Index (RSI).
        
        Args:
            df: DataFrame with price data
            price_column: Name of price column
            
        Returns:
            DataFrame with RSI feature added
        """
        result = df.copy()
        
        if price_column not in df.columns:
            logger.warning("price_column_not_found", column=price_column)
            return result
        
        # Calculate price changes
        delta = df[price_column].diff()
        
        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calculate exponential moving average of gains and losses
        avg_gain = gain.ewm(span=self.rsi_window, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_window, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        result['rsi'] = rsi
        
        logger.info("rsi_generated", window=self.rsi_window)
        
        return result

    def generate_macd(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
    ) -> pd.DataFrame:
        """Generate MACD (Moving Average Convergence Divergence).
        
        Args:
            df: DataFrame with price data
            price_column: Name of price column
            
        Returns:
            DataFrame with MACD features added
        """
        result = df.copy()
        
        if price_column not in df.columns:
            logger.warning("price_column_not_found", column=price_column)
            return result
        
        fast, slow, signal = self.macd_windows
        
        # Calculate EMAs
        ema_fast = df[price_column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[price_column].ewm(span=slow, adjust=False).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        result['macd'] = macd_line
        
        # Signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        result['macd_signal'] = signal_line
        
        # MACD histogram
        result['macd_hist'] = macd_line - signal_line
        
        logger.info("macd_generated", windows=self.macd_windows)
        
        return result

    def generate_bollinger_bands(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
    ) -> pd.DataFrame:
        """Generate Bollinger Bands.
        
        Args:
            df: DataFrame with price data
            price_column: Name of price column
            
        Returns:
            DataFrame with Bollinger Bands features added
        """
        result = df.copy()
        
        if price_column not in df.columns:
            logger.warning("price_column_not_found", column=price_column)
            return result
        
        # Middle band (SMA)
        sma = df[price_column].rolling(window=self.bollinger_window).mean()
        result['bb_middle'] = sma
        
        # Standard deviation
        std = df[price_column].rolling(window=self.bollinger_window).std()
        
        # Upper and lower bands
        result['bb_upper'] = sma + (std * self.bollinger_std)
        result['bb_lower'] = sma - (std * self.bollinger_std)
        
        # Bandwidth
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        
        # %B indicator (position within bands)
        result['bb_pct'] = (df[price_column] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        
        logger.info(
            "bollinger_bands_generated",
            window=self.bollinger_window,
            std=self.bollinger_std,
        )
        
        return result

    def generate_atr(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate Average True Range (ATR).
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with ATR feature added
        """
        result = df.copy()
        
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.warning("required_columns_missing", required=required_cols)
            return result
        
        # True Range components
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        # True Range
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Average True Range
        result['atr'] = tr.rolling(window=self.atr_window).mean()
        
        # ATR as percentage of price
        result['atr_pct'] = result['atr'] / df['close']
        
        logger.info("atr_generated", window=self.atr_window)
        
        return result

    def generate_momentum(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
        window: int = 10,
    ) -> pd.DataFrame:
        """Generate Momentum indicator.
        
        Args:
            df: DataFrame with price data
            price_column: Name of price column
            window: Lookback window for momentum
            
        Returns:
            DataFrame with momentum feature added
        """
        result = df.copy()
        
        if price_column not in df.columns:
            logger.warning("price_column_not_found", column=price_column)
            return result
        
        result['momentum'] = df[price_column] - df[price_column].shift(window)
        result['momentum_pct'] = result['momentum'] / df[price_column].shift(window)
        
        logger.info("momentum_generated", window=window)
        
        return result

    def generate_all_technical_indicators(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
    ) -> pd.DataFrame:
        """Generate all technical indicator features.
        
        Args:
            df: DataFrame with OHLCV data
            price_column: Name of price column
            
        Returns:
            DataFrame with all technical indicators added
        """
        result = df.copy()
        
        # Moving averages
        result = self.generate_sma(result, price_column)
        result = self.generate_ema(result, price_column)
        
        # Oscillators
        result = self.generate_rsi(result, price_column)
        result = self.generate_macd(result, price_column)
        
        # Volatility indicators
        result = self.generate_bollinger_bands(result, price_column)
        result = self.generate_atr(result)
        
        # Momentum
        result = self.generate_momentum(result, price_column)
        
        # Count features
        original_cols = len(df.columns)
        new_cols = len(result.columns)
        features_added = new_cols - original_cols
        
        logger.info(
            "all_technical_indicators_generated",
            original_columns=original_cols,
            new_columns=new_cols,
            features_added=features_added,
        )
        
        return result

