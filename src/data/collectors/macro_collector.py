"""Macro economic data collector from FRED and Yahoo Finance."""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from src.data.collectors.base import BaseCollector
from src.utils.config import Config
from src.utils.logging import get_logger


class MacroCollector(BaseCollector):
    """Collector for macro economic indicators."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        config: Optional[Config] = None,
    ):
        """Initialize macro collector.
        
        Args:
            cache_dir: Directory for caching raw data
            config: Configuration object
        """
        super().__init__("macro", cache_dir)
        
        self.config = config or Config()
        
        # Load macro configurations
        self.macro_config = self.config.load("data_sources")["macro"]
        self.fred_config = self.macro_config["fred"]
        self.yahoo_config = self.macro_config["yahoo_finance"]
        
        # Get API key from environment
        self.fred_api_key = os.getenv(self.fred_config["api_key_env"])
        if not self.fred_api_key:
            self.logger.warning(
                "fred_api_key_missing",
                env_var=self.fred_config["api_key_env"],
            )
        
        self.logger.info("macro_collector_initialized")

    def _fetch_fred_series(
        self,
        series_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Fetch a time series from FRED.
        
        Args:
            series_id: FRED series ID
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with date index and value column
        """
        if not self.fred_api_key:
            self.logger.error("fred_api_key_required", series_id=series_id)
            return None
        
        try:
            url = f"{self.fred_config['base_url']}/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.fred_api_key,
                "file_type": "json",
                "observation_start": start_date.strftime("%Y-%m-%d"),
                "observation_end": end_date.strftime("%Y-%m-%d"),
            }
            
            data = self.fetch_with_retry(url, params=params)
            
            if "observations" not in data:
                self.logger.error("invalid_fred_response", series_id=series_id)
                return None
            
            # Convert to DataFrame
            observations = data["observations"]
            df = pd.DataFrame(observations)
            
            if df.empty:
                self.logger.warning("empty_fred_series", series_id=series_id)
                return None
            
            # Process DataFrame
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["value"] != "."]  # Remove missing values marked as "."
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])
            df = df.set_index("date")
            df = df[["value"]]
            df.columns = [series_id]
            
            self.logger.info(
                "fred_series_fetched",
                series_id=series_id,
                records=len(df),
                start=df.index[0].isoformat(),
                end=df.index[-1].isoformat(),
            )
            
            return df
            
        except Exception as e:
            self.logger.error(
                "fred_fetch_error",
                series_id=series_id,
                error=str(e),
            )
            return None

    def _fetch_yahoo_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance.
        
        Args:
            symbol: Yahoo Finance symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with date index and close price
        """
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )
            
            if df.empty:
                self.logger.warning("empty_yahoo_data", symbol=symbol)
                return None
            
            # Extract close price
            df = df[["Close"]]
            df.columns = [symbol]
            
            self.logger.info(
                "yahoo_data_fetched",
                symbol=symbol,
                records=len(df),
                start=df.index[0].isoformat(),
                end=df.index[-1].isoformat(),
            )
            
            return df
            
        except Exception as e:
            self.logger.error(
                "yahoo_fetch_error",
                symbol=symbol,
                error=str(e),
            )
            return None

    def collect_fred_series(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """Collect all FRED series.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping series name to DataFrame
        """
        results = {}
        
        for name, series_id in self.fred_config["series"].items():
            df = self._fetch_fred_series(series_id, start_date, end_date)
            if df is not None:
                results[name] = df
            
        return results

    def collect_yahoo_symbols(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """Collect all Yahoo Finance symbols.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping symbol name to DataFrame
        """
        results = {}
        
        for name, symbol in self.yahoo_config["symbols"].items():
            df = self._fetch_yahoo_symbol(symbol, start_date, end_date)
            if df is not None:
                results[name] = df
            
        return results

    def collect(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Collect all macro economic data.
        
        Args:
            start_date: Start date (default: 2 years ago)
            end_date: End date (default: today)
            save: Whether to save results to cache
            
        Returns:
            Dictionary containing all collected data
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        
        if start_date is None:
            start_date = end_date - timedelta(days=730)  # 2 years
        
        self.logger.info(
            "collecting_macro_data",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        
        # Collect from FRED
        fred_data = self.collect_fred_series(start_date, end_date)
        
        # Collect from Yahoo Finance
        yahoo_data = self.collect_yahoo_symbols(start_date, end_date)
        
        # Combine results
        results = {
            "fred": fred_data,
            "yahoo": yahoo_data,
            "metadata": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "fred_series_count": len(fred_data),
                "yahoo_symbols_count": len(yahoo_data),
            }
        }
        
        # Save to cache if requested
        if save:
            cache_path = self._get_cache_path(end_date, suffix="_macro")
            
            # Save metadata
            metadata_path = cache_path.parent / f"{cache_path.stem}_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(results["metadata"], f, indent=2)
            
            # Save DataFrames as CSV
            for source, data_dict in [("fred", fred_data), ("yahoo", yahoo_data)]:
                for name, df in data_dict.items():
                    csv_path = cache_path.parent / f"{cache_path.stem}_{source}_{name}.csv"
                    df.to_csv(csv_path)
                    self.logger.debug("data_saved", path=str(csv_path))
            
            self.logger.info("results_saved", path=str(cache_path.parent))
        
        return results

