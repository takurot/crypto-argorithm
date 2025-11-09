"""Exchange data collector for cryptocurrency prices."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from src.data.collectors.base import BaseCollector
from src.utils.config import Config
from src.utils.logging import get_logger


class ExchangeCollector(BaseCollector):
    """Collector for cryptocurrency prices from multiple exchanges."""

    def __init__(
        self,
        symbols: List[str] = None,
        cache_dir: Optional[Path] = None,
        config: Optional[Config] = None,
    ):
        """Initialize exchange collector.
        
        Args:
            symbols: List of symbols to collect (e.g., ["BTC", "ETH", "SOL", "BNB"])
            cache_dir: Directory for caching raw data
            config: Configuration object
        """
        super().__init__("exchange", cache_dir)
        
        self.symbols = symbols or ["BTC"]
        self.config = config or Config()
        
        # Load exchange configurations
        self.exchanges_config = self.config.load("data_sources")["exchanges"]
        self.apis = self.exchanges_config["apis"]
        self.min_required = self.exchanges_config["min_required"]
        self.outlier_threshold = self.exchanges_config["outlier_threshold_pct"] / 100.0
        
        self.logger.info(
            "exchange_collector_initialized",
            symbols=self.symbols,
            num_exchanges=len(self.apis),
            min_required=self.min_required,
        )

    def _fetch_single_exchange(
        self,
        exchange: str,
        symbol: str,
        quote: str = "USD",
    ) -> Optional[Dict[str, Any]]:
        """Fetch price from a single exchange.
        
        Args:
            exchange: Exchange name
            symbol: Base symbol (e.g., "BTC")
            quote: Quote currency (e.g., "USD", "USDT")
            
        Returns:
            Dictionary with exchange data or None if failed
        """
        try:
            config = self.apis[exchange.lower()]
            
            # Format symbol according to exchange convention
            symbol_format = config["symbol_format"]
            formatted_symbol = symbol_format.format(base=symbol, quote=quote)
            
            # Some exchanges require lowercase
            if exchange.lower() in ["bitstamp", "gemini"]:
                formatted_symbol = formatted_symbol.lower()
            
            # Adjust quote for USDT-based exchanges
            if quote == "USD" and exchange.lower() in ["binance", "okx", "kucoin", "gateio", "bybit"]:
                formatted_symbol = symbol_format.format(base=symbol, quote="USDT")
            
            # Build URL
            url = config["url"]
            if "{symbol}" in url:
                url = url.format(symbol=formatted_symbol)
            
            params = {}
            if "{symbol}" not in config["url"] and exchange.lower() in ["kraken", "okx", "kucoin"]:
                if exchange.lower() == "kraken":
                    params = {"pair": formatted_symbol}
                elif exchange.lower() == "okx":
                    params = {"instId": formatted_symbol}
                elif exchange.lower() == "kucoin":
                    params = {"symbol": formatted_symbol}
            
            # Fetch data
            data = self.fetch_with_retry(url, params=params, timeout=10)
            
            # Extract price based on exchange format
            price = self._extract_price(exchange.lower(), data)
            
            if price is None:
                self.logger.warning("price_extraction_failed", exchange=exchange, symbol=symbol)
                return None
            
            return {
                "exchange": exchange,
                "symbol": symbol,
                "quote": quote,
                "price": price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_data": data,
            }
            
        except Exception as e:
            self.logger.error(
                "exchange_fetch_error",
                exchange=exchange,
                symbol=symbol,
                error=str(e),
            )
            return None

    def _extract_price(self, exchange: str, data: Dict[str, Any]) -> Optional[float]:
        """Extract price from exchange-specific response format.
        
        Args:
            exchange: Exchange name (lowercase)
            data: Raw response data
            
        Returns:
            Price as float or None if extraction failed
        """
        try:
            if exchange == "binance":
                return float(data["price"])
            elif exchange == "coinbase":
                return float(data["data"]["amount"])
            elif exchange == "kraken":
                pair_key = list(data["result"].keys())[0]
                return float(data["result"][pair_key]["c"][0])
            elif exchange == "okx":
                return float(data["data"][0]["last"])
            elif exchange == "kucoin":
                return float(data["data"]["price"])
            elif exchange == "gateio":
                return float(data[0]["last"])
            elif exchange == "bitstamp":
                return float(data["last"])
            elif exchange == "gemini":
                return float(data["last"])
            elif exchange == "cryptocom":
                return float(data["result"]["data"][0]["a"])
            elif exchange == "bybit":
                return float(data["result"]["list"][0]["lastPrice"])
            else:
                self.logger.warning("unknown_exchange_format", exchange=exchange)
                return None
        except (KeyError, IndexError, ValueError) as e:
            self.logger.error("price_parse_error", exchange=exchange, error=str(e))
            return None

    def collect_symbol(
        self,
        symbol: str,
        quote: str = "USD",
    ) -> Dict[str, Any]:
        """Collect prices for a single symbol from all exchanges.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC")
            quote: Quote currency
            
        Returns:
            Dictionary containing aggregated price data
        """
        self.logger.info("collecting_prices", symbol=symbol, quote=quote)
        
        prices = []
        
        with ThreadPoolExecutor(max_workers=len(self.apis)) as executor:
            futures = {
                executor.submit(self._fetch_single_exchange, ex, symbol, quote): ex
                for ex in self.apis.keys()
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None and result["price"] is not None:
                    prices.append(result)
        
        # Check minimum exchanges requirement
        num_valid = len(prices)
        if num_valid < self.min_required:
            self.logger.error(
                "insufficient_exchanges",
                symbol=symbol,
                valid=num_valid,
                required=self.min_required,
            )
            return {
                "symbol": symbol,
                "quote": quote,
                "status": "insufficient_data",
                "num_exchanges": num_valid,
                "required_exchanges": self.min_required,
                "prices": prices,
            }
        
        # Calculate statistics and remove outliers
        price_values = [p["price"] for p in prices]
        median_price = np.median(price_values)
        
        # Filter outliers (beyond ±3% of median)
        filtered_prices = []
        outliers = []
        
        for price_data in prices:
            price = price_data["price"]
            deviation = abs(price - median_price) / median_price
            
            if deviation <= self.outlier_threshold:
                filtered_prices.append(price_data)
            else:
                outliers.append(price_data)
                self.logger.warning(
                    "outlier_detected",
                    exchange=price_data["exchange"],
                    price=price,
                    median=median_price,
                    deviation_pct=deviation * 100,
                )
        
        # Recalculate statistics after filtering
        if len(filtered_prices) < self.min_required:
            self.logger.warning(
                "insufficient_after_filtering",
                symbol=symbol,
                filtered=len(filtered_prices),
                outliers=len(outliers),
            )
            # Use all prices if filtering removes too many
            filtered_prices = prices
            outliers = []
        
        final_price_values = [p["price"] for p in filtered_prices]
        
        return {
            "symbol": symbol,
            "quote": quote,
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_exchanges": len(filtered_prices),
            "num_outliers": len(outliers),
            "prices": {
                "median": float(np.median(final_price_values)),
                "mean": float(np.mean(final_price_values)),
                "std": float(np.std(final_price_values)),
                "min": float(np.min(final_price_values)),
                "max": float(np.max(final_price_values)),
            },
            "exchanges": filtered_prices,
            "outliers": outliers,
        }

    def collect(
        self,
        symbols: Optional[List[str]] = None,
        quote: str = "USD",
        save: bool = True,
    ) -> Dict[str, Any]:
        """Collect prices for multiple symbols.
        
        Args:
            symbols: List of symbols to collect (defaults to self.symbols)
            quote: Quote currency
            save: Whether to save results to cache
            
        Returns:
            Dictionary containing all collected data
        """
        symbols = symbols or self.symbols
        
        results = {}
        for symbol in symbols:
            result = self.collect_symbol(symbol, quote)
            results[symbol] = result
        
        # Save to cache if requested
        if save:
            cache_path = self._get_cache_path(datetime.now(timezone.utc), suffix=f"_{quote}")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info("results_saved", path=str(cache_path))
        
        return results

