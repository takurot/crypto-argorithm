"""Base classes for data collectors."""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.logging import get_logger


class BaseCollector(ABC):
    """Abstract base class for data collectors."""

    def __init__(
        self,
        name: str,
        cache_dir: Optional[Path] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        """Initialize base collector.
        
        Args:
            name: Collector name
            cache_dir: Directory for caching raw data
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay for exponential backoff (seconds)
        """
        self.name = name
        self.logger = get_logger(f"{__name__}.{name}")
        self.cache_dir = cache_dir or Path("data/raw")
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        # Setup requests session with retries
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry configuration."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            backoff_factor=1,
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        return delay

    def _get_cache_path(self, date: datetime, suffix: str = "") -> Path:
        """Get cache file path for a given date.
        
        Args:
            date: Date for the data
            suffix: Optional suffix for filename
            
        Returns:
            Path to cache file
        """
        date_str = date.strftime("%Y%m%d")
        filename = f"{self.name}_{date_str}{suffix}.json"
        path = self.cache_dir / date_str / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @abstractmethod
    def collect(self, **kwargs) -> Dict[str, Any]:
        """Collect data from the source.
        
        Returns:
            Dictionary containing collected data
        """
        pass

    def fetch_with_retry(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Fetch data from URL with exponential backoff retry.
        
        Args:
            url: URL to fetch
            params: Optional query parameters
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
            
        Returns:
            JSON response as dictionary
            
        Raises:
            requests.RequestException: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                )
                response.raise_for_status()
                
                self.logger.debug(
                    "fetch_success",
                    url=url,
                    status_code=response.status_code,
                    attempt=attempt + 1,
                )
                
                return response.json()
                
            except requests.RequestException as e:
                self.logger.warning(
                    "fetch_failed",
                    url=url,
                    attempt=attempt + 1,
                    error=str(e),
                    max_retries=self.max_retries,
                )
                
                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff(attempt)
                    self.logger.info("retrying", delay_seconds=delay)
                    time.sleep(delay)
                else:
                    self.logger.error("fetch_exhausted", url=url, error=str(e))
                    raise

        raise requests.RequestException(f"Failed to fetch {url} after {self.max_retries} attempts")

