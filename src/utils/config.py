"""Configuration loader for YAML config files."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Configuration manager for loading YAML config files."""

    def __init__(self, config_dir: str = "config"):
        """Initialize config loader.
        
        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Any] = {}

    def load(self, config_name: str) -> Dict[str, Any]:
        """Load a config file by name.
        
        Args:
            config_name: Name of config file without extension
            
        Returns:
            Dictionary containing configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        if config_name in self._configs:
            return self._configs[config_name]

        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._configs[config_name] = config
        return config

    def get(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """Get a specific config value using dot notation.
        
        Args:
            config_name: Name of config file
            key_path: Dot-separated path to config value (e.g., "exchanges.min_required")
            default: Default value if key not found
            
        Returns:
            Config value or default
        """
        config = self.load(config_name)
        
        keys = key_path.split(".")
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value


def load_env_vars() -> None:
    """Load environment variables from .env file if it exists."""
    from dotenv import load_dotenv
    
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)

