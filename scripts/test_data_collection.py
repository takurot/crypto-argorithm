"""Manual test script for data collection."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collectors.exchange_collector import ExchangeCollector
from src.utils.config import Config, load_env_vars
from src.utils.logging import setup_logging, get_logger

# Setup
setup_logging(log_level="INFO")
load_env_vars()
logger = get_logger(__name__)

def test_exchange_collector():
    """Test exchange data collector with real APIs."""
    logger.info("=== Testing Exchange Data Collector ===")
    
    collector = ExchangeCollector(
        symbols=["BTC"],
        config=Config(),
    )
    
    try:
        # Collect data
        results = collector.collect(symbols=["BTC"], save=True)
        
        # Display results
        for symbol, data in results.items():
            logger.info(
                "collection_result",
                symbol=symbol,
                status=data["status"],
                num_exchanges=data.get("num_exchanges", 0),
            )
            
            if data["status"] == "success":
                prices = data["prices"]
                logger.info(
                    "price_stats",
                    median=f"${prices['median']:.2f}",
                    mean=f"${prices['mean']:.2f}",
                    std=f"${prices['std']:.2f}",
                    min=f"${prices['min']:.2f}",
                    max=f"${prices['max']:.2f}",
                )
                
                logger.info("exchanges_used", count=len(data["exchanges"]))
                for ex in data["exchanges"][:3]:  # Show first 3
                    logger.info(
                        "exchange_detail",
                        exchange=ex["exchange"],
                        price=f"${ex['price']:.2f}",
                    )
                
                if data["num_outliers"] > 0:
                    logger.warning(
                        "outliers_detected",
                        count=data["num_outliers"],
                    )
        
        return True
        
    except Exception as e:
        logger.error("test_failed", error=str(e), exc_info=True)
        return False


def main():
    """Run all tests."""
    logger.info("=== Starting Manual Data Collection Test ===")
    
    success = test_exchange_collector()
    
    if success:
        logger.info("=== All Tests Passed ✅ ===")
        sys.exit(0)
    else:
        logger.error("=== Tests Failed ❌ ===")
        sys.exit(1)


if __name__ == "__main__":
    main()

