"""
Centralized Configuration Module
Manages all configuration settings via environment variables with sensible defaults.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Centralized configuration management."""
    
    # Database Configuration
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgres://user:password@localhost:5432/kaim?schema=public"
    )
    
    # Scraping Configuration
    SCRAPE_MIN_REVIEWS_PER_BANK: int = int(os.getenv("SCRAPE_MIN_REVIEWS_PER_BANK", "400"))
    SCRAPE_RATE_LIMIT_DELAY: float = float(os.getenv("SCRAPE_RATE_LIMIT_DELAY", "2.0"))
    SCRAPE_BETWEEN_BANKS_DELAY: float = float(os.getenv("SCRAPE_BETWEEN_BANKS_DELAY", "3.0"))
    SCRAPE_MAX_RETRIES: int = int(os.getenv("SCRAPE_MAX_RETRIES", "3"))
    SCRAPE_RETRY_DELAY: float = float(os.getenv("SCRAPE_RETRY_DELAY", "5.0"))
    SCRAPE_TIMEOUT: int = int(os.getenv("SCRAPE_TIMEOUT", "30"))
    
    # Database Connection Configuration
    DB_CONNECTION_MAX_RETRIES: int = int(os.getenv("DB_CONNECTION_MAX_RETRIES", "3"))
    DB_CONNECTION_RETRY_DELAY: float = float(os.getenv("DB_CONNECTION_RETRY_DELAY", "2.0"))
    DB_CONNECTION_TIMEOUT: int = int(os.getenv("DB_CONNECTION_TIMEOUT", "10"))
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
    
    # Sentiment Analysis Configuration
    SENTIMENT_METHOD: str = os.getenv("SENTIMENT_METHOD", "vader")  # "vader" or "distilbert"
    SENTIMENT_BATCH_SIZE: int = int(os.getenv("SENTIMENT_BATCH_SIZE", "100"))
    SENTIMENT_MAX_RETRIES: int = int(os.getenv("SENTIMENT_MAX_RETRIES", "2"))
    
    # Thematic Analysis Configuration
    THEME_N_TOPICS: int = int(os.getenv("THEME_N_TOPICS", "5"))
    THEME_MAX_FEATURES: int = int(os.getenv("THEME_MAX_FEATURES", "50"))
    THEME_MIN_DF: int = int(os.getenv("THEME_MIN_DF", "2"))
    
    # File I/O Configuration
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    FIGURES_DIR: str = os.getenv("FIGURES_DIR", "figures")
    SUMMARY_DIR: str = os.getenv("SUMMARY_DIR", "summary")
    FILE_WRITE_RETRIES: int = int(os.getenv("FILE_WRITE_RETRIES", "3"))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", "logs/app.log")
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Bank App IDs Configuration
    BANK_APPS: dict = {
        "Commercial Bank of Ethiopia": {
            "app_id": os.getenv(
                "CBE_APP_ID",
                "com.combanketh.mobilebanking"
            ),
            "app_name": "Commercial Bank of Ethiopia"
        },
        "Bank of Abyssinia": {
            "app_id": os.getenv(
                "BOA_APP_ID",
                "com.boa.boaMobileBanking"
            ),
            "app_name": "BoA Mobile"
        },
        "Dashen Bank": {
            "app_id": os.getenv(
                "DASHEN_APP_ID",
                "com.dashen.dashensuperapp"
            ),
            "app_name": "Dashen Bank"
        }
    }
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get database URL from environment or default."""
        return cls.DATABASE_URL
    
    @classmethod
    def get_bank_apps(cls) -> dict:
        """Get bank apps configuration."""
        return cls.BANK_APPS
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        directories = [cls.DATA_DIR, cls.FIGURES_DIR, cls.SUMMARY_DIR]
        if cls.LOG_FILE:
            log_dir = os.path.dirname(cls.LOG_FILE)
            if log_dir:
                directories.append(log_dir)
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

