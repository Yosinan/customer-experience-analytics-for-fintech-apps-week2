"""
Task 1: Data Collection and Preprocessing
Main script to scrape and preprocess Google Play Store reviews
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper import scrape_all_banks
from src.data_validation import validate_review_data, clean_review_data, generate_quality_report
from src.config import Config
from src.utils import setup_logging, safe_file_operation
import pandas as pd
import logging

# Set up logging
logger = setup_logging(
    log_level=Config.LOG_LEVEL,
    log_file=Config.LOG_FILE,
    log_format=Config.LOG_FORMAT
)

# Ensure directories exist
Config.ensure_directories()


def main():
    """
    Main function to execute Task 1: scraping and preprocessing reviews.
    """
    print("=" * 60)
    print("Task 1: Data Collection and Preprocessing")
    print("=" * 60)
    
    # Scrape reviews for all banks
    logger.info("Starting review scraping...")
    df = scrape_all_banks(min_reviews_per_bank=Config.SCRAPE_MIN_REVIEWS_PER_BANK)
    
    if df.empty:
        logger.error("Failed to collect reviews. Exiting.")
        return
    
    # Save raw data with error handling
    raw_output = os.path.join(Config.DATA_DIR, "raw_reviews.csv")
    try:
        df.to_csv(raw_output, index=False)
        logger.info(f"✓ Raw reviews saved to {raw_output}")
    except Exception as e:
        logger.error(f"Failed to save raw reviews: {e}", exc_info=True)
        return
    
    # Validate data quality
    print("\n" + "-" * 60)
    print("Validating data quality...")
    print("-" * 60)
    
    validation = validate_review_data(df)
    
    if validation['errors']:
        print("\n✗ Validation Errors Found:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("\n⚠ Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    # Clean data based on validation
    print("\nCleaning data...")
    df_clean = clean_review_data(df)
    
    # Generate quality report
    report_path = os.path.join(Config.DATA_DIR, "quality_report.txt")
    try:
        quality_report = generate_quality_report(df_clean, output_path=report_path)
        logger.info("\n" + quality_report)
    except Exception as e:
        logger.error(f"Failed to generate quality report: {e}", exc_info=True)
    
    # Save cleaned data with error handling
    clean_output = os.path.join(Config.DATA_DIR, "cleaned_reviews.csv")
    try:
        df_clean.to_csv(clean_output, index=False)
        logger.info(f"✓ Cleaned reviews saved to {clean_output}")
    except Exception as e:
        logger.error(f"Failed to save cleaned reviews: {e}", exc_info=True)
        return
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Reviews per bank:\n{df_clean['bank'].value_counts()}")
    print(f"  Rating distribution:\n{df_clean['rating'].value_counts().sort_index()}")
    print(f"  Average rating: {df_clean['rating'].mean():.2f}")
    
    print("\n" + "=" * 60)
    print("Task 1 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

