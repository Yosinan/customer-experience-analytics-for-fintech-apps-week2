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
import pandas as pd


def main():
    """
    Main function to execute Task 1: scraping and preprocessing reviews.
    """
    print("=" * 60)
    print("Task 1: Data Collection and Preprocessing")
    print("=" * 60)
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Scrape reviews for all banks (minimum 400 per bank)
    print("\nStarting review scraping...")
    df = scrape_all_banks(min_reviews_per_bank=400)
    
    if df.empty:
        print("\n✗ Failed to collect reviews. Exiting.")
        return
    
    # Save raw data
    raw_output = "data/raw_reviews.csv"
    df.to_csv(raw_output, index=False)
    print(f"\n✓ Raw reviews saved to {raw_output}")
    
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
    report_path = "data/quality_report.txt"
    os.makedirs("data", exist_ok=True)
    quality_report = generate_quality_report(df_clean, output_path=report_path)
    print("\n" + quality_report)
    
    # Save cleaned data
    clean_output = "data/cleaned_reviews.csv"
    df_clean.to_csv(clean_output, index=False)
    print(f"\n✓ Cleaned reviews saved to {clean_output}")
    
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

