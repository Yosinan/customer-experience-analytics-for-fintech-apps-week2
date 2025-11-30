"""
Task 1: Data Collection and Preprocessing
Main script to scrape and preprocess Google Play Store reviews
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper import scrape_all_banks
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
    
    # Additional preprocessing: remove empty reviews, ensure data quality
    df_clean = df.copy()
    df_clean = df_clean[df_clean['review'].str.strip().str.len() > 0]
    
    # Check data quality
    missing_pct = (df_clean.isnull().sum() / len(df_clean) * 100).round(2)
    print(f"\nData Quality Check:")
    print(f"  Total reviews: {len(df_clean)}")
    print(f"  Missing data percentage:\n{missing_pct}")
    
    if missing_pct.max() > 5:
        print(f"\n⚠ Warning: Some columns have >5% missing data")
    else:
        print(f"\n✓ Data quality check passed (<5% missing data)")
    
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

