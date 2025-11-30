"""
Web scraping module for Google Play Store reviews
Task 1: Data Collection and Preprocessing
"""

import pandas as pd
from google_play_scraper import app, reviews, Sort
from typing import List, Dict
import time
from datetime import datetime


# Bank app IDs on Google Play Store
BANK_APPS = {
    "Commercial Bank of Ethiopia": {
        "app_id": "com.combanketh.mobilebanking",
        "app_name": "Commercial Bank of Ethiopia"
    },
    "Bank of Abyssinia": {
        "app_id": "com.boa.boaMobileBanking",
        "app_name": "BoA Mobile"
    },
    "Dashen Bank": {
        "app_id": "com.dashen.dashensuperapp",
        "app_name": "Dashen Bank"
    }
}


def scrape_reviews(
    app_id: str,
    app_name: str,
    bank_name: str,
    count: int = 400,
    sort: Sort = Sort.NEWEST
) -> List[Dict]:
    """
    Scrape reviews from Google Play Store for a given app.
    
    Args:
        app_id: Google Play Store app ID
        app_name: Display name of the app
        bank_name: Name of the bank
        count: Number of reviews to scrape (default: 400)
        sort: Sort order for reviews (default: NEWEST)
    
    Returns:
        List of review dictionaries
    """
    reviews_data = []
    continuation_token = None
    
    try:
        # Scrape reviews in batches
        result, continuation_token = reviews(
            app_id,
            lang='en',
            country='et',  # Ethiopia
            sort=sort,
            count=min(count, 200),  # API limit per call
            filter_score_with=None  # Get all ratings
        )
        
        reviews_data.extend(result)
        
        # Continue scraping if more reviews are needed
        while len(reviews_data) < count and continuation_token:
            time.sleep(2)  # Rate limiting
            result, continuation_token = reviews(
                app_id,
                continuation_token=continuation_token,
                lang='en',
                country='et'
            )
            reviews_data.extend(result)
            
            if not continuation_token:
                break
        
        print(f"✓ Scraped {len(reviews_data)} reviews for {bank_name}")
        return reviews_data[:count]  # Return exactly count reviews
        
    except Exception as e:
        print(f"✗ Error scraping {bank_name}: {str(e)}")
        return []


def preprocess_reviews(reviews_data: List[Dict], bank_name: str, app_name: str) -> pd.DataFrame:
    """
    Preprocess scraped reviews into a clean DataFrame.
    
    Args:
        reviews_data: List of review dictionaries from scraper
        bank_name: Name of the bank
        app_name: Name of the app
    
    Returns:
        Preprocessed DataFrame
    """
    if not reviews_data:
        return pd.DataFrame()
    
    # Extract relevant fields
    processed = []
    for review in reviews_data:
        processed.append({
            'review': review.get('content', ''),
            'rating': review.get('score', None),
            'date': review.get('at', None),
            'bank': bank_name,
            'source': 'Google Play Store'
        })
    
    df = pd.DataFrame(processed)
    
    # Remove duplicates based on review text
    df = df.drop_duplicates(subset=['review'], keep='first')
    
    # Handle missing data
    df = df.dropna(subset=['review', 'rating'])  # Remove rows with missing review or rating
    
    # Normalize dates to YYYY-MM-DD format
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df['date'] = df['date'].astype(str)
    
    # Add app name
    df['app_name'] = app_name
    
    return df


def scrape_all_banks(min_reviews_per_bank: int = 400) -> pd.DataFrame:
    """
    Scrape reviews for all three banks and combine into a single DataFrame.
    
    Args:
        min_reviews_per_bank: Minimum number of reviews per bank (default: 400)
    
    Returns:
        Combined DataFrame with all reviews
    """
    all_reviews = []
    
    for bank_name, app_info in BANK_APPS.items():
        print(f"\nScraping reviews for {bank_name}...")
        reviews_data = scrape_reviews(
            app_id=app_info['app_id'],
            app_name=app_info['app_name'],
            bank_name=bank_name,
            count=min_reviews_per_bank
        )
        
        if reviews_data:
            df = preprocess_reviews(reviews_data, bank_name, app_info['app_name'])
            if not df.empty:
                all_reviews.append(df)
        
        time.sleep(3)  # Rate limiting between banks
    
    if all_reviews:
        combined_df = pd.concat(all_reviews, ignore_index=True)
        print(f"\n✓ Total reviews collected: {len(combined_df)}")
        print(f"✓ Reviews per bank:\n{combined_df['bank'].value_counts()}")
        return combined_df
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    # Scrape reviews for all banks
    df = scrape_all_banks(min_reviews_per_bank=400)
    
    if not df.empty:
        # Save to CSV
        output_path = "data/raw_reviews.csv"
        df.to_csv(output_path, index=False)
        print(f"\n✓ Reviews saved to {output_path}")
        print(f"\nDataFrame shape: {df.shape}")
        print(f"\nMissing data:\n{df.isnull().sum()}")
        print(f"\nSample data:\n{df.head()}")
    else:
        print("\n✗ No reviews were collected. Please check app IDs and try again.")

