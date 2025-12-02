"""
Web scraping module for Google Play Store reviews
Task 1: Data Collection and Preprocessing
"""

import pandas as pd
from google_play_scraper import app, reviews, Sort
from typing import List, Dict, Optional
import time
from datetime import datetime
import logging
from google_play_scraper.exceptions import NotFoundError, ExtraHTTPError

from src.config import Config
from src.utils import retry, log_execution_time

# Set up logger
logger = logging.getLogger(__name__)

# Bank app IDs on Google Play Store - now from config
BANK_APPS = Config.get_bank_apps()


@retry(
    max_retries=Config.SCRAPE_MAX_RETRIES,
    delay=Config.SCRAPE_RETRY_DELAY,
    exceptions=(ExtraHTTPError, ConnectionError, TimeoutError, Exception)
)
@log_execution_time
def _scrape_reviews_batch(
    app_id: str,
    continuation_token: Optional[str] = None,
    count: int = 200,
    sort: Sort = Sort.NEWEST
) -> tuple[List[Dict], Optional[str]]:
    """
    Internal function to scrape a single batch of reviews with retry logic.
    
    Args:
        app_id: Google Play Store app ID
        continuation_token: Token for pagination
        count: Number of reviews per batch
        sort: Sort order for reviews
    
    Returns:
        Tuple of (reviews list, continuation_token)
    """
    try:
        if continuation_token:
            result, token = reviews(
                app_id,
                continuation_token=continuation_token,
                lang='en',
                country='et'
            )
        else:
            result, token = reviews(
                app_id,
                lang='en',
                country='et',
                sort=sort,
                count=min(count, 200),  # API limit per call
                filter_score_with=None  # Get all ratings
            )
        return result, token
    except NotFoundError as e:
        logger.error(f"App not found: {app_id}. Error: {e}")
        raise
    except ExtraHTTPError as e:
        logger.warning(f"HTTP error during scraping: {e}. Will retry...")
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Network error during scraping: {e}. Will retry...")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during scraping batch: {e}")
        raise


def scrape_reviews(
    app_id: str,
    app_name: str,
    bank_name: str,
    count: int = 400,
    sort: Sort = Sort.NEWEST
) -> List[Dict]:
    """
    Scrape reviews from Google Play Store for a given app with error handling and retry logic.
    
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
        logger.info(f"Starting to scrape reviews for {bank_name} (app_id: {app_id})")
        
        # Scrape first batch
        try:
            result, continuation_token = _scrape_reviews_batch(
                app_id=app_id,
                count=count,
                sort=sort
            )
            reviews_data.extend(result)
            logger.info(f"Initial batch: scraped {len(result)} reviews")
        except Exception as e:
            logger.error(f"Failed to scrape initial batch for {bank_name}: {e}")
            return []
        
        # Continue scraping if more reviews are needed
        batch_num = 1
        while len(reviews_data) < count and continuation_token:
            try:
                time.sleep(Config.SCRAPE_RATE_LIMIT_DELAY)  # Rate limiting
                result, continuation_token = _scrape_reviews_batch(
                    app_id=app_id,
                    continuation_token=continuation_token,
                    count=200
                )
                reviews_data.extend(result)
                batch_num += 1
                logger.debug(f"Batch {batch_num}: scraped {len(result)} reviews (total: {len(reviews_data)})")
                
                if not continuation_token:
                    logger.info(f"No more reviews available for {bank_name}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_num} for {bank_name}: {e}")
                # Continue with what we have rather than failing completely
                break
        
        final_count = min(len(reviews_data), count)
        logger.info(f"✓ Successfully scraped {final_count} reviews for {bank_name}")
        return reviews_data[:count]  # Return exactly count reviews
        
    except Exception as e:
        logger.error(f"✗ Critical error scraping {bank_name}: {str(e)}", exc_info=True)
        return []


@log_execution_time
def preprocess_reviews(reviews_data: List[Dict], bank_name: str, app_name: str) -> pd.DataFrame:
    """
    Preprocess scraped reviews into a clean DataFrame with error handling.
    
    Args:
        reviews_data: List of review dictionaries from scraper
        bank_name: Name of the bank
        app_name: Name of the app
    
    Returns:
        Preprocessed DataFrame
    """
    if not reviews_data:
        logger.warning(f"No reviews data provided for {bank_name}")
        return pd.DataFrame()
    
    try:
        # Extract relevant fields
        processed = []
        skipped_count = 0
        
        for review in reviews_data:
            try:
                if not isinstance(review, dict):
                    logger.warning(f"Invalid review format: {type(review)}")
                    skipped_count += 1
                    continue
                
                processed.append({
                    'review': review.get('content', ''),
                    'rating': review.get('score', None),
                    'date': review.get('at', None),
                    'bank': bank_name,
                    'source': 'Google Play Store'
                })
            except Exception as e:
                logger.warning(f"Error processing review: {e}")
                skipped_count += 1
                continue
        
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} invalid reviews for {bank_name}")
        
        if not processed:
            logger.error(f"No valid reviews to process for {bank_name}")
            return pd.DataFrame()
        
        df = pd.DataFrame(processed)
        
        # Remove duplicates based on review text
        initial_count = len(df)
        df = df.drop_duplicates(subset=['review'], keep='first')
        duplicates_removed = initial_count - len(df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate reviews for {bank_name}")
        
        # Handle missing data
        before_drop = len(df)
        df = df.dropna(subset=['review', 'rating'])  # Remove rows with missing review or rating
        dropped_count = before_drop - len(df)
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} reviews with missing data for {bank_name}")
        
        # Normalize dates to YYYY-MM-DD format
        if 'date' in df.columns and len(df) > 0:
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
                df['date'] = df['date'].astype(str)
            except Exception as e:
                logger.warning(f"Error normalizing dates for {bank_name}: {e}")
        
        # Add app name
        df['app_name'] = app_name
        
        logger.info(f"✓ Preprocessed {len(df)} reviews for {bank_name}")
        return df
        
    except Exception as e:
        logger.error(f"✗ Error preprocessing reviews for {bank_name}: {e}", exc_info=True)
        return pd.DataFrame()


@log_execution_time
def scrape_all_banks(min_reviews_per_bank: int = None) -> pd.DataFrame:
    """
    Scrape reviews for all three banks and combine into a single DataFrame with error handling.
    
    Args:
        min_reviews_per_bank: Minimum number of reviews per bank (default: from config)
    
    Returns:
        Combined DataFrame with all reviews
    """
    if min_reviews_per_bank is None:
        min_reviews_per_bank = Config.SCRAPE_MIN_REVIEWS_PER_BANK
    
    all_reviews = []
    failed_banks = []
    
    logger.info(f"Starting to scrape reviews for {len(BANK_APPS)} banks")
    
    for bank_name, app_info in BANK_APPS.items():
        try:
            logger.info(f"\nScraping reviews for {bank_name}...")
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
                    logger.info(f"✓ Successfully collected {len(df)} reviews for {bank_name}")
                else:
                    logger.warning(f"⚠ No valid reviews after preprocessing for {bank_name}")
                    failed_banks.append(bank_name)
            else:
                logger.warning(f"⚠ No reviews collected for {bank_name}")
                failed_banks.append(bank_name)
        except Exception as e:
            logger.error(f"✗ Failed to scrape reviews for {bank_name}: {e}", exc_info=True)
            failed_banks.append(bank_name)
        
        # Rate limiting between banks
        if bank_name != list(BANK_APPS.keys())[-1]:  # Don't sleep after last bank
            time.sleep(Config.SCRAPE_BETWEEN_BANKS_DELAY)
    
    if all_reviews:
        try:
            combined_df = pd.concat(all_reviews, ignore_index=True)
            logger.info(f"\n✓ Total reviews collected: {len(combined_df)}")
            logger.info(f"✓ Reviews per bank:\n{combined_df['bank'].value_counts()}")
            
            if failed_banks:
                logger.warning(f"⚠ Failed to collect reviews for: {', '.join(failed_banks)}")
            
            return combined_df
        except Exception as e:
            logger.error(f"✗ Error combining reviews: {e}", exc_info=True)
            return pd.DataFrame()
    else:
        logger.error("✗ No reviews were collected from any bank")
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

