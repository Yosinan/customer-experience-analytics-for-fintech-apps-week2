"""
Database Query Utilities for Task 3
Common SQL queries for analyzing bank reviews data
"""

import pandas as pd
from typing import Dict, List, Optional
from src.database import DatabaseManager


class DatabaseQueries:
    """
    Utility class for common database queries.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize with database manager.
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
    
    def get_reviews_by_bank(self, bank_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get all reviews, optionally filtered by bank.
        
        Args:
            bank_name: Optional bank name to filter
        
        Returns:
            DataFrame with reviews
        """
        if not self.db.conn:
            return pd.DataFrame()
        
        if bank_name:
            query = """
                SELECT r.*, b.bank_name, b.app_name
                FROM reviews r
                JOIN banks b ON r.bank_id = b.bank_id
                WHERE b.bank_name = %s
                ORDER BY r.review_date DESC, r.created_at DESC;
            """
            return pd.read_sql_query(query, self.db.conn, params=(bank_name,))
        else:
            query = """
                SELECT r.*, b.bank_name, b.app_name
                FROM reviews r
                JOIN banks b ON r.bank_id = b.bank_id
                ORDER BY r.review_date DESC, r.created_at DESC;
            """
            return pd.read_sql_query(query, self.db.conn)
    
    def get_reviews_by_rating(self, rating: int, bank_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get reviews filtered by rating.
        
        Args:
            rating: Rating (1-5)
            bank_name: Optional bank name to filter
        
        Returns:
            DataFrame with reviews
        """
        if not self.db.conn:
            return pd.DataFrame()
        
        if bank_name:
            query = """
                SELECT r.*, b.bank_name, b.app_name
                FROM reviews r
                JOIN banks b ON r.bank_id = b.bank_id
                WHERE r.rating = %s AND b.bank_name = %s
                ORDER BY r.review_date DESC;
            """
            return pd.read_sql_query(query, self.db.conn, params=(rating, bank_name))
        else:
            query = """
                SELECT r.*, b.bank_name, b.app_name
                FROM reviews r
                JOIN banks b ON r.bank_id = b.bank_id
                WHERE r.rating = %s
                ORDER BY r.review_date DESC;
            """
            return pd.read_sql_query(query, self.db.conn, params=(rating,))
    
    def get_reviews_by_sentiment(self, sentiment: str, bank_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get reviews filtered by sentiment.
        
        Args:
            sentiment: Sentiment label (positive, negative, neutral)
            bank_name: Optional bank name to filter
        
        Returns:
            DataFrame with reviews
        """
        if not self.db.conn:
            return pd.DataFrame()
        
        if bank_name:
            query = """
                SELECT r.*, b.bank_name, b.app_name
                FROM reviews r
                JOIN banks b ON r.bank_id = b.bank_id
                WHERE r.sentiment_label = %s AND b.bank_name = %s
                ORDER BY r.sentiment_score DESC;
            """
            return pd.read_sql_query(query, self.db.conn, params=(sentiment, bank_name))
        else:
            query = """
                SELECT r.*, b.bank_name, b.app_name
                FROM reviews r
                JOIN banks b ON r.bank_id = b.bank_id
                WHERE r.sentiment_label = %s
                ORDER BY r.sentiment_score DESC;
            """
            return pd.read_sql_query(query, self.db.conn, params=(sentiment,))
    
    def get_reviews_by_theme(self, theme: str, bank_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get reviews filtered by theme.
        
        Args:
            theme: Theme name
            bank_name: Optional bank name to filter
        
        Returns:
            DataFrame with reviews
        """
        if not self.db.conn:
            return pd.DataFrame()
        
        if bank_name:
            query = """
                SELECT r.*, b.bank_name, b.app_name
                FROM reviews r
                JOIN banks b ON r.bank_id = b.bank_id
                WHERE r.theme = %s AND b.bank_name = %s
                ORDER BY r.review_date DESC;
            """
            return pd.read_sql_query(query, self.db.conn, params=(theme, bank_name))
        else:
            query = """
                SELECT r.*, b.bank_name, b.app_name
                FROM reviews r
                JOIN banks b ON r.bank_id = b.bank_id
                WHERE r.theme = %s
                ORDER BY r.review_date DESC;
            """
            return pd.read_sql_query(query, self.db.conn, params=(theme,))
    
    def get_bank_statistics(self) -> pd.DataFrame:
        """
        Get comprehensive statistics for each bank.
        
        Returns:
            DataFrame with bank statistics
        """
        if not self.db.conn:
            return pd.DataFrame()
        
        query = """
            SELECT 
                b.bank_name,
                b.app_name,
                COUNT(r.review_id) as total_reviews,
                AVG(r.rating) as avg_rating,
                STDDEV(r.rating) as rating_stddev,
                COUNT(CASE WHEN r.rating = 5 THEN 1 END) as five_star_count,
                COUNT(CASE WHEN r.rating = 1 THEN 1 END) as one_star_count,
                AVG(r.sentiment_score) as avg_sentiment_score,
                COUNT(CASE WHEN r.sentiment_label = 'positive' THEN 1 END) as positive_count,
                COUNT(CASE WHEN r.sentiment_label = 'negative' THEN 1 END) as negative_count,
                COUNT(CASE WHEN r.sentiment_label = 'neutral' THEN 1 END) as neutral_count,
                MIN(r.review_date) as earliest_review,
                MAX(r.review_date) as latest_review
            FROM banks b
            LEFT JOIN reviews r ON b.bank_id = r.bank_id
            GROUP BY b.bank_id, b.bank_name, b.app_name
            ORDER BY avg_rating DESC;
        """
        
        return pd.read_sql_query(query, self.db.conn)
    
    def get_theme_statistics(self, bank_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get statistics by theme.
        
        Args:
            bank_name: Optional bank name to filter
        
        Returns:
            DataFrame with theme statistics
        """
        if not self.db.conn:
            return pd.DataFrame()
        
        if bank_name:
            query = """
                SELECT 
                    r.theme,
                    COUNT(r.review_id) as review_count,
                    AVG(r.rating) as avg_rating,
                    AVG(r.sentiment_score) as avg_sentiment,
                    COUNT(CASE WHEN r.sentiment_label = 'positive' THEN 1 END) as positive_count,
                    COUNT(CASE WHEN r.sentiment_label = 'negative' THEN 1 END) as negative_count
                FROM reviews r
                JOIN banks b ON r.bank_id = b.bank_id
                WHERE b.bank_name = %s AND r.theme IS NOT NULL
                GROUP BY r.theme
                ORDER BY review_count DESC;
            """
            return pd.read_sql_query(query, self.db.conn, params=(bank_name,))
        else:
            query = """
                SELECT 
                    r.theme,
                    COUNT(r.review_id) as review_count,
                    AVG(r.rating) as avg_rating,
                    AVG(r.sentiment_score) as avg_sentiment,
                    COUNT(CASE WHEN r.sentiment_label = 'positive' THEN 1 END) as positive_count,
                    COUNT(CASE WHEN r.sentiment_label = 'negative' THEN 1 END) as negative_count
                FROM reviews r
                WHERE r.theme IS NOT NULL
                GROUP BY r.theme
                ORDER BY review_count DESC;
            """
            return pd.read_sql_query(query, self.db.conn)
    
    def get_recent_reviews(self, limit: int = 10, bank_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get most recent reviews.
        
        Args:
            limit: Number of reviews to return
            bank_name: Optional bank name to filter
        
        Returns:
            DataFrame with recent reviews
        """
        if not self.db.conn:
            return pd.DataFrame()
        
        if bank_name:
            query = """
                SELECT r.*, b.bank_name, b.app_name
                FROM reviews r
                JOIN banks b ON r.bank_id = b.bank_id
                WHERE b.bank_name = %s
                ORDER BY r.review_date DESC, r.created_at DESC
                LIMIT %s;
            """
            return pd.read_sql_query(query, self.db.conn, params=(bank_name, limit))
        else:
            query = """
                SELECT r.*, b.bank_name, b.app_name
                FROM reviews r
                JOIN banks b ON r.bank_id = b.bank_id
                ORDER BY r.review_date DESC, r.created_at DESC
                LIMIT %s;
            """
            return pd.read_sql_query(query, self.db.conn, params=(limit,))
    
    def search_reviews(self, search_term: str, bank_name: Optional[str] = None) -> pd.DataFrame:
        """
        Search reviews by text content.
        
        Args:
            search_term: Search term to look for in review text
            bank_name: Optional bank name to filter
        
        Returns:
            DataFrame with matching reviews
        """
        if not self.db.conn:
            return pd.DataFrame()
        
        search_pattern = f"%{search_term}%"
        
        if bank_name:
            query = """
                SELECT r.*, b.bank_name, b.app_name
                FROM reviews r
                JOIN banks b ON r.bank_id = b.bank_id
                WHERE r.review_text ILIKE %s AND b.bank_name = %s
                ORDER BY r.review_date DESC;
            """
            return pd.read_sql_query(query, self.db.conn, params=(search_pattern, bank_name))
        else:
            query = """
                SELECT r.*, b.bank_name, b.app_name
                FROM reviews r
                JOIN banks b ON r.bank_id = b.bank_id
                WHERE r.review_text ILIKE %s
                ORDER BY r.review_date DESC;
            """
            return pd.read_sql_query(query, self.db.conn, params=(search_pattern,))


def export_database_to_csv(db_manager: DatabaseManager, output_dir: str = "data/exports"):
    """
    Export all database tables to CSV files.
    
    Args:
        db_manager: DatabaseManager instance
        output_dir: Output directory for CSV files
    """
    import os
    
    if not db_manager.conn:
        print("✗ No database connection.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    queries = DatabaseQueries(db_manager)
    
    # Export all reviews
    reviews_df = queries.get_reviews_by_bank()
    if not reviews_df.empty:
        reviews_df.to_csv(f"{output_dir}/all_reviews.csv", index=False)
        print(f"✓ Exported {len(reviews_df)} reviews to {output_dir}/all_reviews.csv")
    
    # Export bank statistics
    bank_stats = queries.get_bank_statistics()
    if not bank_stats.empty:
        bank_stats.to_csv(f"{output_dir}/bank_statistics.csv", index=False)
        print(f"✓ Exported bank statistics to {output_dir}/bank_statistics.csv")
    
    # Export theme statistics
    theme_stats = queries.get_theme_statistics()
    if not theme_stats.empty:
        theme_stats.to_csv(f"{output_dir}/theme_statistics.csv", index=False)
        print(f"✓ Exported theme statistics to {output_dir}/theme_statistics.csv")
    
    # Export by bank
    for bank_name in bank_stats['bank_name'].unique():
        bank_reviews = queries.get_reviews_by_bank(bank_name=bank_name)
        if not bank_reviews.empty:
            safe_name = bank_name.replace(' ', '_').replace('/', '_')
            bank_reviews.to_csv(f"{output_dir}/reviews_{safe_name}.csv", index=False)
            print(f"✓ Exported {len(bank_reviews)} reviews for {bank_name}")


if __name__ == "__main__":
    # Example usage
    from src.database import DatabaseManager
    
    db = DatabaseManager()
    if db.connect():
        queries = DatabaseQueries(db)
        
        # Get bank statistics
        stats = queries.get_bank_statistics()
        print(stats)
        
        db.close()

