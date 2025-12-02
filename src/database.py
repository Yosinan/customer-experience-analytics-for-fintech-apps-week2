"""
PostgreSQL Database Module
Task 3: Store Cleaned Data in PostgreSQL
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from psycopg2 import OperationalError, InterfaceError, DatabaseError
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional
import os
from urllib.parse import urlparse
import logging

from src.config import Config
from src.utils import retry, log_execution_time

# Set up logger
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages PostgreSQL database operations for bank reviews.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            database_url: PostgreSQL connection URL
                Format: postgres://user:password@host:port/database?schema=public
        """
        if database_url is None:
            database_url = Config.get_database_url()
        
        self.database_url = database_url
        self.engine = None
        self.conn = None
        logger.info(f"DatabaseManager initialized with URL: {self._mask_password(database_url)}")
    
    @staticmethod
    def _mask_password(url: str) -> str:
        """Mask password in database URL for logging."""
        try:
            parsed = urlparse(url)
            if parsed.password:
                return url.replace(parsed.password, "***")
            return url
        except Exception:
            return "***"
    
    @retry(
        max_retries=Config.DB_CONNECTION_MAX_RETRIES,
        delay=Config.DB_CONNECTION_RETRY_DELAY,
        exceptions=(OperationalError, InterfaceError, ConnectionError, TimeoutError)
    )
    @log_execution_time
    def connect(self):
        """Establish database connection with retry logic."""
        try:
            # Parse URL for psycopg2
            parsed = urlparse(self.database_url)
            
            if not parsed.hostname:
                raise ValueError("Database URL missing hostname")
            
            logger.info(f"Connecting to database at {parsed.hostname}:{parsed.port or 5432}")
            
            self.conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path[1:] if parsed.path else 'kaim',
                user=parsed.username,
                password=parsed.password,
                connect_timeout=Config.DB_CONNECTION_TIMEOUT
            )
            
            # Test connection
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            
            # Create SQLAlchemy engine with connection pooling
            self.engine = create_engine(
                self.database_url,
                pool_size=Config.DB_POOL_SIZE,
                pool_pre_ping=True,  # Verify connections before using
                echo=False
            )
            
            logger.info("✓ Database connection established successfully")
            return True
        except OperationalError as e:
            logger.error(f"✗ Operational error connecting to database: {e}")
            return False
        except InterfaceError as e:
            logger.error(f"✗ Interface error connecting to database: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Unexpected error connecting to database: {e}", exc_info=True)
            return False
    
    @log_execution_time
    def create_schema(self):
        """
        Create database schema (Banks and Reviews tables) with error handling.
        """
        if not self.conn:
            logger.error("✗ No database connection. Call connect() first.")
            return False
        
        try:
            cursor = self.conn.cursor()
            logger.info("Creating database schema...")
            
            # Create Banks table
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS banks (
                        bank_id SERIAL PRIMARY KEY,
                        bank_name VARCHAR(100) NOT NULL UNIQUE,
                        app_name VARCHAR(200) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                logger.debug("Created/verified banks table")
            except DatabaseError as e:
                logger.error(f"Error creating banks table: {e}")
                raise
            
            # Create Reviews table
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS reviews (
                        review_id SERIAL PRIMARY KEY,
                        bank_id INTEGER NOT NULL,
                        review_text TEXT NOT NULL,
                        rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                        review_date DATE,
                        sentiment_label VARCHAR(20),
                        sentiment_score FLOAT,
                        theme VARCHAR(100),
                        source VARCHAR(50) DEFAULT 'Google Play Store',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (bank_id) REFERENCES banks(bank_id) ON DELETE CASCADE
                    );
                """)
                logger.debug("Created/verified reviews table")
            except DatabaseError as e:
                logger.error(f"Error creating reviews table: {e}")
                raise
            
            # Create indexes for better query performance
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_reviews_bank_id ON reviews(bank_id);
                    CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);
                    CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews(sentiment_label);
                    CREATE INDEX IF NOT EXISTS idx_reviews_theme ON reviews(theme);
                """)
                logger.debug("Created/verified indexes")
            except DatabaseError as e:
                logger.warning(f"Error creating indexes (non-critical): {e}")
                # Don't fail on index creation errors
            
            self.conn.commit()
            cursor.close()
            
            logger.info("✓ Database schema created successfully")
            return True
        except DatabaseError as e:
            if self.conn:
                self.conn.rollback()
            logger.error(f"✗ Database error creating schema: {e}", exc_info=True)
            return False
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            logger.error(f"✗ Unexpected error creating schema: {e}", exc_info=True)
            return False
    
    @log_execution_time
    def insert_banks(self, banks_data: pd.DataFrame):
        """
        Insert bank data into Banks table with error handling.
        
        Args:
            banks_data: DataFrame with columns: bank_name, app_name
        """
        if not self.conn:
            logger.error("✗ No database connection.")
            return False
        
        if banks_data.empty:
            logger.warning("No bank data to insert")
            return False
        
        try:
            cursor = self.conn.cursor()
            inserted_count = 0
            
            for idx, row in banks_data.iterrows():
                try:
                    cursor.execute("""
                        INSERT INTO banks (bank_name, app_name)
                        VALUES (%s, %s)
                        ON CONFLICT (bank_name) DO UPDATE
                        SET app_name = EXCLUDED.app_name;
                    """, (row['bank_name'], row['app_name']))
                    inserted_count += 1
                except DatabaseError as e:
                    logger.warning(f"Error inserting bank {row.get('bank_name', 'unknown')}: {e}")
                    continue
            
            self.conn.commit()
            cursor.close()
            
            logger.info(f"✓ Inserted/updated {inserted_count} banks")
            return True
        except DatabaseError as e:
            if self.conn:
                self.conn.rollback()
            logger.error(f"✗ Database error inserting banks: {e}", exc_info=True)
            return False
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            logger.error(f"✗ Unexpected error inserting banks: {e}", exc_info=True)
            return False
    
    def get_bank_ids(self) -> dict:
        """
        Get bank_id mapping from database.
        
        Returns:
            Dictionary mapping bank_name to bank_id
        """
        if not self.conn:
            return {}
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT bank_id, bank_name FROM banks;")
            results = cursor.fetchall()
            cursor.close()
            
            return {name: bank_id for bank_id, name in results}
        except Exception as e:
            print(f"✗ Error fetching bank IDs: {e}")
            return {}
    
    @log_execution_time
    def insert_reviews(self, reviews_data: pd.DataFrame, batch_size: int = 1000):
        """
        Insert review data into Reviews table with error handling and batching.
        
        Args:
            reviews_data: DataFrame with review data
            batch_size: Number of reviews to insert per batch
        """
        if not self.conn:
            logger.error("✗ No database connection.")
            return False
        
        if reviews_data.empty:
            logger.warning("No review data to insert")
            return False
        
        try:
            # Get bank_id mapping
            bank_ids = self.get_bank_ids()
            
            if not bank_ids:
                logger.error("✗ No banks found in database. Insert banks first.")
                return False
            
            # Prepare data for insertion
            cursor = self.conn.cursor()
            insert_count = 0
            skipped_count = 0
            error_count = 0
            
            total_rows = len(reviews_data)
            logger.info(f"Inserting {total_rows} reviews in batches of {batch_size}...")
            
            for idx, row in reviews_data.iterrows():
                try:
                    bank_name = row.get('bank')
                    if not bank_name:
                        skipped_count += 1
                        continue
                    
                    bank_id = bank_ids.get(bank_name)
                    
                    if not bank_id:
                        logger.warning(f"⚠ Bank '{bank_name}' not found in database. Skipping review.")
                        skipped_count += 1
                        continue
                    
                    # Validate required fields
                    review_text = row.get('review', '') or row.get('review_text', '')
                    if not review_text:
                        skipped_count += 1
                        continue
                    
                    cursor.execute("""
                        INSERT INTO reviews (
                            bank_id, review_text, rating, review_date,
                            sentiment_label, sentiment_score, theme, source
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING;
                    """, (
                        bank_id,
                        review_text[:10000],  # Limit text length
                        int(row.get('rating')) if pd.notna(row.get('rating')) else None,
                        row.get('date') if pd.notna(row.get('date')) else None,
                        row.get('sentiment_label'),
                        float(row.get('sentiment_score')) if pd.notna(row.get('sentiment_score')) else None,
                        row.get('theme'),
                        row.get('source', 'Google Play Store')
                    ))
                    insert_count += 1
                    
                    # Commit in batches
                    if insert_count % batch_size == 0:
                        self.conn.commit()
                        logger.debug(f"Committed batch: {insert_count} reviews inserted")
                        
                except DatabaseError as e:
                    error_count += 1
                    logger.warning(f"Database error inserting review {idx}: {e}")
                    if error_count > 100:  # Too many errors, stop
                        logger.error("Too many errors, stopping insertion")
                        break
                    continue
                except Exception as e:
                    error_count += 1
                    logger.warning(f"Error inserting review {idx}: {e}")
                    continue
            
            # Final commit
            self.conn.commit()
            cursor.close()
            
            logger.info(f"✓ Inserted {insert_count} reviews (skipped: {skipped_count}, errors: {error_count})")
            return True
        except DatabaseError as e:
            if self.conn:
                self.conn.rollback()
            logger.error(f"✗ Database error inserting reviews: {e}", exc_info=True)
            return False
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            logger.error(f"✗ Unexpected error inserting reviews: {e}", exc_info=True)
            return False
    
    def verify_data(self):
        """
        Verify data integrity with SQL queries.
        
        Returns:
            Dictionary with verification results
        """
        if not self.conn:
            print("✗ No database connection.")
            return {}
        
        try:
            cursor = self.conn.cursor()
            
            results = {}
            
            # Count reviews per bank
            cursor.execute("""
                SELECT b.bank_name, COUNT(r.review_id) as review_count
                FROM banks b
                LEFT JOIN reviews r ON b.bank_id = r.bank_id
                GROUP BY b.bank_name
                ORDER BY review_count DESC;
            """)
            results['reviews_per_bank'] = cursor.fetchall()
            
            # Average rating per bank
            cursor.execute("""
                SELECT b.bank_name, 
                       AVG(r.rating) as avg_rating,
                       COUNT(r.review_id) as review_count
                FROM banks b
                LEFT JOIN reviews r ON b.bank_id = r.bank_id
                GROUP BY b.bank_name
                ORDER BY avg_rating DESC;
            """)
            results['avg_rating_per_bank'] = cursor.fetchall()
            
            # Total review count
            cursor.execute("SELECT COUNT(*) FROM reviews;")
            results['total_reviews'] = cursor.fetchone()[0]
            
            # Sentiment distribution
            cursor.execute("""
                SELECT sentiment_label, COUNT(*) as count
                FROM reviews
                WHERE sentiment_label IS NOT NULL
                GROUP BY sentiment_label;
            """)
            results['sentiment_distribution'] = cursor.fetchall()
            
            cursor.close()
            
            return results
        except Exception as e:
            print(f"✗ Error verifying data: {e}")
            return {}
    
    def close(self):
        """Close database connection with error handling."""
        try:
            if self.conn:
                self.conn.close()
                logger.info("✓ Database connection closed")
            if self.engine:
                self.engine.dispose()
                logger.debug("SQLAlchemy engine disposed")
        except Exception as e:
            logger.warning(f"Error closing database connection: {e}")


if __name__ == "__main__":
    # Example usage
    db = DatabaseManager()
    if db.connect():
        db.create_schema()
        db.close()

