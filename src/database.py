"""
PostgreSQL Database Module
Task 3: Store Cleaned Data in PostgreSQL
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
from typing import Optional
import os
from urllib.parse import urlparse


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
            database_url = os.getenv(
                "DATABASE_URL",
                "postgres://joey:yosinan@localhost:5432/kaim?schema=public"
            )
        
        self.database_url = database_url
        self.engine = None
        self.conn = None
    
    def connect(self):
        """Establish database connection."""
        try:
            # Parse URL for psycopg2
            parsed = urlparse(self.database_url)
            
            self.conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path[1:] if parsed.path else 'kaim',
                user=parsed.username,
                password=parsed.password
            )
            
            # Create SQLAlchemy engine
            self.engine = create_engine(self.database_url)
            
            print("✓ Database connection established")
            return True
        except Exception as e:
            print(f"✗ Error connecting to database: {e}")
            return False
    
    def create_schema(self):
        """
        Create database schema (Banks and Reviews tables).
        """
        if not self.conn:
            print("✗ No database connection. Call connect() first.")
            return False
        
        try:
            cursor = self.conn.cursor()
            
            # Create Banks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS banks (
                    bank_id SERIAL PRIMARY KEY,
                    bank_name VARCHAR(100) NOT NULL UNIQUE,
                    app_name VARCHAR(200) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create Reviews table
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
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_reviews_bank_id ON reviews(bank_id);
                CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);
                CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews(sentiment_label);
                CREATE INDEX IF NOT EXISTS idx_reviews_theme ON reviews(theme);
            """)
            
            self.conn.commit()
            cursor.close()
            
            print("✓ Database schema created successfully")
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Error creating schema: {e}")
            return False
    
    def insert_banks(self, banks_data: pd.DataFrame):
        """
        Insert bank data into Banks table.
        
        Args:
            banks_data: DataFrame with columns: bank_name, app_name
        """
        if not self.conn:
            print("✗ No database connection.")
            return False
        
        try:
            cursor = self.conn.cursor()
            
            for _, row in banks_data.iterrows():
                cursor.execute("""
                    INSERT INTO banks (bank_name, app_name)
                    VALUES (%s, %s)
                    ON CONFLICT (bank_name) DO UPDATE
                    SET app_name = EXCLUDED.app_name;
                """, (row['bank_name'], row['app_name']))
            
            self.conn.commit()
            cursor.close()
            
            print(f"✓ Inserted {len(banks_data)} banks")
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Error inserting banks: {e}")
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
    
    def insert_reviews(self, reviews_data: pd.DataFrame):
        """
        Insert review data into Reviews table.
        
        Args:
            reviews_data: DataFrame with review data
        """
        if not self.conn:
            print("✗ No database connection.")
            return False
        
        try:
            # Get bank_id mapping
            bank_ids = self.get_bank_ids()
            
            if not bank_ids:
                print("✗ No banks found in database. Insert banks first.")
                return False
            
            # Prepare data for insertion
            cursor = self.conn.cursor()
            
            insert_count = 0
            for _, row in reviews_data.iterrows():
                bank_name = row['bank']
                bank_id = bank_ids.get(bank_name)
                
                if not bank_id:
                    print(f"⚠ Warning: Bank '{bank_name}' not found in database. Skipping review.")
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
                    row.get('review', ''),
                    row.get('rating'),
                    row.get('date') if pd.notna(row.get('date')) else None,
                    row.get('sentiment_label'),
                    row.get('sentiment_score'),
                    row.get('theme'),
                    row.get('source', 'Google Play Store')
                ))
                insert_count += 1
            
            self.conn.commit()
            cursor.close()
            
            print(f"✓ Inserted {insert_count} reviews")
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Error inserting reviews: {e}")
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
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")
        if self.engine:
            self.engine.dispose()


if __name__ == "__main__":
    # Example usage
    db = DatabaseManager()
    if db.connect():
        db.create_schema()
        db.close()

