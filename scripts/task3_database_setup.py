"""
Task 3: Store Cleaned Data in PostgreSQL
Main script to set up database and insert review data
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.database import DatabaseManager


def main():
    """
    Main function to execute Task 3: database setup and data insertion.
    """
    print("=" * 60)
    print("Task 3: Store Cleaned Data in PostgreSQL")
    print("=" * 60)
    
    # Load analyzed reviews
    input_file = "data/analyzed_reviews.csv"
    if not os.path.exists(input_file):
        print(f"✗ Error: {input_file} not found. Please run Task 2 first.")
        return
    
    df = pd.read_csv(input_file)
    print(f"\n✓ Loaded {len(df)} analyzed reviews")
    
    # Initialize database manager
    database_url = os.getenv(
        "DATABASE_URL",
        "postgres://joey:yosinan@localhost:5432/kaim?schema=public"
    )
    
    db = DatabaseManager(database_url=database_url)
    
    # Connect to database
    print("\nConnecting to database...")
    if not db.connect():
        print("✗ Failed to connect to database. Exiting.")
        return
    
    # Create schema
    print("\n" + "-" * 60)
    print("Creating database schema...")
    print("-" * 60)
    if not db.create_schema():
        print("✗ Failed to create schema. Exiting.")
        db.close()
        return
    
    # Prepare banks data
    print("\n" + "-" * 60)
    print("Preparing banks data...")
    print("-" * 60)
    
    banks_data = df[['bank', 'app_name']].drop_duplicates()
    banks_data.columns = ['bank_name', 'app_name']
    print(f"  Banks to insert: {len(banks_data)}")
    for _, row in banks_data.iterrows():
        print(f"    - {row['bank_name']}: {row['app_name']}")
    
    # Insert banks
    if not db.insert_banks(banks_data):
        print("✗ Failed to insert banks. Exiting.")
        db.close()
        return
    
    # Insert reviews
    print("\n" + "-" * 60)
    print("Inserting reviews...")
    print("-" * 60)
    
    if not db.insert_reviews(df):
        print("✗ Failed to insert reviews. Exiting.")
        db.close()
        return
    
    # Verify data
    print("\n" + "-" * 60)
    print("Verifying data integrity...")
    print("-" * 60)
    
    verification = db.verify_data()
    
    print(f"\nTotal reviews in database: {verification.get('total_reviews', 0)}")
    
    print("\nReviews per bank:")
    for bank_name, count in verification.get('reviews_per_bank', []):
        print(f"  {bank_name}: {count}")
    
    print("\nAverage rating per bank:")
    for bank_name, avg_rating, count in verification.get('avg_rating_per_bank', []):
        if avg_rating:
            print(f"  {bank_name}: {avg_rating:.2f} ({count} reviews)")
    
    print("\nSentiment distribution:")
    for sentiment, count in verification.get('sentiment_distribution', []):
        print(f"  {sentiment}: {count}")
    
    # Save schema to file
    print("\n" + "-" * 60)
    print("Saving database schema...")
    print("-" * 60)
    
    schema_file = "database_schema.sql"
    with open(schema_file, 'w') as f:
        f.write("""
-- Database Schema for Bank Reviews
-- Generated for Task 3

-- Banks Table
CREATE TABLE IF NOT EXISTS banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) NOT NULL UNIQUE,
    app_name VARCHAR(200) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Reviews Table
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

-- Indexes
CREATE INDEX IF NOT EXISTS idx_reviews_bank_id ON reviews(bank_id);
CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);
CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews(sentiment_label);
CREATE INDEX IF NOT EXISTS idx_reviews_theme ON reviews(theme);
""")
    
    print(f"✓ Schema saved to {schema_file}")
    
    # Close connection
    db.close()
    
    print("\n" + "=" * 60)
    print("Task 3 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

