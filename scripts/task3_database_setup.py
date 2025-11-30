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
from src.db_queries import DatabaseQueries, export_database_to_csv
from src.db_validation import DatabaseValidator


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
    
    # Database Validation
    print("\n" + "-" * 60)
    print("Running database validation...")
    print("-" * 60)
    
    validator = DatabaseValidator(db)
    validation_results = validator.run_full_validation()
    
    # Schema validation
    if validation_results["schema"]["valid"]:
        print("✓ Schema validation passed")
    else:
        print("✗ Schema validation failed:")
        for error in validation_results["schema"]["errors"]:
            print(f"  - {error}")
    
    if validation_results["schema"]["warnings"]:
        print("⚠ Schema warnings:")
        for warning in validation_results["schema"]["warnings"]:
            print(f"  - {warning}")
    
    # Data integrity validation
    if validation_results["data_integrity"]["valid"]:
        print("\n✓ Data integrity validation passed")
    else:
        print("\n✗ Data integrity validation failed:")
        for error in validation_results["data_integrity"]["errors"]:
            print(f"  - {error}")
    
    if validation_results["data_integrity"]["warnings"]:
        print("⚠ Data integrity warnings:")
        for warning in validation_results["data_integrity"]["warnings"]:
            print(f"  - {warning}")
    
    # Data quality metrics
    quality = validation_results["data_quality"]
    if "metrics" in quality:
        print("\nData Quality Metrics:")
        metrics = quality["metrics"]
        print(f"  Total reviews: {metrics.get('total_reviews', 0)}")
        print(f"  Unique banks: {metrics.get('unique_banks', 0)}")
        print(f"  Reviews with rating: {metrics.get('reviews_with_rating', 0)}")
        print(f"  Reviews with sentiment: {metrics.get('reviews_with_sentiment', 0)}")
        print(f"  Reviews with theme: {metrics.get('reviews_with_theme', 0)}")
        if metrics.get('avg_rating'):
            print(f"  Average rating: {metrics['avg_rating']:.2f}")
        if metrics.get('avg_sentiment_score'):
            print(f"  Average sentiment score: {metrics['avg_sentiment_score']:.3f}")
    
    if quality.get("warnings"):
        print("\n⚠ Data quality warnings:")
        for warning in quality["warnings"]:
            print(f"  - {warning}")
    
    # Database Queries Examples
    print("\n" + "-" * 60)
    print("Database Query Examples")
    print("-" * 60)
    
    queries = DatabaseQueries(db)
    
    # Get bank statistics
    bank_stats = queries.get_bank_statistics()
    if not bank_stats.empty:
        print("\nBank Statistics:")
        print(bank_stats.to_string(index=False))
        
        # Save to file
        bank_stats.to_csv("data/bank_statistics_db.csv", index=False)
        print(f"\n✓ Bank statistics saved to data/bank_statistics_db.csv")
    
    # Get theme statistics
    theme_stats = queries.get_theme_statistics()
    if not theme_stats.empty:
        print("\nTheme Statistics:")
        print(theme_stats.head(10).to_string(index=False))
        
        # Save to file
        theme_stats.to_csv("data/theme_statistics_db.csv", index=False)
        print(f"✓ Theme statistics saved to data/theme_statistics_db.csv")
    
    # Export database to CSV
    print("\n" + "-" * 60)
    print("Exporting database to CSV files...")
    print("-" * 60)
    export_database_to_csv(db, output_dir="data/exports")
    
    # Close connection
    db.close()
    
    print("\n" + "=" * 60)
    print("Task 3 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

