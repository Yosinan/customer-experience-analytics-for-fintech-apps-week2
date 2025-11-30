"""
Database Validation and Health Checks for Task 3
Validate database integrity and data quality
"""

import pandas as pd
from typing import Dict, List
from src.database import DatabaseManager


class DatabaseValidator:
    """
    Validates database integrity and data quality.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize validator.
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
    
    def validate_schema(self) -> Dict:
        """
        Validate database schema exists and is correct.
        
        Returns:
            Dictionary with validation results
        """
        if not self.db.conn:
            return {"valid": False, "errors": ["No database connection"]}
        
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "tables": {}
        }
        
        try:
            cursor = self.db.conn.cursor()
            
            # Check if tables exist
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('banks', 'reviews');
            """)
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['banks', 'reviews']
            for table in required_tables:
                if table in existing_tables:
                    results["tables"][table] = "exists"
                else:
                    results["valid"] = False
                    results["errors"].append(f"Table '{table}' does not exist")
            
            # Check foreign key constraint
            if 'banks' in existing_tables and 'reviews' in existing_tables:
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM information_schema.table_constraints 
                    WHERE constraint_type = 'FOREIGN KEY' 
                    AND table_name = 'reviews';
                """)
                fk_count = cursor.fetchone()[0]
                if fk_count == 0:
                    results["warnings"].append("Foreign key constraint may be missing")
            
            # Check indexes
            cursor.execute("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = 'reviews' 
                AND schemaname = 'public';
            """)
            indexes = [row[0] for row in cursor.fetchall()]
            expected_indexes = ['idx_reviews_bank_id', 'idx_reviews_rating', 
                              'idx_reviews_sentiment', 'idx_reviews_theme']
            for idx in expected_indexes:
                if idx not in indexes:
                    results["warnings"].append(f"Index '{idx}' may be missing")
            
            cursor.close()
            
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Schema validation error: {str(e)}")
        
        return results
    
    def validate_data_integrity(self) -> Dict:
        """
        Validate data integrity constraints.
        
        Returns:
            Dictionary with validation results
        """
        if not self.db.conn:
            return {"valid": False, "errors": ["No database connection"]}
        
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        try:
            cursor = self.db.conn.cursor()
            
            # Check for orphaned reviews (reviews without valid bank_id)
            cursor.execute("""
                SELECT COUNT(*) 
                FROM reviews r 
                LEFT JOIN banks b ON r.bank_id = b.bank_id 
                WHERE b.bank_id IS NULL;
            """)
            orphaned_count = cursor.fetchone()[0]
            if orphaned_count > 0:
                results["valid"] = False
                results["errors"].append(f"Found {orphaned_count} orphaned reviews")
            
            # Check for invalid ratings
            cursor.execute("""
                SELECT COUNT(*) 
                FROM reviews 
                WHERE rating < 1 OR rating > 5;
            """)
            invalid_ratings = cursor.fetchone()[0]
            if invalid_ratings > 0:
                results["valid"] = False
                results["errors"].append(f"Found {invalid_ratings} reviews with invalid ratings")
            
            # Check for missing required fields
            cursor.execute("""
                SELECT COUNT(*) 
                FROM reviews 
                WHERE review_text IS NULL OR review_text = '';
            """)
            empty_reviews = cursor.fetchone()[0]
            if empty_reviews > 0:
                results["warnings"].append(f"Found {empty_reviews} reviews with empty text")
            
            # Check for missing bank_id
            cursor.execute("""
                SELECT COUNT(*) 
                FROM reviews 
                WHERE bank_id IS NULL;
            """)
            missing_bank_id = cursor.fetchone()[0]
            if missing_bank_id > 0:
                results["valid"] = False
                results["errors"].append(f"Found {missing_bank_id} reviews with missing bank_id")
            
            # Get statistics
            cursor.execute("SELECT COUNT(*) FROM reviews;")
            results["statistics"]["total_reviews"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM banks;")
            results["statistics"]["total_banks"] = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(DISTINCT bank_id) 
                FROM reviews;
            """)
            results["statistics"]["banks_with_reviews"] = cursor.fetchone()[0]
            
            cursor.close()
            
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Data integrity check error: {str(e)}")
        
        return results
    
    def validate_data_quality(self) -> Dict:
        """
        Validate data quality metrics.
        
        Returns:
            Dictionary with quality metrics
        """
        if not self.db.conn:
            return {"valid": False, "errors": ["No database connection"]}
        
        results = {
            "valid": True,
            "warnings": [],
            "metrics": {}
        }
        
        try:
            # Get data quality metrics
            query = """
                SELECT 
                    COUNT(*) as total_reviews,
                    COUNT(DISTINCT bank_id) as unique_banks,
                    COUNT(CASE WHEN rating IS NOT NULL THEN 1 END) as reviews_with_rating,
                    COUNT(CASE WHEN sentiment_label IS NOT NULL THEN 1 END) as reviews_with_sentiment,
                    COUNT(CASE WHEN theme IS NOT NULL THEN 1 END) as reviews_with_theme,
                    COUNT(CASE WHEN review_date IS NOT NULL THEN 1 END) as reviews_with_date,
                    AVG(rating) as avg_rating,
                    AVG(sentiment_score) as avg_sentiment_score
                FROM reviews;
            """
            
            df = pd.read_sql_query(query, self.db.conn)
            
            if not df.empty:
                row = df.iloc[0]
                total = row['total_reviews']
                
                results["metrics"] = {
                    "total_reviews": int(total),
                    "unique_banks": int(row['unique_banks']),
                    "reviews_with_rating": int(row['reviews_with_rating']),
                    "reviews_with_sentiment": int(row['reviews_with_sentiment']),
                    "reviews_with_theme": int(row['reviews_with_theme']),
                    "reviews_with_date": int(row['reviews_with_date']),
                    "avg_rating": float(row['avg_rating']) if row['avg_rating'] else None,
                    "avg_sentiment_score": float(row['avg_sentiment_score']) if row['avg_sentiment_score'] else None
                }
                
                # Check completeness
                if total > 0:
                    rating_pct = (row['reviews_with_rating'] / total) * 100
                    sentiment_pct = (row['reviews_with_sentiment'] / total) * 100
                    theme_pct = (row['reviews_with_theme'] / total) * 100
                    
                    if rating_pct < 95:
                        results["warnings"].append(f"Only {rating_pct:.1f}% of reviews have ratings")
                    if sentiment_pct < 90:
                        results["warnings"].append(f"Only {sentiment_pct:.1f}% of reviews have sentiment")
                    if theme_pct < 80:
                        results["warnings"].append(f"Only {theme_pct:.1f}% of reviews have themes")
        
        except Exception as e:
            results["valid"] = False
            results["errors"] = [f"Data quality check error: {str(e)}"]
        
        return results
    
    def run_full_validation(self) -> Dict:
        """
        Run all validation checks.
        
        Returns:
            Dictionary with all validation results
        """
        print("Running database validation...")
        
        results = {
            "schema": self.validate_schema(),
            "data_integrity": self.validate_data_integrity(),
            "data_quality": self.validate_data_quality(),
            "overall_valid": True
        }
        
        # Determine overall validity
        if not results["schema"]["valid"] or not results["data_integrity"]["valid"]:
            results["overall_valid"] = False
        
        return results


if __name__ == "__main__":
    # Example usage
    from src.database import DatabaseManager
    
    db = DatabaseManager()
    if db.connect():
        validator = DatabaseValidator(db)
        results = validator.run_full_validation()
        print(results)
        db.close()

