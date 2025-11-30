"""
Data Validation Module for Task 1
Validates scraped review data quality and completeness
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


def validate_review_data(df: pd.DataFrame) -> Dict:
    """
    Validate review data quality and return validation report.
    
    Args:
        df: DataFrame with review data
    
    Returns:
        Dictionary with validation results
    """
    validation_report = {
        'total_reviews': len(df),
        'passed': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_report['passed'] = False
        validation_report['errors'].append("DataFrame is empty")
        return validation_report
    
    # Required columns check
    required_columns = ['review', 'rating', 'date', 'bank', 'source']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation_report['passed'] = False
        validation_report['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check for missing data
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df) * 100).round(2)
    
    validation_report['statistics']['missing_data'] = missing_data.to_dict()
    validation_report['statistics']['missing_data_percentage'] = missing_pct.to_dict()
    
    # Critical: review and rating should not be missing
    if df['review'].isnull().sum() > 0:
        validation_report['passed'] = False
        validation_report['errors'].append(
            f"Found {df['review'].isnull().sum()} reviews with missing text"
        )
    
    if df['rating'].isnull().sum() > 0:
        validation_report['passed'] = False
        validation_report['errors'].append(
            f"Found {df['rating'].isnull().sum()} reviews with missing ratings"
        )
    
    # Check for empty reviews
    empty_reviews = df[df['review'].str.strip().str.len() == 0]
    if len(empty_reviews) > 0:
        validation_report['warnings'].append(
            f"Found {len(empty_reviews)} empty reviews (will be removed)"
        )
    
    # Validate rating range (1-5)
    invalid_ratings = df[(df['rating'] < 1) | (df['rating'] > 5)]
    if len(invalid_ratings) > 0:
        validation_report['passed'] = False
        validation_report['errors'].append(
            f"Found {len(invalid_ratings)} reviews with invalid ratings (not 1-5)"
        )
    
    # Check for duplicates
    duplicate_count = df.duplicated(subset=['review']).sum()
    if duplicate_count > 0:
        validation_report['warnings'].append(
            f"Found {duplicate_count} duplicate reviews"
        )
    
    # Validate date format
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            invalid_dates = df['date'].isnull().sum()
            if invalid_dates > 0:
                validation_report['warnings'].append(
                    f"Found {invalid_dates} reviews with invalid dates"
                )
        except Exception as e:
            validation_report['warnings'].append(f"Date validation error: {str(e)}")
    
    # Check minimum reviews per bank (400)
    if 'bank' in df.columns:
        reviews_per_bank = df['bank'].value_counts()
        validation_report['statistics']['reviews_per_bank'] = reviews_per_bank.to_dict()
        
        banks_below_minimum = reviews_per_bank[reviews_per_bank < 400]
        if len(banks_below_minimum) > 0:
            validation_report['warnings'].append(
                f"Banks below 400 reviews: {banks_below_minimum.to_dict()}"
            )
    
    # Check total review count (minimum 1200)
    if len(df) < 1200:
        validation_report['warnings'].append(
            f"Total reviews ({len(df)}) below minimum requirement (1200)"
        )
    
    # Check missing data percentage (should be <5%)
    critical_columns = ['review', 'rating', 'bank']
    for col in critical_columns:
        if col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df) * 100)
            if missing_pct > 5:
                validation_report['passed'] = False
                validation_report['errors'].append(
                    f"Column '{col}' has {missing_pct:.2f}% missing data (threshold: 5%)"
                )
    
    return validation_report


def clean_review_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean review data based on validation results.
    
    Args:
        df: DataFrame with review data
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Remove rows with missing review text
    df_clean = df_clean[df_clean['review'].notna()]
    df_clean = df_clean[df_clean['review'].str.strip().str.len() > 0]
    
    # Remove rows with missing rating
    df_clean = df_clean[df_clean['rating'].notna()]
    
    # Remove invalid ratings
    df_clean = df_clean[(df_clean['rating'] >= 1) & (df_clean['rating'] <= 5)]
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates(subset=['review'], keep='first')
    
    # Normalize dates
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        df_clean['date'] = df_clean['date'].dt.date
        df_clean['date'] = df_clean['date'].astype(str)
    
    return df_clean.reset_index(drop=True)


def generate_quality_report(df: pd.DataFrame, output_path: str = None) -> str:
    """
    Generate a data quality report.
    
    Args:
        df: DataFrame with review data
        output_path: Optional path to save report
    
    Returns:
        Report as string
    """
    validation = validate_review_data(df)
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("DATA QUALITY REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"\nTotal Reviews: {validation['total_reviews']}")
    report_lines.append(f"Validation Status: {'✓ PASSED' if validation['passed'] else '✗ FAILED'}")
    
    if validation['errors']:
        report_lines.append("\n" + "-" * 60)
        report_lines.append("ERRORS:")
        report_lines.append("-" * 60)
        for error in validation['errors']:
            report_lines.append(f"  ✗ {error}")
    
    if validation['warnings']:
        report_lines.append("\n" + "-" * 60)
        report_lines.append("WARNINGS:")
        report_lines.append("-" * 60)
        for warning in validation['warnings']:
            report_lines.append(f"  ⚠ {warning}")
    
    if validation['statistics']:
        report_lines.append("\n" + "-" * 60)
        report_lines.append("STATISTICS:")
        report_lines.append("-" * 60)
        
        if 'reviews_per_bank' in validation['statistics']:
            report_lines.append("\nReviews per Bank:")
            for bank, count in validation['statistics']['reviews_per_bank'].items():
                report_lines.append(f"  {bank}: {count}")
        
        if 'missing_data_percentage' in validation['statistics']:
            report_lines.append("\nMissing Data Percentage:")
            for col, pct in validation['statistics']['missing_data_percentage'].items():
                if pct > 0:
                    report_lines.append(f"  {col}: {pct:.2f}%")
    
    report = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"✓ Quality report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'review': ['Great app!', 'Not good', ''],
        'rating': [5, 2, None],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'bank': ['CBE', 'BOA', 'Dashen'],
        'source': ['Google Play'] * 3
    })
    
    validation = validate_review_data(sample_data)
    print(validation)
    
    report = generate_quality_report(sample_data)
    print("\n" + report)

