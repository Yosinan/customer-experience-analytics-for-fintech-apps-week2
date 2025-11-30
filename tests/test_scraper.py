"""
Unit tests for scraper module
"""

import pytest
import pandas as pd
from src.scraper import preprocess_reviews


def test_preprocess_reviews():
    """Test review preprocessing function."""
    # Sample review data
    reviews_data = [
        {
            'content': 'Great app!',
            'score': 5,
            'at': '2024-01-01T00:00:00',
        },
        {
            'content': 'Not good',
            'score': 2,
            'at': '2024-01-02T00:00:00',
        }
    ]
    
    df = preprocess_reviews(reviews_data, 'Test Bank', 'Test App')
    
    assert not df.empty
    assert len(df) == 2
    assert 'review' in df.columns
    assert 'rating' in df.columns
    assert 'bank' in df.columns
    assert df['bank'].iloc[0] == 'Test Bank'


def test_preprocess_reviews_removes_duplicates():
    """Test that duplicate reviews are removed."""
    reviews_data = [
        {
            'content': 'Great app!',
            'score': 5,
            'at': '2024-01-01T00:00:00',
        },
        {
            'content': 'Great app!',  # Duplicate
            'score': 5,
            'at': '2024-01-01T00:00:00',
        }
    ]
    
    df = preprocess_reviews(reviews_data, 'Test Bank', 'Test App')
    
    assert len(df) == 1  # Duplicate removed


def test_preprocess_reviews_handles_missing_data():
    """Test that missing data is handled correctly."""
    reviews_data = [
        {
            'content': 'Great app!',
            'score': 5,
            'at': '2024-01-01T00:00:00',
        },
        {
            'content': None,  # Missing review
            'score': 3,
            'at': '2024-01-02T00:00:00',
        }
    ]
    
    df = preprocess_reviews(reviews_data, 'Test Bank', 'Test App')
    
    # Should remove row with missing review
    assert len(df) == 1


if __name__ == "__main__":
    pytest.main([__file__])

