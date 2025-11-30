"""
Unit tests for sentiment analysis module
"""

import pytest
from src.sentiment_analysis import SentimentAnalyzer


def test_sentiment_analyzer_initialization():
    """Test sentiment analyzer initialization."""
    analyzer = SentimentAnalyzer(method="vader")
    assert analyzer is not None
    assert analyzer.method == "vader"


def test_analyze_text_positive():
    """Test sentiment analysis for positive text."""
    analyzer = SentimentAnalyzer(method="vader")
    result = analyzer.analyze_text("Great app! Very fast and easy to use.")
    
    assert 'sentiment_label' in result
    assert 'sentiment_score' in result
    assert result['sentiment_label'] in ['positive', 'negative', 'neutral']
    assert 0 <= result['sentiment_score'] <= 1


def test_analyze_text_negative():
    """Test sentiment analysis for negative text."""
    analyzer = SentimentAnalyzer(method="vader")
    result = analyzer.analyze_text("This app crashes all the time. Very frustrating.")
    
    assert 'sentiment_label' in result
    assert 'sentiment_score' in result
    assert result['sentiment_label'] in ['positive', 'negative', 'neutral']


def test_analyze_text_empty():
    """Test sentiment analysis for empty text."""
    analyzer = SentimentAnalyzer(method="vader")
    result = analyzer.analyze_text("")
    
    assert result['sentiment_label'] == "neutral"
    assert result['sentiment_score'] == 0.0


def test_analyze_batch():
    """Test batch sentiment analysis."""
    analyzer = SentimentAnalyzer(method="vader")
    texts = [
        "Great app!",
        "Not good",
        "It's okay"
    ]
    
    results = analyzer.analyze_batch(texts)
    
    assert len(results) == 3
    assert 'sentiment_label' in results.columns
    assert 'sentiment_score' in results.columns


if __name__ == "__main__":
    pytest.main([__file__])

