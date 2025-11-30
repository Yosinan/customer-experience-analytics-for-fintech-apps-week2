"""
Sentiment Comparison Module for Task 2
Compare different sentiment analysis methods and analyze patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from src.sentiment_analysis import SentimentAnalyzer


def compare_sentiment_methods(
    texts: List[str],
    methods: List[str] = ["vader", "distilbert"]
) -> pd.DataFrame:
    """
    Compare sentiment analysis results from different methods.
    
    Args:
        texts: List of texts to analyze
        methods: List of methods to compare
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for method in methods:
        try:
            analyzer = SentimentAnalyzer(method=method)
            method_results = analyzer.analyze_batch(texts)
            method_results['method'] = method
            results.append(method_results)
        except Exception as e:
            print(f"Warning: Could not use method {method}: {e}")
    
    if results:
        comparison_df = pd.concat(results, ignore_index=True)
        return comparison_df
    else:
        return pd.DataFrame()


def analyze_sentiment_by_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze sentiment patterns by rating.
    
    Args:
        df: DataFrame with reviews, ratings, and sentiment
    
    Returns:
        Aggregated analysis by rating
    """
    analysis = df.groupby('rating').agg({
        'sentiment_score': ['mean', 'std', 'min', 'max'],
        'sentiment_label': lambda x: x.value_counts().to_dict(),
        'review': 'count'
    }).reset_index()
    
    analysis.columns = ['rating', 'mean_sentiment', 'std_sentiment', 
                        'min_sentiment', 'max_sentiment', 
                        'sentiment_distribution', 'review_count']
    
    return analysis


def analyze_sentiment_by_theme(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze sentiment patterns by theme.
    
    Args:
        df: DataFrame with reviews, themes, and sentiment
    
    Returns:
        Aggregated analysis by theme
    """
    if 'theme' not in df.columns:
        return pd.DataFrame()
    
    analysis = df.groupby('theme').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'sentiment_label': lambda x: x.value_counts().to_dict(),
        'rating': 'mean'
    }).reset_index()
    
    analysis.columns = ['theme', 'mean_sentiment', 'std_sentiment', 
                        'review_count', 'sentiment_distribution', 'mean_rating']
    
    return analysis.sort_values('mean_sentiment', ascending=False)


def identify_sentiment_drivers(df: pd.DataFrame, bank_name: str = None) -> Dict:
    """
    Identify what drives positive and negative sentiment.
    
    Args:
        df: DataFrame with reviews and sentiment
        bank_name: Optional bank name to filter
    
    Returns:
        Dictionary with drivers analysis
    """
    if bank_name:
        df = df[df['bank'] == bank_name].copy()
    
    # Positive sentiment drivers
    positive_reviews = df[df['sentiment_label'] == 'positive']
    negative_reviews = df[df['sentiment_label'] == 'negative']
    
    drivers = {
        'positive_drivers': {},
        'negative_drivers': {},
        'insights': []
    }
    
    # Analyze by theme
    if 'theme' in df.columns:
        positive_themes = positive_reviews['theme'].value_counts().head(5)
        negative_themes = negative_reviews['theme'].value_counts().head(5)
        
        drivers['positive_drivers']['top_themes'] = positive_themes.to_dict()
        drivers['negative_drivers']['top_themes'] = negative_themes.to_dict()
        
        # Insights
        if len(positive_themes) > 0:
            top_positive = positive_themes.index[0]
            drivers['insights'].append(
                f"Most positive reviews mention: {top_positive}"
            )
        
        if len(negative_themes) > 0:
            top_negative = negative_themes.index[0]
            drivers['insights'].append(
                f"Most negative reviews mention: {top_negative}"
            )
    
    # Analyze by rating
    if len(positive_reviews) > 0:
        drivers['positive_drivers']['avg_rating'] = positive_reviews['rating'].mean()
        drivers['positive_drivers']['count'] = len(positive_reviews)
    
    if len(negative_reviews) > 0:
        drivers['negative_drivers']['avg_rating'] = negative_reviews['rating'].mean()
        drivers['negative_drivers']['count'] = len(negative_reviews)
    
    return drivers


def calculate_sentiment_consistency(df: pd.DataFrame) -> Dict:
    """
    Calculate sentiment consistency metrics.
    
    Args:
        df: DataFrame with sentiment analysis
    
    Returns:
        Dictionary with consistency metrics
    """
    metrics = {}
    
    # Sentiment-label consistency (positive sentiment should have high ratings)
    positive_sentiment = df[df['sentiment_label'] == 'positive']
    negative_sentiment = df[df['sentiment_label'] == 'negative']
    
    if len(positive_sentiment) > 0:
        metrics['positive_sentiment_avg_rating'] = positive_sentiment['rating'].mean()
        metrics['positive_sentiment_high_rating_pct'] = (
            (positive_sentiment['rating'] >= 4).sum() / len(positive_sentiment) * 100
        )
    
    if len(negative_sentiment) > 0:
        metrics['negative_sentiment_avg_rating'] = negative_sentiment['rating'].mean()
        metrics['negative_sentiment_low_rating_pct'] = (
            (negative_sentiment['rating'] <= 2).sum() / len(negative_sentiment) * 100
        )
    
    # Overall consistency score
    if 'rating' in df.columns and 'sentiment_score' in df.columns:
        correlation = df['rating'].corr(df['sentiment_score'])
        metrics['rating_sentiment_correlation'] = correlation
    
    return metrics


if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'review': ['Great app!', 'Not good', 'It\'s okay'],
        'rating': [5, 1, 3],
        'sentiment_label': ['positive', 'negative', 'neutral'],
        'sentiment_score': [0.8, 0.2, 0.5],
        'theme': ['UI', 'Bugs', 'General']
    })
    
    analysis = analyze_sentiment_by_rating(sample_data)
    print(analysis)

