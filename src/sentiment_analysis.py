"""
Sentiment Analysis Module
Task 2: Sentiment and Thematic Analysis
"""

import pandas as pd
import numpy as np
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """
    Sentiment analysis using multiple methods (DistilBERT and VADER).
    """
    
    def __init__(self, method: str = "distilbert"):
        """
        Initialize sentiment analyzer.
        
        Args:
            method: "distilbert" or "vader"
        """
        self.method = method
        if method == "distilbert":
            try:
                self.analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1  # CPU
                )
            except Exception as e:
                print(f"Warning: Could not load DistilBERT, falling back to VADER: {e}")
                self.method = "vader"
                self.analyzer = SentimentIntensityAnalyzer()
        else:
            self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment_label and sentiment_score
        """
        if not text or pd.isna(text):
            return {"sentiment_label": "neutral", "sentiment_score": 0.0}
        
        try:
            if self.method == "distilbert":
                result = self.analyzer(text[:512])[0]  # Truncate to model limit
                label = result['label'].lower()
                score = result['score']
                
                # Convert to standard format
                if label == "positive":
                    return {"sentiment_label": "positive", "sentiment_score": score}
                else:
                    return {"sentiment_label": "negative", "sentiment_score": score}
            else:  # VADER
                scores = self.analyzer.polarity_scores(text)
                compound = scores['compound']
                
                if compound >= 0.05:
                    label = "positive"
                    score = compound
                elif compound <= -0.05:
                    label = "negative"
                    score = abs(compound)
                else:
                    label = "neutral"
                    score = abs(compound)
                
                return {"sentiment_label": label, "sentiment_score": score}
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return {"sentiment_label": "neutral", "sentiment_score": 0.0}
    
    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
        
        Returns:
            DataFrame with sentiment_label and sentiment_score columns
        """
        results = []
        for text in texts:
            result = self.analyze_text(text)
            results.append(result)
        
        return pd.DataFrame(results)


def aggregate_sentiment_by_bank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment scores by bank and rating.
    
    Args:
        df: DataFrame with sentiment analysis results
    
    Returns:
        Aggregated DataFrame
    """
    aggregation = df.groupby(['bank', 'rating']).agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'sentiment_label': lambda x: x.value_counts().to_dict()
    }).reset_index()
    
    aggregation.columns = ['bank', 'rating', 'mean_sentiment', 'std_sentiment', 
                          'review_count', 'sentiment_distribution']
    
    return aggregation


if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer(method="vader")
    test_texts = [
        "Great app, very fast and easy to use!",
        "This app crashes all the time. Very frustrating.",
        "It's okay, nothing special."
    ]
    
    results = analyzer.analyze_batch(test_texts)
    print(results)

