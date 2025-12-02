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
import logging

warnings.filterwarnings('ignore')

from src.config import Config
from src.utils import retry, log_execution_time

# Set up logger
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analysis using multiple methods (DistilBERT and VADER).
    """
    
    def __init__(self, method: str = None):
        """
        Initialize sentiment analyzer with error handling.
        
        Args:
            method: "distilbert" or "vader" (default: from config)
        """
        if method is None:
            method = Config.SENTIMENT_METHOD
        
        self.method = method
        if method == "distilbert":
            try:
                logger.info("Loading DistilBERT sentiment analyzer...")
                self.analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1  # CPU
                )
                logger.info("✓ DistilBERT loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load DistilBERT, falling back to VADER: {e}")
                self.method = "vader"
                self.analyzer = SentimentIntensityAnalyzer()
                logger.info("Using VADER sentiment analyzer")
        else:
            logger.info("Using VADER sentiment analyzer")
            self.analyzer = SentimentIntensityAnalyzer()
    
    @retry(
        max_retries=Config.SENTIMENT_MAX_RETRIES,
        delay=1.0,
        exceptions=(Exception,)
    )
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text with error handling.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment_label and sentiment_score
        """
        if not text or pd.isna(text) or not isinstance(text, str):
            logger.debug("Empty or invalid text provided, returning neutral sentiment")
            return {"sentiment_label": "neutral", "sentiment_score": 0.0}
        
        try:
            if self.method == "distilbert":
                # Truncate to model limit
                truncated_text = text[:512] if len(text) > 512 else text
                result = self.analyzer(truncated_text)[0]
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
            logger.warning(f"Error analyzing text (returning neutral): {e}")
            return {"sentiment_label": "neutral", "sentiment_score": 0.0}
    
    @log_execution_time
    def analyze_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts with error handling.
        
        Args:
            texts: List of texts to analyze
        
        Returns:
            DataFrame with sentiment_label and sentiment_score columns
        """
        if not texts:
            logger.warning("Empty text list provided for batch analysis")
            return pd.DataFrame(columns=['sentiment_label', 'sentiment_score'])
        
        results = []
        error_count = 0
        
        for idx, text in enumerate(texts):
            try:
                result = self.analyze_text(text)
                results.append(result)
            except Exception as e:
                error_count += 1
                logger.warning(f"Error analyzing text at index {idx}: {e}")
                results.append({"sentiment_label": "neutral", "sentiment_score": 0.0})
        
        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during batch analysis")
        
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

