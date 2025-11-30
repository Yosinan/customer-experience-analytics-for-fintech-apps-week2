"""
Advanced Theme Clustering Module for Task 2
Enhanced theme identification and clustering
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import re


class ThemeClusterer:
    """
    Advanced theme clustering using multiple techniques.
    """
    
    def __init__(self, n_clusters: int = 5):
        """
        Initialize theme clusterer.
        
        Args:
            n_clusters: Number of theme clusters to create
        """
        self.n_clusters = n_clusters
        self.vectorizer = None
        self.cluster_model = None
    
    def extract_key_phrases(self, texts: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Extract key phrases using TF-IDF.
        
        Args:
            texts: List of review texts
            top_n: Number of top phrases to return
        
        Returns:
            List of (phrase, score) tuples
        """
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        processed_texts = [t for t in processed_texts if t]
        
        if not processed_texts:
            return []
        
        # TF-IDF with n-grams
        vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
            stop_words='english',
            min_df=2
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get mean TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Sort by score
            phrase_scores = list(zip(feature_names, mean_scores))
            phrase_scores.sort(key=lambda x: x[1], reverse=True)
            
            return phrase_scores[:top_n]
        except Exception as e:
            print(f"Error extracting key phrases: {e}")
            return []
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        if not text or pd.isna(text):
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        return text.strip()
    
    def cluster_reviews(self, texts: List[str]) -> Dict[int, List[int]]:
        """
        Cluster reviews into themes using K-means.
        
        Args:
            texts: List of review texts
        
        Returns:
            Dictionary mapping cluster_id to list of review indices
        """
        processed_texts = [self._preprocess_text(text) for text in texts]
        processed_texts = [t for t in processed_texts if t]
        
        if len(processed_texts) < self.n_clusters:
            return {}
        
        # Vectorize
        self.vectorizer = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
            
            # K-means clustering
            self.cluster_model = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
            cluster_labels = self.cluster_model.fit_predict(tfidf_matrix)
            
            # Group by cluster
            clusters = defaultdict(list)
            for idx, label in enumerate(cluster_labels):
                clusters[label].append(idx)
            
            return dict(clusters)
        except Exception as e:
            print(f"Error clustering reviews: {e}")
            return {}
    
    def get_cluster_keywords(self, texts: List[str], cluster_id: int) -> List[str]:
        """
        Get top keywords for a specific cluster.
        
        Args:
            texts: List of review texts
            cluster_id: Cluster ID
        
        Returns:
            List of top keywords
        """
        if not self.cluster_model or not self.vectorizer:
            return []
        
        # Get cluster center
        cluster_center = self.cluster_model.cluster_centers_[cluster_id]
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Sort by importance
        top_indices = cluster_center.argsort()[-10:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]
        
        return top_keywords
    
    def assign_theme_names(self, clusters: Dict[int, List[int]], 
                          texts: List[str]) -> Dict[int, str]:
        """
        Assign meaningful theme names to clusters.
        
        Args:
            clusters: Dictionary of cluster_id to review indices
            texts: List of review texts
        
        Returns:
            Dictionary mapping cluster_id to theme name
        """
        theme_names = {}
        
        # Theme name mapping based on keywords
        theme_keywords = {
            'Account Access': ['login', 'password', 'account', 'access', 'biometric'],
            'Transaction': ['transfer', 'payment', 'transaction', 'money', 'send'],
            'User Interface': ['ui', 'interface', 'design', 'layout', 'screen'],
            'Performance': ['slow', 'fast', 'speed', 'loading', 'timeout'],
            'Support': ['support', 'help', 'service', 'contact', 'response'],
            'Bugs': ['crash', 'error', 'bug', 'glitch', 'freeze'],
            'Features': ['feature', 'add', 'missing', 'need', 'want']
        }
        
        for cluster_id, review_indices in clusters.items():
            # Get keywords for this cluster
            keywords = self.get_cluster_keywords(texts, cluster_id)
            
            # Match to theme
            best_match = 'General Feedback'
            best_score = 0
            
            for theme_name, theme_keywords_list in theme_keywords.items():
                score = sum(1 for kw in keywords if any(tk in kw for tk in theme_keywords_list))
                if score > best_score:
                    best_score = score
                    best_match = theme_name
            
            theme_names[cluster_id] = best_match
        
        return theme_names


def analyze_theme_sentiment_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze correlation between themes and sentiment.
    
    Args:
        df: DataFrame with themes and sentiment
    
    Returns:
        Correlation analysis DataFrame
    """
    if 'theme' not in df.columns or 'sentiment_score' not in df.columns:
        return pd.DataFrame()
    
    correlation = df.groupby('theme').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'rating': 'mean',
        'sentiment_label': lambda x: (x == 'positive').sum() / len(x) * 100
    }).reset_index()
    
    correlation.columns = ['theme', 'mean_sentiment', 'std_sentiment', 
                          'review_count', 'mean_rating', 'positive_pct']
    
    return correlation.sort_values('mean_sentiment', ascending=False)


def identify_theme_trends(df: pd.DataFrame) -> Dict:
    """
    Identify trends in themes over time.
    
    Args:
        df: DataFrame with themes and dates
    
    Returns:
        Dictionary with trend analysis
    """
    if 'date' not in df.columns or 'theme' not in df.columns:
        return {}
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    if df.empty:
        return {}
    
    # Group by month and theme
    df['year_month'] = df['date'].dt.to_period('M')
    theme_trends = df.groupby(['year_month', 'theme']).size().unstack(fill_value=0)
    
    # Calculate trends
    trends = {}
    for theme in theme_trends.columns:
        theme_data = theme_trends[theme]
        if len(theme_data) > 1:
            # Simple trend: increasing or decreasing
            recent_avg = theme_data.tail(3).mean()
            earlier_avg = theme_data.head(3).mean() if len(theme_data) >= 3 else theme_data.iloc[0]
            
            if recent_avg > earlier_avg * 1.1:
                trend = "increasing"
            elif recent_avg < earlier_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
            
            trends[theme] = {
                'trend': trend,
                'recent_avg': recent_avg,
                'earlier_avg': earlier_avg
            }
    
    return trends


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "Great app, very fast transfers!",
        "Login issues, can't access my account",
        "Love the new UI design",
        "App crashes when making payments"
    ]
    
    clusterer = ThemeClusterer(n_clusters=3)
    clusters = clusterer.cluster_reviews(sample_texts)
    print("Clusters:", clusters)

