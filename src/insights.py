"""
Insights and Recommendations Module
Task 4: Insights and Recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter


def identify_drivers_and_pain_points(df: pd.DataFrame, bank_name: str) -> Dict:
    """
    Identify satisfaction drivers and pain points for a specific bank.
    
    Args:
        df: DataFrame with analyzed reviews
        bank_name: Name of the bank
    
    Returns:
        Dictionary with drivers and pain points
    """
    bank_df = df[df['bank'] == bank_name].copy()
    
    if bank_df.empty:
        return {"drivers": [], "pain_points": []}
    
    # Drivers: Positive sentiment + high ratings
    positive_reviews = bank_df[
        (bank_df['sentiment_label'] == 'positive') & 
        (bank_df['rating'] >= 4)
    ]
    
    # Pain points: Negative sentiment + low ratings
    negative_reviews = bank_df[
        (bank_df['sentiment_label'] == 'negative') & 
        (bank_df['rating'] <= 2)
    ]
    
    # Extract themes from positive reviews (drivers)
    driver_themes = positive_reviews['theme'].value_counts().head(3).to_dict()
    drivers = [
        {
            "theme": theme,
            "count": count,
            "percentage": (count / len(positive_reviews) * 100) if len(positive_reviews) > 0 else 0
        }
        for theme, count in driver_themes.items()
    ]
    
    # Extract themes from negative reviews (pain points)
    pain_point_themes = negative_reviews['theme'].value_counts().head(3).to_dict()
    pain_points = [
        {
            "theme": theme,
            "count": count,
            "percentage": (count / len(negative_reviews) * 100) if len(negative_reviews) > 0 else 0
        }
        for theme, count in pain_point_themes.items()
    ]
    
    return {
        "drivers": drivers,
        "pain_points": pain_points,
        "total_reviews": len(bank_df),
        "avg_rating": bank_df['rating'].mean(),
        "avg_sentiment": bank_df['sentiment_score'].mean()
    }


def compare_banks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare metrics across banks.
    
    Args:
        df: DataFrame with analyzed reviews
    
    Returns:
        Comparison DataFrame
    """
    comparison = []
    
    for bank in df['bank'].unique():
        bank_df = df[df['bank'] == bank]
        
        comparison.append({
            'bank': bank,
            'total_reviews': len(bank_df),
            'avg_rating': bank_df['rating'].mean(),
            'avg_sentiment_score': bank_df['sentiment_score'].mean(),
            'positive_pct': (bank_df['sentiment_label'] == 'positive').sum() / len(bank_df) * 100,
            'negative_pct': (bank_df['sentiment_label'] == 'negative').sum() / len(bank_df) * 100,
            'neutral_pct': (bank_df['sentiment_label'] == 'neutral').sum() / len(bank_df) * 100,
            '5_star_pct': (bank_df['rating'] == 5).sum() / len(bank_df) * 100,
            '1_star_pct': (bank_df['rating'] == 1).sum() / len(bank_df) * 100
        })
    
    return pd.DataFrame(comparison)


def generate_recommendations(
    df: pd.DataFrame,
    bank_name: str,
    insights: Dict
) -> List[str]:
    """
    Generate actionable recommendations for a bank.
    
    Args:
        df: DataFrame with analyzed reviews
        bank_name: Name of the bank
        insights: Dictionary with drivers and pain points
    
    Returns:
        List of recommendation strings
    """
    bank_df = df[df['bank'] == bank_name].copy()
    recommendations = []
    
    # Analyze pain points
    pain_points = insights.get('pain_points', [])
    
    for pain_point in pain_points[:2]:  # Top 2 pain points
        theme = pain_point['theme']
        
        if theme == "Bugs & Reliability":
            recommendations.append(
                f"Prioritize bug fixes and stability improvements. "
                f"{pain_point['count']} negative reviews ({pain_point['percentage']:.1f}%) "
                f"mention reliability issues. Consider implementing automated testing "
                f"and a more robust QA process."
            )
        elif theme == "Transaction Performance":
            recommendations.append(
                f"Optimize transaction processing speed. "
                f"{pain_point['count']} reviews ({pain_point['percentage']:.1f}%) "
                f"complain about slow transfers. Investigate backend infrastructure "
                f"and consider implementing caching or load balancing."
            )
        elif theme == "Account Access Issues":
            recommendations.append(
                f"Improve authentication and login experience. "
                f"{pain_point['count']} reviews ({pain_point['percentage']:.1f}%) "
                f"report access problems. Consider enhancing biometric authentication "
                f"and password recovery processes."
            )
        elif theme == "User Interface & Experience":
            recommendations.append(
                f"Redesign UI/UX based on user feedback. "
                f"{pain_point['count']} reviews ({pain_point['percentage']:.1f}%) "
                f"mention interface issues. Conduct user testing sessions and "
                f"implement a more intuitive navigation structure."
            )
        elif theme == "Customer Support":
            recommendations.append(
                f"Enhance customer support responsiveness. "
                f"{pain_point['count']} reviews ({pain_point['percentage']:.1f}%) "
                f"mention support issues. Consider implementing AI chatbot integration "
                f"and reducing response times."
            )
    
    # Analyze drivers
    drivers = insights.get('drivers', [])
    
    if drivers:
        top_driver = drivers[0]['theme']
        recommendations.append(
            f"Leverage strength in {top_driver}. "
            f"{drivers[0]['count']} positive reviews highlight this area. "
            f"Continue investing in this feature and use it as a differentiator "
            f"in marketing campaigns."
        )
    
    # General recommendations based on rating
    avg_rating = insights.get('avg_rating', 0)
    if avg_rating < 3.5:
        recommendations.append(
            f"Overall rating is {avg_rating:.1f}/5.0, indicating significant "
            f"room for improvement. Focus on addressing the top pain points "
            f"identified above to improve user satisfaction."
        )
    
    return recommendations


def analyze_slow_loading_issue(df: pd.DataFrame) -> Dict:
    """
    Analyze slow loading/transfer issues across banks (Scenario 1).
    
    Args:
        df: DataFrame with analyzed reviews
    
    Returns:
        Dictionary with analysis results
    """
    # Search for reviews mentioning slow loading/transfer
    slow_keywords = ['slow', 'loading', 'transfer', 'timeout', 'delay', 'lag']
    
    slow_reviews = df[
        df['review'].str.lower().str.contains('|'.join(slow_keywords), na=False)
    ]
    
    analysis = {
        'total_slow_mentions': len(slow_reviews),
        'percentage_of_total': len(slow_reviews) / len(df) * 100 if len(df) > 0 else 0,
        'by_bank': {}
    }
    
    for bank in df['bank'].unique():
        bank_df = df[df['bank'] == bank]
        bank_slow = slow_reviews[slow_reviews['bank'] == bank]
        
        analysis['by_bank'][bank] = {
            'count': len(bank_slow),
            'percentage': len(bank_slow) / len(bank_df) * 100 if len(bank_df) > 0 else 0,
            'avg_rating': bank_slow['rating'].mean() if len(bank_slow) > 0 else 0,
            'avg_sentiment': bank_slow['sentiment_score'].mean() if len(bank_slow) > 0 else 0
        }
    
    return analysis


def extract_feature_requests(df: pd.DataFrame) -> Dict:
    """
    Extract desired features from reviews (Scenario 2).
    
    Args:
        df: DataFrame with analyzed reviews
    
    Returns:
        Dictionary with feature requests by bank
    """
    feature_keywords = {
        'transfer': ['transfer', 'send money', 'payment'],
        'biometric': ['fingerprint', 'face id', 'biometric', 'touch id'],
        'budgeting': ['budget', 'spending', 'expense tracking'],
        'notifications': ['notification', 'alert', 'reminder'],
        'bill_payment': ['bill', 'utility', 'electricity', 'water'],
        'investment': ['investment', 'savings', 'interest']
    }
    
    feature_requests = {}
    
    for bank in df['bank'].unique():
        bank_df = df[df['bank'] == bank]
        bank_features = {}
        
        for feature_name, keywords in feature_keywords.items():
            pattern = '|'.join(keywords)
            matches = bank_df[
                bank_df['review'].str.lower().str.contains(pattern, na=False)
            ]
            bank_features[feature_name] = {
                'count': len(matches),
                'percentage': len(matches) / len(bank_df) * 100 if len(bank_df) > 0 else 0
            }
        
        feature_requests[bank] = bank_features
    
    return feature_requests


def cluster_complaints(df: pd.DataFrame) -> Dict:
    """
    Cluster and track complaints for chatbot integration (Scenario 3).
    
    Args:
        df: DataFrame with analyzed reviews
    
    Returns:
        Dictionary with complaint clusters
    """
    # Focus on negative reviews
    negative_reviews = df[df['sentiment_label'] == 'negative'].copy()
    
    # Cluster by theme
    complaint_clusters = {}
    
    for theme in negative_reviews['theme'].unique():
        if pd.notna(theme):
            theme_reviews = negative_reviews[negative_reviews['theme'] == theme]
            
            complaint_clusters[theme] = {
                'count': len(theme_reviews),
                'avg_rating': theme_reviews['rating'].mean(),
                'by_bank': {}
            }
            
            for bank in theme_reviews['bank'].unique():
                bank_complaints = theme_reviews[theme_reviews['bank'] == bank]
                complaint_clusters[theme]['by_bank'][bank] = {
                    'count': len(bank_complaints),
                    'sample_reviews': bank_complaints['review'].head(3).tolist()
                }
    
    return complaint_clusters

