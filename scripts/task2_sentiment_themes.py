"""
Task 2: Sentiment and Thematic Analysis
Main script to analyze sentiment and extract themes from reviews
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.sentiment_analysis import SentimentAnalyzer, aggregate_sentiment_by_bank
from src.thematic_analysis import ThematicAnalyzer, assign_manual_themes
from src.sentiment_comparison import (
    analyze_sentiment_by_rating,
    analyze_sentiment_by_theme,
    identify_sentiment_drivers,
    calculate_sentiment_consistency
)
from src.theme_clustering import (
    analyze_theme_sentiment_correlation,
    identify_theme_trends
)
import numpy as np


def main():
    """
    Main function to execute Task 2: sentiment and thematic analysis.
    """
    print("=" * 60)
    print("Task 2: Sentiment and Thematic Analysis")
    print("=" * 60)
    
    # Load cleaned reviews
    input_file = "data/cleaned_reviews.csv"
    if not os.path.exists(input_file):
        print(f"✗ Error: {input_file} not found. Please run Task 1 first.")
        return
    
    df = pd.read_csv(input_file)
    print(f"\n✓ Loaded {len(df)} reviews")
    
    # Initialize analyzers
    print("\nInitializing sentiment analyzer...")
    sentiment_analyzer = SentimentAnalyzer(method="vader")  # Start with VADER (faster)
    
    print("\nInitializing thematic analyzer...")
    thematic_analyzer = ThematicAnalyzer()
    
    # Sentiment Analysis
    print("\n" + "-" * 60)
    print("Performing sentiment analysis...")
    print("-" * 60)
    
    sentiment_results = []
    batch_size = 100
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_texts = batch['review'].tolist()
        batch_sentiments = sentiment_analyzer.analyze_batch(batch_texts)
        sentiment_results.append(batch_sentiments)
        
        if (i + batch_size) % 500 == 0:
            print(f"  Processed {min(i + batch_size, len(df))} / {len(df)} reviews...")
    
    sentiment_df = pd.concat(sentiment_results, ignore_index=True)
    df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
    
    print(f"✓ Sentiment analysis complete")
    print(f"  Sentiment distribution:\n{df['sentiment_label'].value_counts()}")
    
    # Thematic Analysis
    print("\n" + "-" * 60)
    print("Performing thematic analysis...")
    print("-" * 60)
    
    # Extract keywords per bank
    print("\nExtracting keywords...")
    keywords_by_bank = {}
    for bank in df['bank'].unique():
        bank_reviews = df[df['bank'] == bank]['review'].tolist()
        keywords = thematic_analyzer.extract_keywords_tfidf(bank_reviews, max_features=30)
        keywords_by_bank[bank] = keywords[:20]  # Top 20
        print(f"\n  {bank} - Top keywords:")
        for keyword, score in keywords[:10]:
            print(f"    - {keyword}: {score:.4f}")
    
    # Assign themes using manual matching
    print("\nAssigning themes to reviews...")
    df['theme'] = df['review'].apply(assign_manual_themes)
    
    print(f"✓ Theme assignment complete")
    print(f"  Theme distribution:\n{df['theme'].value_counts()}")
    
    # Aggregate sentiment by bank and rating
    print("\n" + "-" * 60)
    print("Aggregating sentiment by bank and rating...")
    print("-" * 60)
    
    sentiment_agg = aggregate_sentiment_by_bank(df)
    print(sentiment_agg)
    
    # Save results
    output_file = "data/analyzed_reviews.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Analyzed reviews saved to {output_file}")
    
    # Save keywords
    keywords_file = "data/keywords_by_bank.csv"
    keywords_data = []
    for bank, keywords in keywords_by_bank.items():
        for keyword, score in keywords:
            keywords_data.append({
                'bank': bank,
                'keyword': keyword,
                'tfidf_score': score
            })
    keywords_df = pd.DataFrame(keywords_data)
    keywords_df.to_csv(keywords_file, index=False)
    print(f"✓ Keywords saved to {keywords_file}")
    
    # Advanced Analysis
    print("\n" + "-" * 60)
    print("Advanced Sentiment Analysis")
    print("-" * 60)
    
    # Analyze sentiment by rating
    sentiment_by_rating = analyze_sentiment_by_rating(df)
    print("\nSentiment Analysis by Rating:")
    print(sentiment_by_rating.to_string(index=False))
    
    # Analyze sentiment by theme
    if 'theme' in df.columns:
        sentiment_by_theme = analyze_sentiment_by_theme(df)
        print("\nSentiment Analysis by Theme:")
        print(sentiment_by_theme.to_string(index=False))
        
        # Theme-sentiment correlation
        theme_correlation = analyze_theme_sentiment_correlation(df)
        print("\nTheme-Sentiment Correlation:")
        print(theme_correlation.to_string(index=False))
    
    # Sentiment consistency metrics
    consistency = calculate_sentiment_consistency(df)
    print("\nSentiment Consistency Metrics:")
    for key, value in consistency.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Identify sentiment drivers for each bank
    print("\n" + "-" * 60)
    print("Sentiment Drivers Analysis")
    print("-" * 60)
    
    for bank in df['bank'].unique():
        drivers = identify_sentiment_drivers(df, bank_name=bank)
        print(f"\n{bank}:")
        if drivers['insights']:
            for insight in drivers['insights']:
                print(f"  - {insight}")
        if 'positive_drivers' in drivers and 'count' in drivers['positive_drivers']:
            print(f"  Positive reviews: {drivers['positive_drivers']['count']}")
        if 'negative_drivers' in drivers and 'count' in drivers['negative_drivers']:
            print(f"  Negative reviews: {drivers['negative_drivers']['count']}")
    
    # Theme trends (if dates are available)
    if 'date' in df.columns:
        print("\n" + "-" * 60)
        print("Theme Trends Analysis")
        print("-" * 60)
        try:
            theme_trends = identify_theme_trends(df)
            if theme_trends:
                print("\nTheme Trends Over Time:")
                for theme, trend_data in theme_trends.items():
                    print(f"  {theme}: {trend_data['trend']} "
                          f"(recent: {trend_data['recent_avg']:.1f}, "
                          f"earlier: {trend_data['earlier_avg']:.1f})")
        except Exception as e:
            print(f"  Could not analyze trends: {e}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    for bank in df['bank'].unique():
        bank_df = df[df['bank'] == bank]
        print(f"\n{bank}:")
        print(f"  Total reviews: {len(bank_df)}")
        print(f"  Average sentiment score: {bank_df['sentiment_score'].mean():.3f}")
        print(f"  Sentiment distribution:")
        for label, count in bank_df['sentiment_label'].value_counts().items():
            print(f"    {label}: {count} ({count/len(bank_df)*100:.1f}%)")
        print(f"  Top themes:")
        for theme, count in bank_df['theme'].value_counts().head(3).items():
            print(f"    {theme}: {count}")
    
    # Save additional analysis results
    os.makedirs("data", exist_ok=True)
    
    if not sentiment_by_rating.empty:
        sentiment_by_rating.to_csv("data/sentiment_by_rating.csv", index=False)
        print(f"\n✓ Sentiment by rating saved to data/sentiment_by_rating.csv")
    
    if 'theme' in df.columns and not sentiment_by_theme.empty:
        sentiment_by_theme.to_csv("data/sentiment_by_theme.csv", index=False)
        print(f"✓ Sentiment by theme saved to data/sentiment_by_theme.csv")
    
    if 'theme' in df.columns and not theme_correlation.empty:
        theme_correlation.to_csv("data/theme_sentiment_correlation.csv", index=False)
        print(f"✓ Theme-sentiment correlation saved to data/theme_sentiment_correlation.csv")
    
    print("\n" + "=" * 60)
    print("Task 2 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

