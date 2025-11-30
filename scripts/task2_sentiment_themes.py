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
    
    print("\n" + "=" * 60)
    print("Task 2 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

