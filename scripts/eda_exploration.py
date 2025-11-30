"""
Exploratory Data Analysis Script
Use this as a template for exploring the data
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def main():
    """Run exploratory data analysis."""
    # Load analyzed reviews
    input_file = "data/analyzed_reviews.csv"
    if not os.path.exists(input_file):
        print(f"✗ Error: {input_file} not found. Please run Task 2 first.")
        return
    
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df)} analyzed reviews")
    
    # Basic statistics
    print("\n" + "=" * 60)
    print("Basic Statistics")
    print("=" * 60)
    print(f"\nTotal reviews: {len(df)}")
    print(f"\nReviews per bank:\n{df['bank'].value_counts()}")
    print(f"\nRating distribution:\n{df['rating'].value_counts().sort_index()}")
    print(f"\nSentiment distribution:\n{df['sentiment_label'].value_counts()}")
    print(f"\nTheme distribution:\n{df['theme'].value_counts()}")
    
    # Missing data
    print(f"\nMissing data:\n{df.isnull().sum()}")
    
    # Average metrics by bank
    print("\n" + "=" * 60)
    print("Average Metrics by Bank")
    print("=" * 60)
    metrics = df.groupby('bank').agg({
        'rating': 'mean',
        'sentiment_score': 'mean'
    }).round(3)
    print(metrics)
    
    # Sample reviews
    print("\n" + "=" * 60)
    print("Sample Positive Reviews")
    print("=" * 60)
    positive = df[df['sentiment_label'] == 'positive'].head(3)
    for idx, row in positive.iterrows():
        print(f"\n[{row['bank']}] Rating: {row['rating']}/5")
        print(f"{row['review'][:150]}...")
    
    print("\n" + "=" * 60)
    print("Sample Negative Reviews")
    print("=" * 60)
    negative = df[df['sentiment_label'] == 'negative'].head(3)
    for idx, row in negative.iterrows():
        print(f"\n[{row['bank']}] Rating: {row['rating']}/5")
        print(f"{row['review'][:150]}...")
    
    print("\n✓ EDA complete!")


if __name__ == "__main__":
    main()

