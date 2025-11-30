"""
Visualization Module
Task 4: Create visualizations for insights
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from typing import Optional
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_sentiment_distribution(df: pd.DataFrame, output_path: Optional[str] = None):
    """
    Plot sentiment distribution by bank.
    
    Args:
        df: DataFrame with sentiment analysis
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sentiment distribution
    sentiment_counts = df.groupby(['bank', 'sentiment_label']).size().unstack(fill_value=0)
    sentiment_counts.plot(kind='bar', ax=axes[0], color=['#ff4444', '#ffaa00', '#44ff44'])
    axes[0].set_title('Sentiment Distribution by Bank', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Bank', fontsize=12)
    axes[0].set_ylabel('Number of Reviews', fontsize=12)
    axes[0].legend(title='Sentiment', title_fontsize=10)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Sentiment scores
    df.boxplot(column='sentiment_score', by='bank', ax=axes[1])
    axes[1].set_title('Sentiment Score Distribution by Bank', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Bank', fontsize=12)
    axes[1].set_ylabel('Sentiment Score', fontsize=12)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
    plt.suptitle('')  # Remove default title
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved sentiment distribution plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_rating_distribution(df: pd.DataFrame, output_path: Optional[str] = None):
    """
    Plot rating distribution by bank.
    
    Args:
        df: DataFrame with ratings
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Rating distribution
    rating_counts = df.groupby(['bank', 'rating']).size().unstack(fill_value=0)
    rating_counts.plot(kind='bar', ax=axes[0], color=['#ff0000', '#ff6600', '#ffaa00', '#aaff00', '#00ff00'])
    axes[0].set_title('Rating Distribution by Bank', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Bank', fontsize=12)
    axes[0].set_ylabel('Number of Reviews', fontsize=12)
    axes[0].legend(title='Rating', title_fontsize=10, labels=['1★', '2★', '3★', '4★', '5★'])
    axes[0].tick_params(axis='x', rotation=45)
    
    # Average rating comparison
    avg_ratings = df.groupby('bank')['rating'].mean().sort_values(ascending=False)
    avg_ratings.plot(kind='barh', ax=axes[1], color='steelblue')
    axes[1].set_title('Average Rating by Bank', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Average Rating', fontsize=12)
    axes[1].set_ylabel('Bank', fontsize=12)
    axes[1].axvline(x=3.0, color='red', linestyle='--', alpha=0.5, label='Threshold (3.0)')
    axes[1].legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved rating distribution plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_theme_distribution(df: pd.DataFrame, output_path: Optional[str] = None):
    """
    Plot theme distribution by bank.
    
    Args:
        df: DataFrame with themes
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    theme_counts = df.groupby(['bank', 'theme']).size().unstack(fill_value=0)
    theme_counts.plot(kind='barh', ax=ax, stacked=False)
    ax.set_title('Theme Distribution by Bank', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Reviews', fontsize=12)
    ax.set_ylabel('Bank', fontsize=12)
    ax.legend(title='Theme', title_fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved theme distribution plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_sentiment_trends(df: pd.DataFrame, output_path: Optional[str] = None):
    """
    Plot sentiment trends over time.
    
    Args:
        df: DataFrame with dates and sentiment
        output_path: Path to save figure
    """
    # Convert date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    if df.empty:
        print("⚠ No valid dates found for trend analysis")
        return
    
    # Group by month and bank
    df['year_month'] = df['date'].dt.to_period('M')
    monthly_sentiment = df.groupby(['year_month', 'bank'])['sentiment_score'].mean().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for bank in monthly_sentiment.columns:
        ax.plot(monthly_sentiment.index.astype(str), monthly_sentiment[bank], 
                marker='o', label=bank, linewidth=2)
    
    ax.set_title('Sentiment Trends Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Average Sentiment Score', fontsize=12)
    ax.legend(title='Bank', title_fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved sentiment trends plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_wordcloud(df: pd.DataFrame, bank_name: str, output_path: Optional[str] = None):
    """
    Create word cloud for a specific bank.
    
    Args:
        df: DataFrame with reviews
        bank_name: Name of the bank
        output_path: Path to save figure
    """
    bank_df = df[df['bank'] == bank_name]
    
    if bank_df.empty:
        print(f"⚠ No reviews found for {bank_name}")
        return
    
    # Combine all reviews
    text = ' '.join(bank_df['review'].astype(str).tolist())
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud: {bank_name}', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved word cloud to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_bank_comparison(df: pd.DataFrame, output_path: Optional[str] = None):
    """
    Create comprehensive bank comparison visualization.
    
    Args:
        df: DataFrame with all analysis
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Average ratings
    avg_ratings = df.groupby('bank')['rating'].mean().sort_values(ascending=False)
    axes[0, 0].barh(avg_ratings.index, avg_ratings.values, color='steelblue')
    axes[0, 0].set_title('Average Rating by Bank', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Average Rating')
    axes[0, 0].axvline(x=3.0, color='red', linestyle='--', alpha=0.5)
    
    # 2. Sentiment scores
    avg_sentiment = df.groupby('bank')['sentiment_score'].mean().sort_values(ascending=False)
    axes[0, 1].barh(avg_sentiment.index, avg_sentiment.values, color='green')
    axes[0, 1].set_title('Average Sentiment Score by Bank', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Average Sentiment Score')
    
    # 3. Review counts
    review_counts = df['bank'].value_counts()
    axes[1, 0].bar(review_counts.index, review_counts.values, color='orange')
    axes[1, 0].set_title('Total Reviews by Bank', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Number of Reviews')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Positive vs Negative percentage
    sentiment_pct = df.groupby('bank')['sentiment_label'].apply(
        lambda x: pd.Series({
            'positive': (x == 'positive').sum() / len(x) * 100,
            'negative': (x == 'negative').sum() / len(x) * 100
        })
    ).unstack()
    sentiment_pct.plot(kind='bar', ax=axes[1, 1], color=['green', 'red'])
    axes[1, 1].set_title('Sentiment Percentage by Bank', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].set_xlabel('Bank')
    axes[1, 1].legend(title='Sentiment')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Bank Comparison Dashboard', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved bank comparison plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_all_visualizations(df: pd.DataFrame, output_dir: str = "figures"):
    """
    Create all visualizations and save to output directory.
    
    Args:
        df: DataFrame with analyzed reviews
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating visualizations...")
    
    plot_sentiment_distribution(df, os.path.join(output_dir, "sentiment_distribution.png"))
    plot_rating_distribution(df, os.path.join(output_dir, "rating_distribution.png"))
    plot_theme_distribution(df, os.path.join(output_dir, "theme_distribution.png"))
    plot_bank_comparison(df, os.path.join(output_dir, "bank_comparison.png"))
    
    # Try sentiment trends (may fail if dates are invalid)
    try:
        plot_sentiment_trends(df, os.path.join(output_dir, "sentiment_trends.png"))
    except Exception as e:
        print(f"⚠ Could not create sentiment trends: {e}")
    
    # Create word clouds for each bank
    for bank in df['bank'].unique():
        create_wordcloud(df, bank, os.path.join(output_dir, f"wordcloud_{bank.replace(' ', '_')}.png"))
    
    print(f"\n✓ All visualizations saved to {output_dir}/")

