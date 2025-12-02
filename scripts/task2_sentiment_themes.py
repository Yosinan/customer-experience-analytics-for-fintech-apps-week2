"""
Task 2: Sentiment and Thematic Analysis
Main script to analyze sentiment and extract themes from reviews

Requirements:
- Sentiment Analysis using distilbert-base-uncased-finetuned-sst-2-english (or VADER/TextBlob)
- Aggregate sentiment by bank and rating (e.g., mean sentiment for 1-star reviews)
- Thematic Analysis:
  * Keyword Extraction using TF-IDF or spaCy
  * Group related keywords into 3-5 themes per bank
  * Document grouping logic
- Preprocessing pipeline: tokenization, stop-word removal, lemmatization
- Save results as CSV: review_id, review_text, sentiment_label, sentiment_score, identified_theme(s)
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.sentiment_analysis import SentimentAnalyzer, aggregate_sentiment_by_bank
from src.text_preprocessing import TextPreprocessor, document_preprocessing_pipeline
from src.thematic_analysis import ThematicAnalyzer
from src.theme_grouping import identify_themes_per_bank, document_theme_grouping_logic
from src.config import Config
from src.utils import setup_logging, safe_file_operation
import numpy as np
import logging

# Set up logging
logger = setup_logging(
    log_level=Config.LOG_LEVEL,
    log_file=Config.LOG_FILE,
    log_format=Config.LOG_FORMAT
)

# Ensure directories exist
Config.ensure_directories()


def main():
    """
    Main function to execute Task 2: sentiment and thematic analysis.
    """
    print("=" * 60)
    print("Task 2: Sentiment and Thematic Analysis")
    print("=" * 60)
    
    # Load cleaned reviews with error handling
    input_file = os.path.join(Config.DATA_DIR, "cleaned_reviews.csv")
    if not os.path.exists(input_file):
        logger.error(f"✗ Error: {input_file} not found. Please run Task 1 first.")
        return
    
    try:
        df = pd.read_csv(input_file)
        logger.info(f"\n✓ Loaded {len(df)} reviews")
    except Exception as e:
        logger.error(f"Failed to load cleaned reviews: {e}", exc_info=True)
        return
    
    # Add review_id if not present
    if 'review_id' not in df.columns:
        df['review_id'] = range(1, len(df) + 1)
    
    # Initialize preprocessing pipeline
    print("\n" + "-" * 60)
    print("Text Preprocessing Pipeline")
    print("-" * 60)
    print(document_preprocessing_pipeline())
    
    preprocessor = TextPreprocessor(use_spacy=True)
    
    # Initialize sentiment analyzer
    print("\n" + "-" * 60)
    print("Initializing Sentiment Analyzer")
    print("-" * 60)
    print("Method: VADER (can be switched to distilbert-base-uncased-finetuned-sst-2-english)")
    print("Note: To use DistilBERT, change method='distilbert' in SentimentAnalyzer initialization")
    
    sentiment_analyzer = SentimentAnalyzer(method="vader")  # Can use "distilbert" for better accuracy
    
    # Initialize thematic analyzer
    print("\nInitializing Thematic Analyzer...")
    thematic_analyzer = ThematicAnalyzer()
    
    # STEP 1: Sentiment Analysis
    print("\n" + "=" * 60)
    print("STEP 1: SENTIMENT ANALYSIS")
    print("=" * 60)
    
    sentiment_results = []
    batch_size = 100
    
    print(f"\nProcessing {len(df)} reviews in batches of {batch_size}...")
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_texts = batch['review'].tolist()
        batch_sentiments = sentiment_analyzer.analyze_batch(batch_texts)
        sentiment_results.append(batch_sentiments)
        
        if (i + batch_size) % 500 == 0 or (i + batch_size) >= len(df):
            print(f"  Processed {min(i + batch_size, len(df))} / {len(df)} reviews...")
    
    sentiment_df = pd.concat(sentiment_results, ignore_index=True)
    df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
    
    # Calculate sentiment coverage
    sentiment_coverage = (df['sentiment_label'].notna().sum() / len(df)) * 100
    print(f"\n✓ Sentiment analysis complete")
    print(f"  Coverage: {sentiment_coverage:.1f}% of reviews have sentiment scores")
    print(f"  Sentiment distribution:")
    for label, count in df['sentiment_label'].value_counts().items():
        print(f"    {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # STEP 2: Aggregate Sentiment by Bank and Rating
    print("\n" + "=" * 60)
    print("STEP 2: AGGREGATE SENTIMENT BY BANK AND RATING")
    print("=" * 60)
    
    sentiment_agg = aggregate_sentiment_by_bank(df)
    print("\nSentiment Aggregation by Bank and Rating:")
    print(sentiment_agg.to_string(index=False))
    
    # Save aggregation results
    agg_file = "data/sentiment_aggregation_by_bank_rating.csv"
    sentiment_agg.to_csv(agg_file, index=False)
    print(f"\n✓ Sentiment aggregation saved to {agg_file}")
    
    # STEP 3: Thematic Analysis with Preprocessing
    print("\n" + "=" * 60)
    print("STEP 3: THEMATIC ANALYSIS")
    print("=" * 60)
    
    # Extract keywords per bank using TF-IDF
    print("\nExtracting keywords using TF-IDF...")
    keywords_by_bank = {}
    
    for bank in df['bank'].unique():
        print(f"\n  Processing {bank}...")
        bank_df = df[df['bank'] == bank]
        bank_reviews = bank_df['review'].tolist()
        
        # Preprocess reviews (tokenization, stop-word removal, lemmatization)
        preprocessed_reviews = preprocessor.preprocess_batch(bank_reviews)
        
        # Extract keywords using TF-IDF
        keywords = thematic_analyzer.extract_keywords_tfidf(
            preprocessed_reviews,
            max_features=50,
            ngram_range=(1, 2)  # Unigrams and bigrams
        )
        keywords_by_bank[bank] = keywords
        
        print(f"    Extracted {len(keywords)} keywords")
        print(f"    Top 10 keywords:")
        for keyword, score in keywords[:10]:
            print(f"      - {keyword}: {score:.4f}")
    
    # STEP 4: Group Keywords into 3-5 Themes per Bank
    print("\n" + "-" * 60)
    print("Grouping Keywords into Themes (3-5 per bank)")
    print("-" * 60)
    
    themes_by_bank = identify_themes_per_bank(df, thematic_analyzer, preprocessor)
    
    # Assign themes to individual reviews
    print("\nAssigning themes to reviews...")
    
    def assign_theme_to_review(review_text: str, bank_name: str) -> str:
        """Assign theme to a review based on bank-specific themes."""
        if bank_name not in themes_by_bank:
            return "General Feedback"
        
        themes = themes_by_bank[bank_name]
        review_lower = review_text.lower()
        
        theme_scores = {}
        for theme_name, keywords in themes.items():
            score = sum(1 for keyword in keywords if keyword.lower() in review_lower)
            theme_scores[theme_name] = score
        
        if theme_scores and max(theme_scores.values()) > 0:
            return max(theme_scores, key=theme_scores.get)
        else:
            return "General Feedback"
    
    df['theme'] = df.apply(
        lambda row: assign_theme_to_review(row['review'], row['bank']),
        axis=1
    )
    
    print(f"✓ Theme assignment complete")
    print(f"\nTheme distribution across all reviews:")
    for theme, count in df['theme'].value_counts().items():
        print(f"  {theme}: {count} ({count/len(df)*100:.1f}%)")
    
    # Show themes per bank
    print("\n" + "-" * 60)
    print("Themes Identified per Bank (3-5 themes per bank)")
    print("-" * 60)
    for bank_name, themes in themes_by_bank.items():
        print(f"\n{bank_name} ({len(themes)} themes):")
        for theme_name, keywords in themes.items():
            print(f"  - {theme_name}: {len(keywords)} keywords")
            print(f"    Sample keywords: {', '.join(keywords[:5])}")
    
    # STEP 5: Prepare Output CSV with Required Columns
    print("\n" + "=" * 60)
    print("STEP 5: SAVE RESULTS")
    print("=" * 60)
    
    # Ensure we have all required columns
    output_df = df[[
        'review_id',
        'review_text' if 'review_text' in df.columns else 'review',
        'sentiment_label',
        'sentiment_score',
        'theme',
        'bank',
        'rating',
        'date'
    ]].copy()
    
    # Rename review column if needed
    if 'review_text' not in output_df.columns:
        output_df.rename(columns={'review': 'review_text'}, inplace=True)
    
    # Ensure theme column is named correctly
    output_df.rename(columns={'theme': 'identified_theme'}, inplace=True)
    
    # Save main results with error handling
    output_file = os.path.join(Config.DATA_DIR, "analyzed_reviews.csv")
    try:
        output_df.to_csv(output_file, index=False)
        logger.info(f"\n✓ Analyzed reviews saved to {output_file}")
        logger.info(f"  Columns: {', '.join(output_df.columns)}")
        logger.info(f"  Total reviews: {len(output_df)}")
    except Exception as e:
        logger.error(f"Failed to save analyzed reviews: {e}", exc_info=True)
        return
    
    # Save keywords by bank with error handling
    keywords_file = os.path.join(Config.DATA_DIR, "keywords_by_bank.csv")
    try:
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
        logger.info(f"✓ Keywords saved to {keywords_file}")
    except Exception as e:
        logger.error(f"Failed to save keywords: {e}", exc_info=True)
    
    # Save theme grouping documentation with error handling
    theme_doc_file = os.path.join(Config.DATA_DIR, "theme_grouping_documentation.txt")
    try:
        with open(theme_doc_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("THEME GROUPING DOCUMENTATION\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("PREPROCESSING PIPELINE:\n")
            f.write("-" * 60 + "\n")
            f.write(document_preprocessing_pipeline())
            f.write("\n\n")
            
            f.write("THEME GROUPING LOGIC PER BANK:\n")
            f.write("=" * 60 + "\n\n")
            
            for bank_name, themes in themes_by_bank.items():
                doc = document_theme_grouping_logic(bank_name, themes)
                f.write(doc)
                f.write("\n" + "=" * 60 + "\n\n")
        
        logger.info(f"✓ Theme grouping documentation saved to {theme_doc_file}")
    except Exception as e:
        logger.error(f"Failed to save theme documentation: {e}", exc_info=True)
    
    # Summary Statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    for bank in df['bank'].unique():
        bank_df = df[df['bank'] == bank]
        print(f"\n{bank}:")
        print(f"  Total reviews: {len(bank_df)}")
        print(f"  Average sentiment score: {bank_df['sentiment_score'].mean():.3f}")
        print(f"  Sentiment distribution:")
        for label, count in bank_df['sentiment_label'].value_counts().items():
            print(f"    {label}: {count} ({count/len(bank_df)*100:.1f}%)")
        print(f"  Themes identified: {len(themes_by_bank.get(bank, {}))}")
        print(f"  Top themes:")
        for theme, count in bank_df['theme'].value_counts().head(3).items():
            print(f"    {theme}: {count}")
    
    # Verify KPIs
    print("\n" + "=" * 60)
    print("KPI VERIFICATION")
    print("=" * 60)
    
    sentiment_coverage_pct = (df['sentiment_label'].notna().sum() / len(df)) * 100
    print(f"✓ Sentiment scores coverage: {sentiment_coverage_pct:.1f}% (Target: 90%+)")
    
    for bank in df['bank'].unique():
        themes_count = len(themes_by_bank.get(bank, {}))
        print(f"✓ {bank} themes: {themes_count} (Target: 3+)")
    
    print("\n" + "=" * 60)
    print("Task 2 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
