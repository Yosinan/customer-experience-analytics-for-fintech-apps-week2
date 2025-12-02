"""
Task 4: Insights and Recommendations
Main script to generate insights, visualizations, and recommendations
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.insights import (
    identify_drivers_and_pain_points,
    compare_banks,
    generate_recommendations,
    analyze_slow_loading_issue,
    extract_feature_requests,
    cluster_complaints
)
from src.visualizations import create_all_visualizations
from src.config import Config
from src.utils import setup_logging
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
    Main function to execute Task 4: insights and visualizations.
    """
    print("=" * 60)
    print("Task 4: Insights and Recommendations")
    print("=" * 60)
    
    # Load analyzed reviews with error handling
    input_file = os.path.join(Config.DATA_DIR, "analyzed_reviews.csv")
    if not os.path.exists(input_file):
        logger.error(f"✗ Error: {input_file} not found. Please run Task 2 first.")
        return
    
    try:
        df = pd.read_csv(input_file)
        logger.info(f"\n✓ Loaded {len(df)} analyzed reviews")
    except Exception as e:
        logger.error(f"Failed to load analyzed reviews: {e}", exc_info=True)
        return
    
    # Generate visualizations with error handling
    logger.info("\n" + "-" * 60)
    logger.info("Generating visualizations...")
    logger.info("-" * 60)
    try:
        create_all_visualizations(df, output_dir=Config.FIGURES_DIR)
    except Exception as e:
        logger.error(f"Failed to create visualizations: {e}", exc_info=True)
        return
    
    # Bank comparison
    print("\n" + "-" * 60)
    print("Bank Comparison")
    print("-" * 60)
    comparison = compare_banks(df)
    print(comparison.to_string(index=False))
    
    # Insights for each bank
    print("\n" + "=" * 60)
    print("Insights by Bank")
    print("=" * 60)
    
    all_insights = {}
    all_recommendations = {}
    
    for bank in df['bank'].unique():
        print(f"\n{bank}:")
        print("-" * 60)
        
        insights = identify_drivers_and_pain_points(df, bank)
        all_insights[bank] = insights
        
        print(f"  Total Reviews: {insights['total_reviews']}")
        print(f"  Average Rating: {insights['avg_rating']:.2f}/5.0")
        print(f"  Average Sentiment: {insights['avg_sentiment']:.3f}")
        
        print(f"\n  Satisfaction Drivers:")
        for driver in insights['drivers']:
            print(f"    - {driver['theme']}: {driver['count']} reviews ({driver['percentage']:.1f}%)")
        
        print(f"\n  Pain Points:")
        for pain_point in insights['pain_points']:
            print(f"    - {pain_point['theme']}: {pain_point['count']} reviews ({pain_point['percentage']:.1f}%)")
        
        # Generate recommendations
        recommendations = generate_recommendations(df, bank, insights)
        all_recommendations[bank] = recommendations
        
        print(f"\n  Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"    {i}. {rec}")
    
    # Scenario 1: Slow loading analysis
    print("\n" + "=" * 60)
    print("Scenario 1: Slow Loading/Transfer Analysis")
    print("=" * 60)
    slow_analysis = analyze_slow_loading_issue(df)
    print(f"Total reviews mentioning slow loading/transfer: {slow_analysis['total_slow_mentions']}")
    print(f"Percentage of total reviews: {slow_analysis['percentage_of_total']:.2f}%")
    print("\nBy Bank:")
    for bank, data in slow_analysis['by_bank'].items():
        print(f"  {bank}:")
        print(f"    Mentions: {data['count']} ({data['percentage']:.1f}% of bank reviews)")
        print(f"    Avg Rating (slow mentions): {data['avg_rating']:.2f}")
        print(f"    Avg Sentiment: {data['avg_sentiment']:.3f}")
    
    # Scenario 2: Feature requests
    print("\n" + "=" * 60)
    print("Scenario 2: Feature Requests Analysis")
    print("=" * 60)
    feature_requests = extract_feature_requests(df)
    for bank, features in feature_requests.items():
        print(f"\n{bank}:")
        for feature, data in sorted(features.items(), key=lambda x: x[1]['count'], reverse=True):
            if data['count'] > 0:
                print(f"  {feature}: {data['count']} mentions ({data['percentage']:.1f}%)")
    
    # Scenario 3: Complaint clustering
    print("\n" + "=" * 60)
    print("Scenario 3: Complaint Clustering for Chatbot")
    print("=" * 60)
    complaint_clusters = cluster_complaints(df)
    for theme, cluster_data in complaint_clusters.items():
        print(f"\n{theme}:")
        print(f"  Total complaints: {cluster_data['count']}")
        print(f"  Average rating: {cluster_data['avg_rating']:.2f}")
        print(f"  By Bank:")
        for bank, bank_data in cluster_data['by_bank'].items():
            print(f"    {bank}: {bank_data['count']} complaints")
            if bank_data['sample_reviews']:
                print(f"      Sample: \"{bank_data['sample_reviews'][0][:100]}...\"")
    
    # Save insights to file with error handling
    logger.info("\n" + "-" * 60)
    logger.info("Saving insights...")
    logger.info("-" * 60)
    
    insights_file = os.path.join(Config.SUMMARY_DIR, "insights_summary.txt")
    
    try:
        with open(insights_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("INSIGHTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("BANK COMPARISON\n")
        f.write("-" * 60 + "\n")
        f.write(comparison.to_string(index=False))
        f.write("\n\n")
        
        for bank in df['bank'].unique():
            f.write(f"\n{bank}\n")
            f.write("=" * 60 + "\n")
            insights = all_insights[bank]
            f.write(f"Total Reviews: {insights['total_reviews']}\n")
            f.write(f"Average Rating: {insights['avg_rating']:.2f}/5.0\n")
            f.write(f"Average Sentiment: {insights['avg_sentiment']:.3f}\n\n")
            
            f.write("Satisfaction Drivers:\n")
            for driver in insights['drivers']:
                f.write(f"  - {driver['theme']}: {driver['count']} reviews ({driver['percentage']:.1f}%)\n")
            
            f.write("\nPain Points:\n")
            for pain_point in insights['pain_points']:
                f.write(f"  - {pain_point['theme']}: {pain_point['count']} reviews ({pain_point['percentage']:.1f}%)\n")
            
            f.write("\nRecommendations:\n")
            for i, rec in enumerate(all_recommendations[bank], 1):
                f.write(f"  {i}. {rec}\n")
            f.write("\n")
        logger.info(f"✓ Insights saved to {insights_file}")
    except Exception as e:
        logger.error(f"Failed to save insights: {e}", exc_info=True)
    
    print("\n" + "=" * 60)
    print("Task 4 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

