"""
Theme Grouping Logic for Task 2
Groups keywords into 3-5 themes per bank with documented logic
"""

import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter
from src.text_preprocessing import TextPreprocessor
from src.thematic_analysis import ThematicAnalyzer


def group_keywords_into_themes(
    keywords: List[Tuple[str, float]],
    bank_name: str
) -> Dict[str, List[str]]:
    """
    Group extracted keywords into 3-5 themes per bank.
    
    Args:
        keywords: List of (keyword, score) tuples from TF-IDF
        bank_name: Name of the bank
    
    Returns:
        Dictionary mapping theme names to lists of keywords
    """
    # Theme categories with keyword patterns
    theme_patterns = {
        "Account Access & Authentication": [
            "login", "password", "account", "access", "authentication",
            "biometric", "fingerprint", "face", "unlock", "verify",
            "security", "pin", "credential"
        ],
        "Transaction & Payment Performance": [
            "transfer", "transaction", "payment", "send", "money",
            "slow", "fast", "speed", "loading", "timeout", "delay",
            "process", "complete", "success", "fail"
        ],
        "User Interface & Design": [
            "ui", "interface", "design", "layout", "navigation",
            "menu", "button", "screen", "display", "page", "view",
            "user experience", "ux", "look", "appearance"
        ],
        "Customer Support & Service": [
            "support", "help", "service", "contact", "response",
            "assistance", "chat", "email", "phone", "customer service",
            "reply", "answer", "resolve"
        ],
        "App Stability & Bugs": [
            "crash", "error", "bug", "glitch", "freeze", "hang",
            "not working", "broken", "issue", "problem", "fail",
            "close", "stop", "restart"
        ],
        "Features & Functionality": [
            "feature", "add", "missing", "need", "want", "request",
            "functionality", "option", "setting", "improvement",
            "enhancement", "update", "new"
        ]
    }
    
    # Group keywords by theme
    themes = {theme: [] for theme in theme_patterns.keys()}
    unassigned = []
    
    for keyword, score in keywords:
        keyword_lower = keyword.lower()
        assigned = False
        
        # Check which theme this keyword belongs to
        for theme_name, patterns in theme_patterns.items():
            if any(pattern in keyword_lower for pattern in patterns):
                themes[theme_name].append(keyword)
                assigned = True
                break
        
        if not assigned:
            unassigned.append(keyword)
    
    # Filter out empty themes and keep only top 3-5 themes with most keywords
    themes = {k: v for k, v in themes.items() if len(v) > 0}
    
    # Sort by number of keywords and take top 3-5
    sorted_themes = sorted(themes.items(), key=lambda x: len(x[1]), reverse=True)
    top_themes = dict(sorted_themes[:5])  # Maximum 5 themes
    
    # If we have unassigned keywords, add them to the most relevant theme
    if unassigned and top_themes:
        # Add unassigned to the theme with most keywords
        top_theme_name = max(top_themes.keys(), key=lambda k: len(top_themes[k]))
        top_themes[top_theme_name].extend(unassigned[:5])  # Limit unassigned
    
    return top_themes


def document_theme_grouping_logic(bank_name: str, themes: Dict[str, List[str]]) -> str:
    """
    Document the theme grouping logic for a bank.
    
    Args:
        bank_name: Name of the bank
        themes: Dictionary of themes and keywords
    
    Returns:
        Documentation string
    """
    doc = f"""
    THEME GROUPING LOGIC FOR {bank_name.upper()}
    ===========================================
    
    Theme Identification Method:
    - Keywords were extracted using TF-IDF (Term Frequency-Inverse Document Frequency)
    - Keywords were grouped into themes based on semantic similarity and domain knowledge
    - Each theme represents a category of user feedback
    
    Theme Categories Identified ({len(themes)} themes):
    """
    
    for i, (theme_name, keywords) in enumerate(themes.items(), 1):
        doc += f"""
    {i}. {theme_name}
       Keywords ({len(keywords)}): {', '.join(keywords[:10])}
       {f'... and {len(keywords) - 10} more' if len(keywords) > 10 else ''}
    """
    
    doc += """
    
    Grouping Rules:
    1. Keywords are matched against predefined theme patterns
    2. Each keyword is assigned to the most relevant theme
    3. Themes are ranked by number of associated keywords
    4. Top 3-5 themes (by keyword count) are selected per bank
    5. Unassigned keywords are added to the most relevant theme
    
    Theme Definitions:
    - Account Access & Authentication: Login, security, biometric features
    - Transaction & Payment Performance: Money transfers, payment speed, processing
    - User Interface & Design: App appearance, navigation, layout
    - Customer Support & Service: Help, support, customer service
    - App Stability & Bugs: Crashes, errors, technical issues
    - Features & Functionality: Feature requests, missing features, improvements
    """
    
    return doc


def identify_themes_per_bank(
    df: pd.DataFrame,
    thematic_analyzer: ThematicAnalyzer,
    preprocessor: TextPreprocessor
) -> Dict[str, Dict[str, List[str]]]:
    """
    Identify 3-5 themes per bank with documented grouping logic.
    
    Args:
        df: DataFrame with reviews
        thematic_analyzer: ThematicAnalyzer instance
        preprocessor: TextPreprocessor instance
    
    Returns:
        Dictionary mapping bank_name to themes dictionary
    """
    themes_by_bank = {}
    
    for bank_name in df['bank'].unique():
        bank_df = df[df['bank'] == bank_name]
        bank_reviews = bank_df['review'].tolist()
        
        # Preprocess reviews
        preprocessed_reviews = preprocessor.preprocess_batch(bank_reviews)
        
        # Extract keywords using TF-IDF
        keywords = thematic_analyzer.extract_keywords_tfidf(
            preprocessed_reviews,
            max_features=50,
            ngram_range=(1, 2)
        )
        
        # Group keywords into themes
        themes = group_keywords_into_themes(keywords, bank_name)
        themes_by_bank[bank_name] = themes
        
        # Document grouping logic
        doc = document_theme_grouping_logic(bank_name, themes)
        print(doc)
    
    return themes_by_bank


if __name__ == "__main__":
    # Example usage
    sample_keywords = [
        ("login error", 0.85),
        ("slow transfer", 0.82),
        ("crash", 0.78),
        ("good ui", 0.75),
        ("support", 0.70)
    ]
    
    themes = group_keywords_into_themes(sample_keywords, "Test Bank")
    print("Themes:", themes)

