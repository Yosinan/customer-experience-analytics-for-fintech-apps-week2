"""
Thematic Analysis Module
Task 2: Keyword Extraction and Theme Clustering
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import spacy
from collections import Counter
from typing import List, Dict, Tuple
import re


class ThematicAnalyzer:
    """
    Extract keywords and identify themes from reviews.
    """
    
    def __init__(self):
        """Initialize thematic analyzer."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found.")
            print("Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text: lowercase, remove special chars, etc.
        
        Args:
            text: Raw text
        
        Returns:
            Preprocessed text
        """
        if not text or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z\s]', '', text)
        
        return text.strip()
    
    def extract_keywords_tfidf(
        self, 
        texts: List[str], 
        max_features: int = 50,
        ngram_range: Tuple[int, int] = (1, 2)
    ) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF.
        
        Args:
            texts: List of review texts
            max_features: Maximum number of features
            ngram_range: Range for n-grams
        
        Returns:
            List of (keyword, score) tuples
        """
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Remove empty texts
        processed_texts = [t for t in processed_texts if t]
        
        if not processed_texts:
            return []
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2  # Word must appear in at least 2 documents
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get mean TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Sort by score
            keyword_scores = list(zip(feature_names, mean_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return keyword_scores
        except Exception as e:
            print(f"Error in TF-IDF extraction: {e}")
            return []
    
    def extract_keywords_spacy(self, texts: List[str], top_n: int = 50) -> List[str]:
        """
        Extract keywords using spaCy (nouns, adjectives, verbs).
        
        Args:
            texts: List of review texts
            top_n: Number of top keywords to return
        
        Returns:
            List of keywords
        """
        if not self.nlp:
            return []
        
        keywords = []
        
        for text in texts:
            if not text or pd.isna(text):
                continue
            
            doc = self.nlp(text)
            for token in doc:
                # Extract nouns, adjectives, and verbs
                if (token.pos_ in ['NOUN', 'ADJ', 'VERB'] and 
                    not token.is_stop and 
                    not token.is_punct and
                    len(token.text) > 2):
                    keywords.append(token.lemma_.lower())
        
        # Count and return top keywords
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(top_n)]
    
    def identify_themes(
        self, 
        texts: List[str], 
        n_topics: int = 5,
        method: str = "lda"
    ) -> Dict[str, List[str]]:
        """
        Identify themes using topic modeling.
        
        Args:
            texts: List of review texts
            n_topics: Number of topics/themes to identify
            method: "lda" or "nmf"
        
        Returns:
            Dictionary mapping theme names to keywords
        """
        processed_texts = [self.preprocess_text(text) for text in texts]
        processed_texts = [t for t in processed_texts if t]
        
        if not processed_texts:
            return {}
        
        # Vectorize
        vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )
        
        try:
            doc_term_matrix = vectorizer.fit_transform(processed_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Topic modeling
            if method == "lda":
                model = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=10
                )
            else:  # NMF
                model = NMF(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=200
                )
            
            model.fit(doc_term_matrix)
            
            # Extract top words for each topic
            themes = {}
            for idx, topic in enumerate(model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                theme_name = f"Theme_{idx+1}"
                themes[theme_name] = top_words
            
            return themes
        except Exception as e:
            print(f"Error in theme identification: {e}")
            return {}
    
    def assign_theme_to_review(
        self, 
        review_text: str, 
        themes: Dict[str, List[str]]
    ) -> str:
        """
        Assign a theme to a review based on keyword matching.
        
        Args:
            review_text: Review text
            themes: Dictionary of themes and their keywords
        
        Returns:
            Assigned theme name
        """
        if not review_text or pd.isna(review_text):
            return "Unknown"
        
        review_lower = review_text.lower()
        theme_scores = {}
        
        for theme_name, keywords in themes.items():
            score = sum(1 for keyword in keywords if keyword in review_lower)
            theme_scores[theme_name] = score
        
        if theme_scores:
            return max(theme_scores, key=theme_scores.get)
        else:
            return "Unknown"


# Manual theme definitions based on common fintech app issues
MANUAL_THEMES = {
    "Account Access Issues": [
        "login", "password", "account", "access", "authentication",
        "biometric", "fingerprint", "face id", "unlock"
    ],
    "Transaction Performance": [
        "transfer", "transaction", "payment", "slow", "fast", "speed",
        "loading", "timeout", "delay", "process"
    ],
    "User Interface & Experience": [
        "ui", "interface", "design", "layout", "navigation", "menu",
        "button", "screen", "display", "user experience"
    ],
    "Customer Support": [
        "support", "help", "service", "contact", "response", "assistance",
        "chat", "email", "phone", "customer service"
    ],
    "Feature Requests": [
        "feature", "add", "missing", "need", "want", "request",
        "functionality", "option", "setting", "improvement"
    ],
    "Bugs & Reliability": [
        "crash", "error", "bug", "glitch", "freeze", "hang",
        "not working", "broken", "issue", "problem"
    ]
}


def assign_manual_themes(review_text: str) -> str:
    """
    Assign theme to review using manual keyword matching.
    
    Args:
        review_text: Review text
    
    Returns:
        Assigned theme name
    """
    if not review_text or pd.isna(review_text):
        return "Unknown"
    
    review_lower = review_text.lower()
    theme_scores = {}
    
    for theme_name, keywords in MANUAL_THEMES.items():
        score = sum(1 for keyword in keywords if keyword in review_lower)
        theme_scores[theme_name] = score
    
    if theme_scores and max(theme_scores.values()) > 0:
        return max(theme_scores, key=theme_scores.get)
    else:
        return "General Feedback"

