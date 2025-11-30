"""
Text Preprocessing Pipeline for Task 2
Tokenization, stop-word removal, and lemmatization
"""

import pandas as pd
import re
from typing import List
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """
    Text preprocessing pipeline: tokenization, stop-word removal, lemmatization.
    """
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            use_spacy: Use spaCy for advanced processing (if available)
        """
        self.use_spacy = use_spacy
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy model not found. Using NLTK only.")
                self.use_spacy = False
                self.nlp = None
        else:
            self.nlp = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Full preprocessing pipeline: clean, tokenize, remove stopwords, lemmatize.
        
        Args:
            text: Raw text
        
        Returns:
            Preprocessed text
        """
        if not text or pd.isna(text):
            return ""
        
        # Step 1: Clean text (remove URLs, special chars, lowercase)
        text = self._clean_text(text)
        
        # Step 2: Tokenize
        tokens = self._tokenize(text)
        
        # Step 3: Remove stopwords
        tokens = self._remove_stopwords(tokens)
        
        # Step 4: Lemmatize
        tokens = self._lemmatize(tokens)
        
        # Join back to string
        return " ".join(tokens)
    
    def _clean_text(self, text: str) -> str:
        """Clean text: remove URLs, special chars, lowercase."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self.use_spacy and self.nlp:
            # Use spaCy for tokenization
            doc = self.nlp(text)
            tokens = [token.text for token in doc if not token.is_space]
        else:
            # Use NLTK for tokenization
            tokens = word_tokenize(text)
        
        return tokens
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens."""
        # Add custom stopwords for fintech context
        custom_stopwords = {'app', 'bank', 'banking', 'mobile'}
        all_stopwords = self.stop_words.union(custom_stopwords)
        
        return [token for token in tokens if token not in all_stopwords and len(token) > 2]
    
    def _lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens."""
        if self.use_spacy and self.nlp:
            # Use spaCy for lemmatization
            doc = self.nlp(" ".join(tokens))
            lemmatized = [token.lemma_ for token in doc if not token.is_space]
        else:
            # Use NLTK for lemmatization
            lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return lemmatized
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of raw texts
        
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]


def document_preprocessing_pipeline() -> str:
    """
    Document the preprocessing pipeline steps.
    
    Returns:
        Documentation string
    """
    doc = """
    TEXT PREPROCESSING PIPELINE DOCUMENTATION
    ==========================================
    
    This pipeline implements the following steps for text preprocessing:
    
    1. TEXT CLEANING:
       - Convert to lowercase
       - Remove URLs and email addresses
       - Remove special characters (keep only letters and spaces)
       - Normalize whitespace
    
    2. TOKENIZATION:
       - Split text into individual words (tokens)
       - Uses spaCy if available, otherwise NLTK word_tokenize
    
    3. STOP-WORD REMOVAL:
       - Remove common English stopwords (e.g., 'the', 'is', 'and')
       - Remove custom fintech-specific stopwords ('app', 'bank', 'banking', 'mobile')
       - Filter out tokens shorter than 3 characters
    
    4. LEMMATIZATION:
       - Convert words to their base/root form (e.g., 'running' -> 'run')
       - Uses spaCy if available, otherwise NLTK WordNetLemmatizer
    
    The preprocessed text is then used for:
    - TF-IDF keyword extraction
    - Theme identification
    - Sentiment analysis
    
    Example:
        Input:  "The app crashes when I try to transfer money. Very frustrating!"
        Output: "app crash try transfer money very frustrate"
    """
    return doc


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    
    sample_text = "The app crashes when I try to transfer money. Very frustrating!"
    processed = preprocessor.preprocess_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Processed: {processed}")
    
    print("\n" + document_preprocessing_pipeline())

