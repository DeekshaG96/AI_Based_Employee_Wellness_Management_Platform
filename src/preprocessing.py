"""
Text Preprocessing Pipeline for Employee Wellness Feedback.
Fulfills M1 Deliverables:
- Text Cleaning
- Tokenization
- Stop-word filtering
- Lemmatization
- Invalid/Empty input handling
"""

import re
import string
from typing import List, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Ensure essential NLTK resources are available
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except (LookupError, AttributeError):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

class TextPreprocessor:
    """Preprocesses employee feedback text for NLP analysis."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set([
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
                'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                'can', 'will', 'just', 'don', 'should', 'now'
            ])
        # Sentiment-carrying negation words to preserve so sentiment analysis remains accurate
        self.sentiment_negations = {'not', 'no', 'nor', 'neither', 'never', 'hardly', 'barely', 'scarcely'}

    def clean_text(self, text: str) -> str:
        """Cleans input text by removing URLs, emails, special symbols, and extra whitespace."""
        if not text or not isinstance(text, str):
            return ""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Normalize punctuation to spaces
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        # Normalize whitespace and convert to lowercase
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Splits cleaned text into individual word tokens."""
        if not text or not isinstance(text, str):
            return []
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        # Keep only alphabetic tokens
        return [tok for tok in tokens if tok.isalpha()]

    def filter_stopwords(self, tokens: List[str], preserve_negations: bool = True) -> List[str]:
        """Filters out non-informative stopwords while preserving negation context if requested."""
        if not tokens:
            return []
        filtered = []
        for tok in tokens:
            if preserve_negations and tok in self.sentiment_negations:
                filtered.append(tok)
            elif tok not in self.stop_words:
                filtered.append(tok)
        return filtered

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Reduces tokens to their root dictionary form using WordNetLemmatizer."""
        if not tokens:
            return []
        lemmatized = []
        for tok in tokens:
            # Lemmatize both as verb and noun fallback for optimal root form
            lem = self.lemmatizer.lemmatize(tok, pos='v')
            if lem == tok:
                lem = self.lemmatizer.lemmatize(tok, pos='n')
            lemmatized.append(lem)
        return lemmatized

    def process(self, text: str) -> Dict[str, Any]:
        """
        Executes the entire NLP preprocessing pipeline.
        Returns detailed intermediate states for M1 demonstration.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "is_valid": False,
                "error": "Input is empty or contains only whitespace.",
                "raw_text": text or "",
                "cleaned_text": "",
                "tokens": [],
                "filtered_tokens": [],
                "lemmatized_tokens": [],
                "token_count": 0,
                "vocab_count": 0
            }

        cleaned = self.clean_text(text)
        if not cleaned:
            return {
                "is_valid": False,
                "error": "Input does not contain valid alphabetic words after cleaning.",
                "raw_text": text,
                "cleaned_text": "",
                "tokens": [],
                "filtered_tokens": [],
                "lemmatized_tokens": [],
                "token_count": 0,
                "vocab_count": 0
            }

        tokens = self.tokenize(cleaned)
        filtered = self.filter_stopwords(tokens, preserve_negations=True)
        lemmatized = self.lemmatize(filtered)

        return {
            "is_valid": True,
            "error": None,
            "raw_text": text,
            "cleaned_text": cleaned,
            "tokens": tokens,
            "filtered_tokens": filtered,
            "lemmatized_tokens": lemmatized,
            "token_count": len(tokens),
            "vocab_count": len(set(lemmatized))
        }
