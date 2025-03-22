import streamlit as st
import re
import nltk
from typing import List, Dict, Any, Union, Optional
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class TextAnalyzer:
    """
    Analyze text content from research documents for various metrics.
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive text analysis.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Basic metrics
        word_count = len(word_tokenize(text))
        sentence_count = len(sent_tokenize(text))
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Vocabulary metrics
        words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
        unique_words = set(words)
        vocabulary_size = len(unique_words)
        lexical_diversity = vocabulary_size / max(len(words), 1)
        
        # Content metrics
        content_words = [word for word in words if word not in self.stop_words]
        content_word_density = len(content_words) / max(len(words), 1)
        
        # Keyword extraction
        keywords = self._extract_keywords(text, top_n=20)
        
        # Readability
        readability_scores = self._calculate_readability(text)
        
        return {
            "basic_metrics": {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": avg_sentence_length,
                "paragraph_count": len(re.split(r'\n\s*\n', text))
            },
            "vocabulary_metrics": {
                "vocabulary_size": vocabulary_size,
                "lexical_diversity": lexical_diversity,
                "content_word_density": content_word_density
            },
            "readability": readability_scores,
            "keywords": keywords
        }
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[Dict[str, Union[str, int]]]:
        """
        Extract the most frequent content words as keywords.
        
        Args:
            text: Text to analyze for keywords
            top_n: Number of top keywords to return
            
        Returns:
            List of keyword dictionaries with word and frequency
        """
        words = word_tokenize(text.lower())
        content_words = [word for word in words if word.isalpha() and word not in self.stop_words and len(word) > 2]
        
        # Count word frequencies
        word_freq = Counter(content_words)
        
        # Get top N keywords
        top_keywords = [{"word": word, "count": count} for word, count in word_freq.most_common(top_n)]
        
        return top_keywords
    
    def _calculate_readability(self, text: str) -> Dict[str, float]:
        """
        Calculate readability metrics for the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with readability scores
        """
        # Basic components for readability formulas
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        word_count = len(words)
        sentence_count = len(sentences)
        
        # Calculate syllable count (approximation)
        syllable_count = 0
        for word in words:
            word_clean = word.lower().strip(".,!?:;\"'()[]{}").strip()
            if not word_clean or not word_clean[0].isalpha():
                continue
                
            # Count vowel sequences as syllables (simple approximation)
            syllable_count += max(1, len(re.findall(r'[aeiouy]+', word_clean)))
        
        # Calculate metrics (if text is long enough)
        if word_count > 0 and sentence_count > 0:
            # Flesch Reading Ease
            flesch = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
            
            # Gunning Fog Index
            complex_words = sum(1 for word in words if len(word) > 7)
            fog_index = 0.4 * ((word_count / sentence_count) + 100 * (complex_words / word_count))
            
            return {
                "flesch_reading_ease": max(0, min(100, flesch)),
                "gunning_fog_index": fog_index,
                "avg_syllables_per_word": syllable_count / max(word_count, 1)
            }
        else:
            return {
                "flesch_reading_ease": 0,
                "gunning_fog_index": 0,
                "avg_syllables_per_word": 0
            }
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare two texts for similarity in content and style.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            Dictionary with comparison results
        """
        # Analyze both texts
        analysis1 = self.analyze_text(text1)
        analysis2 = self.analyze_text(text2)
        
        # Compare vocabularies
        words1 = set([word.lower() for word in word_tokenize(text1) if word.isalpha()])
        words2 = set([word.lower() for word in word_tokenize(text2) if word.isalpha()])
        
        common_words = words1.intersection(words2)
        
        vocabulary_overlap = len(common_words) / max(1, min(len(words1), len(words2)))
        
        # Compare content words (excluding stopwords)
        content_words1 = set([word for word in words1 if word not in self.stop_words])
        content_words2 = set([word for word in words2 if word not in self.stop_words])
        
        common_content_words = content_words1.intersection(content_words2)
        content_overlap = len(common_content_words) / max(1, min(len(content_words1), len(content_words2)))
        
        # Compare style metrics
        style_similarity = 1 - min(1, abs(analysis1["basic_metrics"]["avg_sentence_length"] - 
                                        analysis2["basic_metrics"]["avg_sentence_length"]) / 
                                max(1, max(analysis1["basic_metrics"]["avg_sentence_length"],
                                          analysis2["basic_metrics"]["avg_sentence_length"])))
        
        return {
            "vocabulary_overlap": vocabulary_overlap,
            "content_overlap": content_overlap,
            "style_similarity": style_similarity,
            "common_words": list(common_words)[:100],  # Limit to prevent overwhelming UI
            "common_content_words": list(common_content_words)[:100]
        }
    
    def analyze_citations(self, text: str) -> Dict[str, Any]:
        """
        Analyze citation patterns in the text.
        
        Args:
            text: Text to analyze for citations
            
        Returns:
            Dictionary with citation analysis
        """
        # Look for common citation formats
        # APA style: (Author, Year)
        apa_citations = re.findall(r'\(([A-Za-z\s]+),\s+(\d{4}[a-z]?)\)', text)
        
        # IEEE style: [1], [2], etc.
        ieee_citations = re.findall(r'\[(\d+)\]', text)
        
        # Harvard style: Author (Year)
        harvard_citations = re.findall(r'([A-Za-z\s]+)\s+\((\d{4}[a-z]?)\)', text)
        
        # Check which style is likely used
        citation_counts = {
            "apa": len(apa_citations),
            "ieee": len(ieee_citations),
            "harvard": len(harvard_citations)
        }
        
        dominant_style = max(citation_counts, key=citation_counts.get)
        
        # Extract cited years if applicable
        cited_years = []
        if dominant_style == "apa":
            cited_years = [int(year) for _, year in apa_citations if year.isdigit()]
        elif dominant_style == "harvard":
            cited_years = [int(year) for _, year in harvard_citations if year.isdigit()]
            
        # Calculate citation statistics
        citation_stats = {
            "total_citations": citation_counts[dominant_style],
            "citation_density": citation_counts[dominant_style] / max(1, len(sent_tokenize(text))),
            "dominant_style": dominant_style
        }
        
        # Add year-based stats if available
        if cited_years:
            current_year = 2025  # Assuming current year
            citation_stats["avg_citation_age"] = current_year - (sum(cited_years) / max(1, len(cited_years)))
            citation_stats["recent_citations"] = sum(1 for year in cited_years if (current_year - year) <= 5)
            citation_stats["recent_citations_percentage"] = citation_stats["recent_citations"] / max(1, len(cited_years))
            
        return citation_stats
    
    def extract_key_sentences(self, text: str, n: int = 5) -> List[str]:
        """
        Extract key sentences from the text.
        
        Args:
            text: Text to analyze
            n: Number of key sentences to extract
            
        Returns:
            List of key sentences
        """
        sentences = sent_tokenize(text)
        
        if len(sentences) <= n:
            return sentences
            
        # Score sentences based on position and keyword presence
        keywords = set(kw["word"] for kw in self._extract_keywords(text, top_n=30))
        
        scored_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Position score (beginning and end sentences are important)
            pos_score = 1.0
            if i < len(sentences) / 10:  # First 10% of sentences
                pos_score = 1.5
            elif i > len(sentences) * 0.9:  # Last 10% of sentences
                pos_score = 1.2
                
            # Keyword score
            words = word_tokenize(sentence.lower())
            keyword_count = sum(1 for word in words if word in keywords)
            keyword_score = keyword_count / max(1, len(words))
            
            # Length score (prefer medium-length sentences)
            length = len(words)
            length_score = 1.0
            if 8 <= length <= 20:
                length_score = 1.2
                
            # Combine scores
            total_score = pos_score * keyword_score * length_score
            
            scored_sentences.append((sentence, total_score))
            
        # Sort by score and get top N
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sent for sent, _ in scored_sentences[:n]]

def analyze_document(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a document and return comprehensive analysis.
    
    Args:
        document: Document dictionary with full_text field
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = TextAnalyzer()
    text = document.get("full_text", "")
    
    if not text:
        return {"error": "No text content found in document"}
        
    # Perform analysis
    text_analysis = analyzer.analyze_text(text)
    citation_analysis = analyzer.analyze_citations(text)
    key_sentences = analyzer.extract_key_sentences(text, n=5)
    
    return {
        "document_id": document.get("document_id", ""),
        "filename": document.get("filename", ""),
        "text_analysis": text_analysis,
        "citation_analysis": citation_analysis,
        "key_sentences": key_sentences
    }