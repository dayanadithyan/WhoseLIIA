from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nltk
import functools
import textstat
import numpy as np
import seaborn as sns
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from mpl_toolkits.mplot3d import Axes3D
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import re
import sys
import os
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stylometry.log')
    ]
)
logger = logging.getLogger('stylometry')

class TextCleaner:
    """Class for cleaning and normalizing text data for stylometric analysis"""
    
    def __init__(self):
        """Initialize text cleaner with required resources"""
        self.ensure_nltk_resources(['punkt', 'stopwords', 'wordnet'])
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        
        # Extended stopwords that might be relevant for specific stylometric tasks
        self.extended_stopwords = self.stopwords.union({
            'said', 'would', 'could', 'should', 'one', 'also', 'may', 'must', 'much'
        })
        
    def ensure_nltk_resources(self, resources):
        """Ensure NLTK resources are available"""
        for resource in resources:
            try:
                if resource == 'punkt':
                    nltk.data.find(f'tokenizers/{resource}')
                else:
                    nltk.data.find(f'corpora/{resource}')
            except LookupError:
                logger.info(f"Downloading NLTK resource: {resource}...")
                nltk.download(resource)
    
    def preprocess(self, text, lemmatize=False, extended_stopwords=False, 
                   lowercase=True, remove_digits=True, remove_punct=True):
        """
        Preprocess text with configurable options:
        - lemmatize: Apply lemmatization to reduce words to base forms
        - extended_stopwords: Use extended stopword list
        - lowercase: Convert to lowercase
        - remove_digits: Remove numerical digits
        - remove_punct: Remove punctuation
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Apply case normalization if requested
        if lowercase:
            text = text.lower()
            
        # Remove punctuation if requested
        if remove_punct:
            text = re.sub(r'[^\w\s]', '', text)
            
        # Remove digits if requested
        if remove_digits:
            text = re.sub(r'\d+', '', text)
            
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter stopwords
        stopword_set = self.extended_stopwords if extended_stopwords else self.stopwords
        filtered_tokens = [token for token in tokens if token not in stopword_set]
        
        # Apply lemmatization if requested
        if lemmatize:
            filtered_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
            
        return ' '.join(filtered_tokens)
        
    def clean_for_analysis(self, text, analysis_type='general'):
        """
        Apply cleaning tailored for specific analysis types:
        - general: Standard cleaning for general stylometry
        - authorship: Optimized for authorship attribution
        - sentiment: Preserves elements needed for sentiment analysis
        - readability: Minimal cleaning to maintain structure for readability
        """
        if analysis_type == 'authorship':
            # For authorship, we want to preserve author's unique patterns
            return self.preprocess(text, lemmatize=False, extended_stopwords=False, 
                                  lowercase=True, remove_digits=True)
        elif analysis_type == 'sentiment':
            # For sentiment, we need to keep punctuation and structure
            return self.preprocess(text, lemmatize=False, extended_stopwords=False,
                                  lowercase=True, remove_digits=True, remove_punct=False)
        elif analysis_type == 'readability':
            # For readability, minimize modifications
            return self.preprocess(text, lemmatize=False, extended_stopwords=False,
                                  lowercase=False, remove_digits=False, remove_punct=False)
        else:
            # General case
            return self.preprocess(text, lemmatize=True, extended_stopwords=True,
                                  lowercase=True, remove_digits=True)
                                  
    def normalize_line_breaks(self, text):
        """Normalize various line break patterns to standard Unix style"""
        text = text.replace('\r\n', '\n')  # Windows to Unix
        text = text.replace('\r', '\n')    # Mac OS Classic to Unix
        return text
        
    def remove_boilerplate(self, text, patterns=None):
        """Remove boilerplate text based on patterns"""
        default_patterns = [
            r'©.*?\d{4}.*?\n',              # Copyright notices
            r'All rights reserved\..*?\n',   # Rights statements
            r'http[s]?://\S+',               # URLs
            r'This book/text is in the public domain',  # Public domain notices
            r'Distributed by Project Gutenberg',        # Project Gutenberg notices
            r'END OF THE PROJECT GUTENBERG.*?$',           # Gutenberg footers
            r'START OF THE PROJECT GUTENBERG.*?\n'         # Gutenberg headers
        ]
        
        patterns = patterns or default_patterns
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text
        
    def clean_file(self, input_path, output_path=None, analysis_type='general'):
        """Clean an entire text file and optionally save the result"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Normalize line breaks
            text = self.normalize_line_breaks(text)
            
            # Remove boilerplate
            text = self.remove_boilerplate(text)
            
            # Clean according to analysis type
            cleaned_text = self.clean_for_analysis(text, analysis_type)
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                logger.info(f"Cleaned text saved to {output_path}")
                
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error cleaning file {input_path}: {str(e)}")
            return None


class StylometryAnalyzer:
    def __init__(self, text, cleaned_text=None, author=None, title=None):
        """
        Initialize analyzer with raw and cleaned text
        
        Parameters:
        - text: Raw original text for readability metrics
        - cleaned_text: Pre-processed text for stylistic analysis (if None, text is used)
        - author: Optional metadata about the author
        - title: Optional metadata about the work
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text cannot be empty")
            
        self.raw_text = text
        self.cleaned_text = cleaned_text if cleaned_text is not None else text
        self.author = author
        self.title = title
            
        # Process with NLP tools
        self._process_text()
        
    def _process_text(self):
        """Process text with NLP tools and extract basic features"""
        try:
            # Tokenize raw text for readability metrics
            self.raw_sentences = sent_tokenize(self.raw_text)
            
            # Process cleaned text for stylistic features
            self.sentences = sent_tokenize(self.cleaned_text)
            self.words = word_tokenize(self.cleaned_text.lower())
            self.word_counts = Counter(self.words)
            
            # Extract n-grams
            self.bigrams = list(ngrams(self.words, 2))
            self.trigrams = list(ngrams(self.words, 3))
            
            # Process with spaCy for deeper linguistic features
            global nlp
            try:
                if 'nlp' not in globals():
                    nlp = spacy.load("en_core_web_sm")
            except:
                spacy.cli.download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
            
            # Process with spaCy in chunks if text is very large
            if len(self.cleaned_text) > 1000000:  # For texts > 1MB
                self.doc = self._process_large_text(self.cleaned_text)
            else:
                self.doc = nlp(self.cleaned_text)
                
            # Extract linguistic features
            self._extract_linguistic_features()
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise ValueError(f"Text processing failed: {str(e)}")
            
    def _process_large_text(self, text):
        """Process large texts by chunking"""
        # Split text into manageable chunks
        chunk_size = 100000  # characters
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Process each chunk
        docs = list(nlp.pipe(chunks))
        
        # Concatenate results (simplified)
        return spacy.tokens.Doc.from_docs(docs)
        
    def _extract_linguistic_features(self):
        """Extract key linguistic features from processed text"""
        # POS tags
        self.pos_tags = [token.pos_ for token in self.doc]
        self.pos_counts = Counter(self.pos_tags)
        
        # Dependencies
        self.dep_tags = [token.dep_ for token in self.doc]
        self.dep_counts = Counter(self.dep_tags)
        
        # Named entities
        self.entities = [(ent.text, ent.label_) for ent in self.doc.ents]
        self.entity_counts = Counter([ent[1] for ent in self.entities])
        
        # Syntactic features
        self.sentence_lengths = [len(word_tokenize(sent)) for sent in self.sentences]
        self.word_lengths = [len(word) for word in self.words]
        
        # Punctuation
        self.punctuation = [token.text for token in self.doc if token.is_punct]
        self.punctuation_counts = Counter(self.punctuation)
        
    def calculate_lexical_statistics(self):
        """Calculate statistics related to vocabulary and word usage"""
        stats = {}
        
        # Basic counts
        stats['total_words'] = len(self.words)
        stats['unique_words'] = len(self.word_counts)
        stats['sentences'] = len(self.sentences)
        
        # Lexical density and diversity
        stats['type_token_ratio'] = round(stats['unique_words'] / max(stats['total_words'], 1), 4)
        stats['lexical_density'] = round(sum(1 for token in self.doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']) / 
                                        max(len(self.doc), 1), 4)
        
        # Hapax legomena and dis legomena (words occurring once or twice)
        stats['hapax_legomena'] = sum(1 for word, count in self.word_counts.items() if count == 1)
        stats['dis_legomena'] = sum(1 for word, count in self.word_counts.items() if count == 2)
        stats['hapax_ratio'] = round(stats['hapax_legomena'] / max(stats['total_words'], 1), 4)
        
        # Yule's K (vocabulary richness)
        m1 = stats['total_words']
        m2 = sum([freq ** 2 for freq in self.word_counts.values()])
        stats['yules_k'] = round((10000 * (m2 - m1)) / (m1 ** 2), 2)
        
        # Average word length
        stats['avg_word_length'] = round(sum(self.word_lengths) / max(len(self.word_lengths), 1), 2)
        
        # Word length distribution
        stats['word_length_distribution'] = Counter(self.word_lengths)
        
        # Most common words
        stats['most_common_words'] = dict(self.word_counts.most_common(20))
        
        return stats
        
    def calculate_syntactic_statistics(self):
        """Calculate statistics related to syntax and sentence structure"""
        stats = {}
        
        # Sentence length statistics
        stats['avg_sentence_length'] = round(sum(self.sentence_lengths) / max(len(self.sentence_lengths), 1), 2)
        stats['max_sentence_length'] = max(self.sentence_lengths) if self.sentence_lengths else 0
        stats['min_sentence_length'] = min(self.sentence_lengths) if self.sentence_lengths else 0
        stats['sentence_length_std'] = round(np.std(self.sentence_lengths), 2) if self.sentence_lengths else 0
        
        # POS tag statistics
        stats['pos_distribution'] = dict(self.pos_counts)
        
        # Normalized POS frequencies (per 1000 words)
        total_words = len(self.words)
        if total_words > 0:
            stats['normalized_pos'] = {pos: round(count * 1000 / total_words, 2) 
                                     for pos, count in self.pos_counts.items()}
        
        # Parts of speech ratios
        noun_count = self.pos_counts.get('NOUN', 0) + self.pos_counts.get('PROPN', 0)
        verb_count = self.pos_counts.get('VERB', 0)
        adj_count = self.pos_counts.get('ADJ', 0)
        adv_count = self.pos_counts.get('ADV', 0)
        
        stats['noun_verb_ratio'] = round(noun_count / max(verb_count, 1), 2)
        stats['adj_noun_ratio'] = round(adj_count / max(noun_count, 1), 2)
        stats['adv_verb_ratio'] = round(adv_count / max(verb_count, 1), 2)
        
        # Dependency relationships
        stats['dependency_distribution'] = dict(self.dep_counts)
        
        # Syntactic complexity (rough approximation)
        subordinating_conj = self.pos_counts.get('SCONJ', 0)
        stats['subordinating_conj_ratio'] = round(subordinating_conj / max(len(self.sentences), 1), 4)
        
        # Punctuation statistics
        stats['punctuation_distribution'] = dict(self.punctuation_counts)
        
        return stats
        
    def calculate_readability_metrics(self):
        """Calculate readability scores using the original, unprocessed text"""
        stats = {}
        
        # Standard readability metrics
        stats['flesch_reading_ease'] = round(textstat.flesch_reading_ease(self.raw_text), 2)
        stats['flesch_kincaid_grade'] = round(textstat.flesch_kincaid_grade(self.raw_text), 2)
        stats['gunning_fog'] = round(textstat.gunning_fog(self.raw_text), 2)
        stats['smog_index'] = round(textstat.smog_index(self.raw_text), 2)
        stats['coleman_liau_index'] = round(textstat.coleman_liau_index(self.raw_text), 2)
        stats['automated_readability_index'] = round(textstat.automated_readability_index(self.raw_text), 2)
        stats['dale_chall_readability_score'] = round(textstat.dale_chall_readability_score(self.raw_text), 2)
        
        # Text complexity metrics
        stats['avg_syllables_per_word'] = round(textstat.syllable_count(self.raw_text) / 
                                              max(len(word_tokenize(self.raw_text)), 1), 2)
        stats['difficult_words_ratio'] = round(textstat.difficult_words(self.raw_text) / 
                                             max(len(word_tokenize(self.raw_text)), 1), 4)
        
        return stats
        
    def calculate_distinctive_features(self):
        """Identify stylistically distinctive features"""
        stats = {}
        
        # n-gram analysis (most common bigrams and trigrams)
        bigram_counts = Counter(self.bigrams)
        trigram_counts = Counter(self.trigrams)
        
        stats['common_bigrams'] = dict(bigram_counts.most_common(15))
        stats['common_trigrams'] = dict(trigram_counts.most_common(15))
        
        # Named entity analysis
        stats['named_entity_types'] = dict(self.entity_counts)
        
        # Function words analysis (approximation using most common words)
        function_words = [word for word, _ in self.word_counts.most_common(50)]
        stats['function_words'] = function_words
        
        # Sentence-initial words
        sentence_initial_words = [sent.split()[0] for sent in self.sentences if sent.split()]
        stats['sentence_initial_distribution'] = dict(Counter(sentence_initial_words).most_common(10))
        
        return stats
        
    def analyze(self):
        """Run comprehensive stylometric analysis"""
        results = {}
        
        # Metadata if available
        if self.author or self.title:
            metadata = {}
            if self.author:
                metadata['author'] = self.author
            if self.title:
                metadata['title'] = self.title
            results['metadata'] = metadata
        
        # Lexical features
        results['lexical_statistics'] = self.calculate_lexical_statistics()
        
        # Syntactic features
        results['syntactic_statistics'] = self.calculate_syntactic_statistics()
        
        # Readability metrics
        results['readability_metrics'] = self.calculate_readability_metrics()
        
        # Distinctive features
        results['distinctive_features'] = self.calculate_distinctive_features()
        
        return results
        
    def export_analysis(self, output_dir, formats=None):
        """Export analysis results in multiple formats"""
        if formats is None:
            formats = ['json', 'csv']
            
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Get analysis results
        results = self.analyze()
        
        # File name base (use title if available)
        base_name = self.title.lower().replace(' ', '_') if self.title else 'stylometry_analysis'
        
        # Export in requested formats
        if 'json' in formats:
            # Flatten certain dictionaries for better JSON representation
            json_results = self._prepare_for_json(results)
            
            json_path = os.path.join(output_dir, f"{base_name}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2)
            logger.info(f"Analysis exported to JSON: {json_path}")
            
        if 'csv' in formats:
            # Flatten nested dictionaries for CSV
            flat_results = self._flatten_dict(results)
            
            # Convert to DataFrame
            df = pd.DataFrame([flat_results])
            
            csv_path = os.path.join(output_dir, f"{base_name}.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Analysis exported to CSV: {csv_path}")
            
        return results
            
    def _prepare_for_json(self, results):
        """Prepare analysis results for JSON export"""
        json_ready = dict(results)
        
        # Convert tuples to strings in n-grams
        if 'distinctive_features' in json_ready:
            if 'common_bigrams' in json_ready['distinctive_features']:
                json_ready['distinctive_features']['common_bigrams'] = {
                    ' '.join(k): v for k, v in results['distinctive_features']['common_bigrams'].items()
                }
            if 'common_trigrams' in json_ready['distinctive_features']:
                json_ready['distinctive_features']['common_trigrams'] = {
                    ' '.join(k): v for k, v in results['distinctive_features']['common_trigrams'].items()
                }
                
        return json_ready
        
    def _flatten_dict(self, d, parent_key='', sep='_'):
        """Flatten nested dictionaries for CSV export"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                # Skip dictionaries that would be too large for CSV
                if k in ['most_common_words', 'common_bigrams', 'common_trigrams', 
                         'word_length_distribution', 'punctuation_distribution']:
                    items.append((new_key, str(v)))
                else:
                    items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
                
        return dict(items)


class StylometryVisualizer:
    """Class for visualizing stylometric analysis results"""
    
    def __init__(self, analyzer):
        """Initialize with a StylometryAnalyzer instance"""
        self.analyzer = analyzer
        self.results = analyzer.analyze()
        
    def save_fig(self, fig, output_dir, filename):
        """Save figure with proper handling"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path = os.path.join(output_dir, filename)
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        logger.info(f"Figure saved: {output_path}")
        return output_path
        
    def plot_lexical_density(self, output_dir=None):
        """Visualize lexical density metrics"""
        lexical = self.results['lexical_statistics']
        
        # Prepare data
        labels = ['Type-Token Ratio', 'Lexical Density', 'Hapax Ratio']
        values = [
            lexical['type_token_ratio'],
            self.results['syntactic_statistics'].get('lexical_density', 0),
            lexical['hapax_ratio']
        ]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, values, color='steelblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.4f}',
                ha='center', va='bottom'
            )
            
        plt.title('Lexical Diversity Metrics', fontsize=15)
        plt.ylim(0, max(values) * 1.2)
        plt.ylabel('Ratio')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if output_dir:
            return self.save_fig(fig, output_dir, 'lexical_density.png')
        else:
            plt.show()
            
    def plot_word_length_distribution(self, output_dir=None):
        """Visualize distribution of word lengths"""
        word_lengths = self.analyzer.word_lengths
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot histogram with KDE
        sns.histplot(word_lengths, kde=True, bins=range(1, max(word_lengths) + 2),
                    color='steelblue', ax=ax)
        
        # Annotate with statistics
        avg_len = np.mean(word_lengths)
        median_len = np.median(word_lengths)
        
        ax.axvline(avg_len, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {avg_len:.2f}')
        ax.axvline(median_len, color='green', linestyle=':', linewidth=2,
                  label=f'Median: {median_len:.2f}')
        
        plt.title('Word Length Distribution', fontsize=15)
        plt.xlabel('Word Length (characters)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.legend()
        
        if output_dir:
            return self.save_fig(fig, output_dir, 'word_length_distribution.png')
        else:
            plt.show()
            
    def plot_pos_distribution(self, output_dir=None):
        """Visualize part-of-speech distribution"""
        pos_counts = self.results['syntactic_statistics']['pos_distribution']
        
        # Sort by frequency
        sorted_pos = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
        labels, values = zip(*sorted_pos)
        
        # POS tag descriptions for more readable labels
        pos_descriptions = {
            'ADJ': 'Adjective', 'ADP': 'Adposition', 'ADV': 'Adverb', 
            'AUX': 'Auxiliary', 'CCONJ': 'Coord Conj', 'DET': 'Determiner',
            'INTJ': 'Interjection', 'NOUN': 'Noun', 'NUM': 'Numeral',
            'PART': 'Particle', 'PRON': 'Pronoun', 'PROPN': 'Proper Noun',
            'PUNCT': 'Punctuation', 'SCONJ': 'Subord Conj', 'SYM': 'Symbol',
            'VERB': 'Verb', 'X': 'Other'
        }
        readable_labels = [f"{pos_descriptions.get(label, label)}\n({label})" for label in labels]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Calculate normalized frequencies per 1000 words
        total_words = self.results['lexical_statistics']['total_words']
        normalized_values = [count * 1000 / total_words for count in values]
        
        # Plot both raw and normalized values
        x = np.arange(len(readable_labels))
        width = 0.35
        
        ax1 = ax
        bars1 = ax1.bar(x - width/2, values, width, label='Raw Count', color='steelblue')
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_ylim(0, max(values) * 1.1)
        
        # Create second y-axis for normalized values
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, normalized_values, width, label='Per 1000 Words', 
                       color='lightcoral', alpha=0.7)
        ax2.set_ylabel('Frequency per 1000 words', fontsize=12)
        ax2.set_ylim(0, max(normalized_values) * 1.1)
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Customize plot
        ax1.set_xticks(x)
        ax1.set_xticklabels(readable_labels, rotation=45, ha='right')
        ax1.set_title('Part-of-Speech Distribution', fontsize=15)
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            return self.save_fig(fig, output_dir, 'pos_distribution.png')
        else:
            plt.show()
            
    def plot_readability_metrics(self, output_dir=None):
        """Visualize readability metrics with interpretations"""
        readability = self.results['readability_metrics']
        
        # Select key metrics
        metrics = {
            'Flesch Reading Ease': readability['flesch_reading_ease'],
            'Flesch-Kincaid Grade': readability['flesch_kincaid_grade'],
            'Gunning Fog Index': readability['gunning_fog'],
            'SMOG Index': readability['smog_index'],
            'Coleman-Liau Index': readability['coleman_liau_index']
        }
        
        # Readability interpretations
        interpretations = {
            'Flesch Reading Ease': {
                90: 'Very Easy (5th grade)',
                80: 'Easy (6th grade)',
                70: 'Fairly Easy (7th grade)',
                60: 'Standard (8-9th grade)',
                50: 'Fairly Difficult (10-12th grade)',
                30: 'Difficult (College)',
                10: 'Very Difficult (Graduate)'
            },
            'Grade Level': {
                5: 'Elementary',
                8: 'Middle School',
                12: 'High School',
                16: 'College',
                20: 'Graduate'
            }
        }
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Flesch Reading Ease with interpretation bands
        ease_score = metrics['Flesch Reading Ease']
        ax1.barh(['Flesch Reading Ease'], [ease_score], color='steelblue')
        
        ease_bands = sorted(interpretations['Flesch Reading Ease'].items(), reverse=True)
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(ease_bands)))
        
        for (score, label), color in zip(ease_bands, colors):
            ax1.axvline(x=score, color=color, linestyle='--', alpha=0.7)
            ax1.text(score, 0, f" {label}", va='center', alpha=0.7, color=color)
            
        ax1.set_xlim(0, 100)
        ax1.set_title('Flesch Reading Ease', fontsize=14)
        ax1.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Plot 2: Flesch-Kincaid Grade with interpretation
        grade_score = metrics['Flesch-Kincaid Grade']
        ax2.barh(['Flesch-Kincaid Grade'], [grade_score], color='steelblue')
        
        # Determine grade interpretation using sorted thresholds
        grade_interpretation = "Unknown"
        for thresh in sorted(interpretations['Grade Level'].keys()):
            if grade_score <= thresh:
                grade_interpretation = interpretations['Grade Level'][thresh]
                break
        ax2.text(grade_score, 0, f" {grade_score:.2f} ({grade_interpretation})", va='center', color='black')
        ax2.set_title('Flesch-Kincaid Grade', fontsize=14)
        ax2.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if output_dir:
            return self.save_fig(fig, output_dir, 'readability_metrics.png')
        else:
            plt.show()
            
    def export_d3_data(self, output_path):
        """Export analysis results to a JSON file for D3.js visualization"""
        d3_data = {
            "most_common_words": self.analyzer.calculate_lexical_statistics().get("most_common_words", {}),
            "pos_distribution": self.results['syntactic_statistics'].get("pos_distribution", {}),
            "readability_metrics": self.results.get("readability_metrics", {})
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(d3_data, f, indent=2)
        logger.info(f"D3 data exported to {output_path}")
        return output_path


if __name__ == "__main__":
    # Example usage: Clean a text file, analyze it, visualize results, and export D3 data
    input_path = "/Volumes/LNX/NEW/stylometry-dh/data/paradise_lost_cleaned.txt"  # Ensure this file exists
    output_dir = "output"
    
    # Clean the input text file
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean_file(input_path, output_path=os.path.join(output_dir, "cleaned_input.txt"))
    
    # Read raw text (for readability metrics) and initialize analyzer
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    analyzer = StylometryAnalyzer(text=raw_text, cleaned_text=cleaned_text, author="Author Name", title="Sample Title")
    
    # Export analysis in JSON and CSV formats
    analyzer.export_analysis(output_dir, formats=['json', 'csv'])
    
    # Create visualizations using matplotlib/seaborn
    visualizer = StylometryVisualizer(analyzer)
    visualizer.plot_lexical_density(output_dir)
    visualizer.plot_word_length_distribution(output_dir)
    visualizer.plot_pos_distribution(output_dir)
    visualizer.plot_readability_metrics(output_dir)
    
    # Export data for D3.js visualization
    visualizer.export_d3_data(os.path.join(output_dir, "d3_data.json"))
