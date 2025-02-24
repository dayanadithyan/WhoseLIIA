# src/stylometry/analyzer.py
from pathlib import Path
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import matplotlib.pyplot as plt

nltk.download('punkt', quiet=True)

class StylometryAnalyzer:
    def __init__(self, text: str):
        """Initialize analyzer with text for stylometric analysis
        
        Args:
            text (str): Input text to analyze
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")
            
        self.text = text
        self.sentences = sent_tokenize(text)
        self.words = self._process_words()
        self._validate_initialization()

    def _validate_initialization(self):
        """Ensure text processing succeeded"""
        if not self.words:
            raise ValueError("Text processing failed - no valid words found")
        if not self.sentences:
            raise ValueError("Text processing failed - no sentences found")

    def _process_words(self) -> list:
        """Clean and tokenize text into words"""
        try:
            tokens = word_tokenize(self.text)
            return [token.lower() for token in tokens if token.isalpha()]
        except Exception as e:
            raise ValueError(f"Text tokenization failed: {str(e)}")

    @property
    def word_counts(self) -> Counter:
        """Get frequency distribution of words"""
        return Counter(self.words)

    def analyze(self) -> dict:
        """Run complete analysis and return metrics"""
        return {
            'word_count': len(self.words),
            'unique_words': len(self.word_counts),
            'avg_word_length': self.average_word_length(),
            'type_token_ratio': self.type_token_ratio(),
            'hapax_legomena': self.hapax_legomena(),
            'avg_sentence_length': self.average_sentence_length(),
            'flesch_kincaid': self.flesch_kincaid_grade()
        }

    def average_word_length(self) -> float:
        """Calculate average word length in characters"""
        return round(sum(len(word) for word in self.words) / len(self.words), 2)

    def type_token_ratio(self) -> float:
        """Calculate Type-Token Ratio (TTR)"""
        return round(len(self.word_counts) / len(self.words), 4)

    def hapax_legomena(self) -> float:
        """Calculate Hapax Legomena ratio"""
        hapax = sum(1 for count in self.word_counts.values() if count == 1)
        return round(hapax / len(self.words), 4)

    def average_sentence_length(self) -> float:
        """Calculate average sentence length in words"""
        return round(len(self.words) / len(self.sentences), 2)

    def flesch_kincaid_grade(self) -> float:
        """Calculate Flesch-Kincaid Grade Level"""
        total_syllables = sum(self._count_syllables(word) for word in self.words)
        return round(
            (0.39 * (len(self.words)/len(self.sentences)) + 
             11.8 * (total_syllables/len(self.words))) - 15.59,
            2
        )

    def _count_syllables(self, word: str) -> int:
        """Estimate syllables in a word"""
        word = word.lower().strip(".:;?!")
        if len(word) <= 3:
            return 1
            
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        
        for char in word:
            if char in vowels and not prev_vowel:
                count += 1
                prev_vowel = True
            else:
                prev_vowel = False
                
        if word.endswith('e') and count > 1:
            count -= 1
            
        return max(1, count)

    def plot_word_length_dist(self, output_path: str = None):
        """Generate word length distribution histogram"""
        lengths = [len(word) for word in self.words]
        
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=range(1, max(lengths)+2), edgecolor='black')
        plt.title(f'Word Length Distribution (n={len(self.words)})')
        plt.xlabel('Word Length (characters)')
        plt.ylabel('Frequency')
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    @classmethod
    def process_directory(cls, dir_path: str, output_csv: str = None) -> pd.DataFrame:
        """Batch process text files in a directory"""
        path = Path(dir_path)
        if not path.is_dir():
            raise ValueError(f"Invalid directory: {dir_path}")
            
        results = []
        
        for file in path.glob('*.txt'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                analyzer = cls(text)
                analysis = analyzer.analyze()
                analysis['filename'] = file.name
                results.append(analysis)
                
            except Exception as e:
                print(f"Skipped {file.name}: {str(e)}")
                continue
                
        df = pd.DataFrame(results)
        if output_csv:
            df.to_csv(output_csv, index=False)
        return df