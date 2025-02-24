from pathlib import Path
from collections import Counter
import math
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import matplotlib.pyplot as plt
nltk.download('punkt', quiet=True)

class StylometryAnalyzer:
    def __init__(self, text: str):
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        self.text = text
        self.sentences = sent_tokenize(text)
        self.words = self._process_words()
        
    def _process_words(self):
        tokens = word_tokenize(self.text)
        return [token.lower() for token in tokens if token.isalpha()]
    
    @property
    def word_counts(self):
        return Counter(self.words)
    
    def average_word_length(self) -> float:
        if not self.words:
            return 0.0
        return sum(len(word) for word in self.words) / len(self.words)
    
    def type_token_ratio(self) -> float:
        if not self.words:
            return 0.0
        return len(self.word_counts) / len(self.words)
    
    def hapax_legomena(self) -> float:
        if not self.words:
            return 0.0
        hapax = sum(1 for count in self.word_counts.values() if count == 1)
        return hapax / len(self.words)
    
    def average_sentence_length(self) -> float:
        if not self.sentences:
            return 0.0
        return len(self.words) / len(self.sentences)
    
    def flesch_kincaid_grade(self) -> float:
        if not self.sentences or not self.words:
            return 0.0
        total_syllables = sum(self._count_syllables(word) for word in self.words)
        return (0.39 * (len(self.words)/len(self.sentences)) + 
               11.8 * (total_syllables/len(self.words))) - 15.59
    
    def _count_syllables(self, word: str) -> int:
        word = word.lower()
        if len(word) <= 3:
            return 1
        
        count = 0
        vowels = 'aeiouy'
        prev_vowel = False
        
        for i, char in enumerate(word):
            if char in vowels:
                if not prev_vowel:
                    count += 1
                prev_vowel = True
            else:
                prev_vowel = False
        
        if word.endswith('e') and count > 1:
            count -= 1
            
        return max(1, count)
    
    def plot_word_length_dist(self, output_path: str = None):
        lengths = [len(word) for word in self.words]
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=range(1, max(lengths)+2))
        plt.title('Word Length Distribution')
        plt.xlabel('Word Length')
        plt.ylabel('Frequency')
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    @classmethod
    def process_directory(cls, dir_path: str, output_csv: str = None) -> pd.DataFrame:
        path = Path(dir_path)
        results = []
        
        for file in path.glob('*.txt'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read()
                analyzer = cls(text)
                results.append({
                    'filename': file.name,
                    'avg_word_length': analyzer.average_word_length(),
                    'ttr': analyzer.type_token_ratio(),
                    'hapax': analyzer.hapax_legomena(),
                    'avg_sentence_length': analyzer.average_sentence_length(),
                    'flesch_kincaid': analyzer.flesch_kincaid_grade()
                })
            except Exception as e:
                print(f"Error processing {file.name}: {str(e)}")
        
        df = pd.DataFrame(results)
        if output_csv:
            df.to_csv(output_csv, index=False)
        return df