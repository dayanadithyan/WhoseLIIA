import re
import json
import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import nltk
import spacy
import textacy
from nltk.corpus import wordnet
from transformers import pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

class AdvancedStylometryAnalyzer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        try:
            nltk.download(['wordnet', 'punkt'], quiet=True)
            self.nlp = spacy.load('en_core_web_sm')
            self.ner_pipeline = pipeline(
                "ner", 
                model="dslim/bert-base-NER", 
                aggregation_strategy="simple"
            )
        except Exception as e:
            self.logger.error(f"NLP model loading error: {e}")
            raise

    def read_txt(self, txt_path: str, encoding: str = 'utf-8') -> str:
        try:
            with open(txt_path, 'r', encoding=encoding) as file:
                text = file.read()
            
            text = re.sub(r'\s+', ' ', text).strip()
            
            if not text:
                raise ValueError("No text could be extracted from the file")
            
            return text
        except Exception as e:
            self.logger.error(f"Text file reading error: {e}")
            raise

    def _calculate_dependency_depth(self, doc):
        def token_depth(token):
            depth = 0
            while token.head != token:
                depth += 1
                token = token.head
            return depth
        
        return max((token_depth(token) for token in doc if token.dep_ != 'ROOT'), default=0)

    def _analyze_verb_complexity(self, doc):
        return {
            "modal_verbs": [token.text for token in doc if token.pos_ == 'AUX'],
            "passive_constructions": [sent.text for sent in doc.sents if any(token.dep_ == 'nsubjpass' for token in sent)]
        }

    def _analyze_clause_complexity(self, doc):
        return {
            "subordination_index": len([sent for sent in doc.sents if any(token.dep_ == 'mark' for token in sent.root.children)]),
            "coordination_complexity": len([sent for sent in doc.sents if any(token.dep_ == 'cc' for token in sent.root.children)])
        }

    def _detect_latin_linguistic_markers(self, text):
        latin_markers = ['ae', 'orum', 'um', 'que', 'iter']
        return {
            "marker_frequency": {marker: text.lower().count(marker) for marker in latin_markers},
            "total_latin_markers": sum(text.lower().count(marker) for marker in latin_markers)
        }

    def _identify_archaic_linguistic_patterns(self, text):
        archaic_patterns = [
            r'\b\w+eth\b',  # -eth endings
            r'\b\w+est\b',  # -est endings
            r'\bwhoso\b',   # archaic pronouns
            r'\bthee\b',    # archaic second-person pronouns
            r'\bthine\b'
        ]
        
        return [
            match.group(0) 
            for pattern in archaic_patterns 
            for match in re.finditer(pattern, text, re.IGNORECASE)
        ]

    def _detect_grammatical_archaisms(self, doc):
        return {
            "archaic_verb_forms": [
                token.text for token in doc 
                if token.pos_ == 'VERB' and re.search(r'(eth|est)$', token.text)
            ]
        }

    def _advanced_punctuation_analysis(self, text):
        punctuation_types = [';', ':', '—', '…', '\'']
        return {
            "punctuation_frequency": {
                punct: text.count(punct) for punct in punctuation_types
            },
            "complex_punctuation_ratio": len(
                re.findall(r'[;:—…\'"]', text)
            ) / len(text)
        }

    def _detect_orthographic_variations(self, text):
        return {
            "spelling_variations": {
                "archaic_spellings": len(re.findall(r'\b\w+(?:our|or)\b', text)),
                "unconventional_orthography": len(re.findall(r'\b\w+[ck](?:our|or)\b', text))
            }
        }

    def _extract_named_entities(self, text):
        return self.ner_pipeline(text)

    def _calculate_thematic_coherence(self, doc):
        return {
            "top_semantic_concepts": [
                {"concept": ent.text, "type": ent.label_} 
                for ent in doc.ents
            ]
        }

    def linguistic_complexity_analysis(self, text):
        doc = self.nlp(text)
        
        return {
            "linguistic_metrics": {
                "dependency_depth": self._calculate_dependency_depth(doc),
                "constituency_complexity": textacy.text_stats.parse_tree_depth(text),
                "avg_sentence_length": np.mean([len(sent) for sent in doc.sents]),
            },
            
            "grammatical_analysis": {
                "noun_phrases": [np.text for np in doc.noun_chunks],
                "verb_complexity": self._analyze_verb_complexity(doc),
                "clause_structure": self._analyze_clause_complexity(doc)
            },
            
            "linguistic_heritage": {
                "latin_influence": self._detect_latin_linguistic_markers(text),
                "archaic_linguistic_features": {
                    "archaic_words": self._identify_archaic_linguistic_patterns(text),
                    "grammatical_archaisms": self._detect_grammatical_archaisms(doc)
                }
            },
            
            "stylistic_features": {
                "punctuation_diversity": self._advanced_punctuation_analysis(text),
                "orthographic_variation": self._detect_orthographic_variations(text)
            },
            
            "semantic_insights": {
                "named_entities": self._extract_named_entities(text),
                "thematic_coherence": self._calculate_thematic_coherence(doc)
            }
        }

class StylometryMLAnalyzer:
    def __init__(self, stylometry_data):
        self.data = stylometry_data
        
    def extract_ml_features(self):
        features = {
            'dependency_depth': self.data['linguistic_metrics']['dependency_depth'],
            'avg_sentence_length': self.data['linguistic_metrics']['avg_sentence_length'],
            'latin_marker_frequency': sum(self.data['linguistic_heritage']['latin_influence']['marker_frequency'].values()),
            'archaic_word_count': len(self.data['linguistic_heritage']['archaic_linguistic_features']['archaic_words']),
            'punctuation_complexity': self.data['stylistic_features']['punctuation_diversity']['complex_punctuation_ratio']
        }
        return pd.DataFrame([features])

def visualize_stylometric_analysis(analysis):
    ml_analyzer = StylometryMLAnalyzer(analysis)
    features = ml_analyzer.extract_ml_features()
    
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c='blue', alpha=0.7)
    plt.title('Stylometric Feature Space')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.tight_layout()
    plt.savefig('stylometry_pca.png')
    plt.close()
    
    categories = [
        'Dependency Depth', 
        'Sentence Length', 
        'Latin Influence', 
        'Archaic Words', 
        'Punctuation Complexity'
    ]
    values = [
        analysis['linguistic_metrics']['dependency_depth'],
        analysis['linguistic_metrics']['avg_sentence_length'],
        sum(analysis['linguistic_heritage']['latin_influence']['marker_frequency'].values()),
        len(analysis['linguistic_heritage']['archaic_linguistic_features']['archaic_words']),
        analysis['stylistic_features']['punctuation_diversity']['complex_punctuation_ratio']
    ]
    
    plt.figure(figsize=(8, 8))
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    plt.polar(angles, values, 'o-', linewidth=2)
    plt.fill(angles, values, alpha=0.25)
    plt.xticks(angles[:-1], categories)
    plt.title('Linguistic Style Radar Chart')
    plt.tight_layout()
    plt.savefig('linguistic_radar.png')
    plt.close()

def streamlit_stylometry_app():
    st.title('Advanced Stylometric Analysis')
    
    file_path = st.text_input('Enter path to text file')
    
    if st.button('Analyze') and file_path:
        try:
            analyzer = AdvancedStylometryAnalyzer()
            text = analyzer.read_txt(file_path)
            analysis = analyzer.linguistic_complexity_analysis(text)
            
            st.header('Linguistic Complexity Metrics')
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Dependency Depth", 
                    analysis['linguistic_metrics']['dependency_depth']
                )
                st.metric(
                    "Avg Sentence Length", 
                    f"{analysis['linguistic_metrics']['avg_sentence_length']:.2f}"
                )
            
            with col2:
                st.metric(
                    "Latin Linguistic Markers", 
                    sum(analysis['linguistic_heritage']['latin_influence']['marker_frequency'].values())
                )
                st.metric(
                    "Archaic Words", 
                    len(analysis['linguistic_heritage']['archaic_linguistic_features']['archaic_words'])
                )
            
            st.header('Stylometric Visualizations')
            visualize_stylometric_analysis(analysis)
            
            st.image('stylometry_pca.png', caption='PCA Feature Space')
            st.image('linguistic_radar.png', caption='Linguistic Style Radar Chart')
            
            with st.expander("Full Analysis Details"):
                st.json(analysis)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

def main():
    streamlit_stylometry_app()

if __name__ == "__main__":
    main()