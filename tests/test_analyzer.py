import pytest
from src.stylometry.analyzer import StylometryAnalyzer

SAMPLE_TEXT = """
The quick brown fox jumps over the lazy dog. This sentence contains all letters 
of the English alphabet. Stylometry helps analyze literary style through 
quantitative methods.
"""

def test_average_word_length():
    analyzer = StylometryAnalyzer(SAMPLE_TEXT)
    assert round(analyzer.average_word_length(), 2) == 4.68

def test_flesch_kincaid():
    analyzer = StylometryAnalyzer("Simple sentence. With two examples.")
    assert round(analyzer.flesch_kincaid_grade(), 1) == 9.0

def test_empty_text():
    with pytest.raises(ValueError):
        StylometryAnalyzer("")

def test_syllable_counting():
    analyzer = StylometryAnalyzer("dummy")
    assert analyzer._count_syllables("make") == 1
    assert analyzer._count_syllables("example") == 3
    assert analyzer._count_syllables("the") == 1