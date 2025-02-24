from stylometry.analyzer import StylometryAnalyzer

def test_analyzer_initialization():
    # Provide required text argument
    analyzer = StylometryAnalyzer(text="Sample text for testing")
    assert analyzer.text == "Sample text for testing"