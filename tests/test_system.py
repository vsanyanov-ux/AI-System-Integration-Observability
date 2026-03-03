import pytest
from app import SentimentAnalyzer

def test_sentiment_positive():
    system = SentimentAnalyzer()
    result = system.analyze("I am very happy")
    assert result['label'] == 'POSITIVE'

def test_sentiment_negative():
    system = SentimentAnalyzer()
    result = system.analyze("I am very sad")
    assert result['label'] == 'NEGATIVE'
