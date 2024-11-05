from nltk.tokenize import TweetTokenizer
from .config import EXCHANGES, REGEX_PATTERN
from .telegram_message_normalizer import TelegramMessageNormalizer

"""
Module to provide a pre-configured instance of TelegramMessageNormalizer for use in other scripts.
"""

# Create an instance of TelegramMessageNormalizer with predefined parameters
StandardNormalizer = TelegramMessageNormalizer(
    exchanges=EXCHANGES,
    regex_pattern=REGEX_PATTERN,
    tokenizer=TweetTokenizer()
)

__all__ = ['StandardNormalizer']
