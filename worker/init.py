"""
Sentiment Analysis Worker Module

This module processes social media posts from the message queue and performs
sentiment analysis using Large Language Models (LLMs) including OpenAI GPT
and Hugging Face transformers.
"""

__version__ = "1.0.0"
__author__ = "Sentiment Tracker Team"

from .worker import SentimentWorker
from .llm_service import LLMSentimentService

__all__ = ["SentimentWorker", "LLMSentimentService"]
