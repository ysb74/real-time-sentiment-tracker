"""
Social Media Data Ingestor Module

This module handles real-time data ingestion from various social media platforms
including Twitter/X and Reddit. It uses streaming APIs to collect posts and comments
and publishes them to a message queue for processing.
"""

__version__ = "1.0.0"
__author__ = "Sentiment Tracker Team"

from .ingestor import SocialMediaIngestor

__all__ = ["SocialMediaIngestor"]
