#!/usr/bin/env python3
"""
LLM Service Module for Sentiment Analysis

This module provides an abstraction layer for interacting with Google's Gemini API
to perform sentiment analysis on social media posts. It handles API authentication,
prompt construction, response parsing, and error handling.

The service is designed to be:
- Robust: Handles API errors and rate limits gracefully
- Efficient: Uses optimized prompts for quick, accurate results  
- Scalable: Can be easily extended to support other LLM providers
- Reliable: Includes retry logic and fallback responses

Author: Your Name
Date: 2024
"""

import os
import time
import logging
import google.generativeai as genai
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    A service class that handles sentiment analysis using Google's Gemini API.
    
    This class encapsulates all the logic needed to:
    1. Authenticate with the Gemini API
    2. Construct effective prompts for sentiment analysis
    3. Parse and validate API responses
    4. Handle errors and rate limits gracefully
    """
    
    def __init__(self):
        """
        Initialize the sentiment analyzer with API credentials and configuration.
        
        Sets up the Gemini API client and validates credentials.
        """
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        # Using gemini-pro for general text tasks
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Configuration for generation
        self.generation_config = {
            'temperature': 0.1,  # Low temperature for consistent, focused responses
            'top_p': 0.8,
            'top_k': 40,
            'max_output_tokens': 100,  # Keep responses short and focused
        }
        
        logger.info("Sentiment Analyzer initialized with Gemini API")
    
    def _construct_prompt(self, text: str) -> str:
        """
        Construct an optimized prompt for sentiment analysis.
        
        The prompt is carefully designed to:
        - Get consistent, parseable responses
        - Minimize token usage for cost efficiency
        - Provide clear instructions for the model
        - Handle edge cases and ambiguous text
        
        Args:
            text (str): The text to analyze for sentiment
            
        Returns:
            str: A well-structured prompt for the LLM
        """
        # Clean and prepare the input text
        cleaned_text = text.strip()[:800]  # Limit text length
        
        prompt = f"""Analyze the sentiment of this social media post and respond with ONLY the requested format.

Post: "{cleaned_text}"

Respond with exactly this format:
SENTIMENT: [positive/negative/neutral]
CONFIDENCE: [0-100]
REASON: [brief explanation in 10 words or less]

Rules:
- positive: optimistic, happy, excited, supportive content
- negative: angry, sad, frustrated, critical, pessimistic content  
- neutral: informational, questions, balanced, or unclear sentiment
- confidence: how certain you are (0=very uncertain, 100=very certain)
- reason: key words/phrases that influenced your decision

Example response:
SENTIMENT: positive
CONFIDENCE: 85
REASON: enthusiastic language and positive outcomes mentioned"""

        return prompt
    
    def _parse_response(self, response_text: str) -> Dict[str, any]:
        """
        Parse the LLM response into structured data.
        
        This method handles the parsing of the model's response and provides
        fallback values if the response format is unexpected.
        
        Args:
            response_text (str): Raw response from the LLM
            
        Returns:
            Dict: Parsed sentiment data with sentiment, confidence, and reason
        """
        try:
            # Initialize default values
            result = {
                'sentiment': 'neutral',
                'confidence': 50,
                'reason': 'unable to parse response'
            }
            
            # Parse line by line
            lines = response_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('SENTIMENT:'):
                    sentiment = line.replace('SENTIMENT:', '').strip().lower()
                    if sentiment in ['positive', 'negative', 'neutral']:
                        result['sentiment'] = sentiment
                
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = int(line.replace('CONFIDENCE:', '').strip())
                        # Ensure confidence is within valid range
                        result['confidence'] = max(0, min(100, confidence))
                    except ValueError:
                        logger.warning(f"Invalid confidence value in response: {line}")
                
                elif line.startswith('REASON:'):
                    reason = line.replace('REASON:', '').strip()
                    if reason and len(reason) > 0:
                        result['reason'] = reason[:100]  # Limit reason length
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            logger.debug(f"Response text was: {response_text}")
            
            # Return safe default values
            return {
                'sentiment': 'neutral',
                'confidence': 30,
                'reason': 'parsing error occurred'
            }
    
    def analyze_sentiment(self, text: str, max_retries: int = 3) -> Dict[str, any]:
        """
        Analyze the sentiment of given text using the Gemini API.
        
        This is the main public method that orchestrates the entire sentiment
        analysis process, including error handling and retries.
        
        Args:
            text (str): Text to analyze for sentiment
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            Dict: Analysis results containing sentiment, confidence, and reasoning
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for sentiment analysis")
            return {
                'sentiment': 'neutral',
                'confidence': 0,
                'reason': 'no text provided',
                'error': 'empty_input'
            }
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Construct the analysis prompt
                prompt = self._construct_prompt(text)
                
                # Make the API call with retry logic
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
                
                # Check if we got a valid response
                if not response.text:
                    raise ValueError("Empty response from Gemini API")
                
                # Parse the response
                result = self._parse_response(response.text)
                
                # Add metadata about the analysis
                result.update({
                    'analyzed_text_length': len(text),
                    'api_response_raw': response.text[:200],  # First 200 chars for debugging
                    'attempt_number': attempt + 1
                })
                
                logger.info(f"Sentiment analysis completed: {result['sentiment']} ({result['confidence']}%)")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Sentiment analysis attempt {attempt + 1} failed: {str(e)}")
                
                # If this isn't the last retry, wait before trying again
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        # If all retries failed, return error response
        logger.error(f"All sentiment analysis attempts failed. Last error: {str(last_error)}")
        return {
            'sentiment': 'neutral',
            'confidence': 0,
            'reason': 'analysis failed - api error',
            'error': str(last_error),
            'analyzed_text_length': len(text) if text else 0
        }
    
    def batch_analyze(self, texts: list, delay_between_calls: float = 0.5) -> list:
        """
        Analyze sentiment for multiple texts with rate limiting.
        
        This method processes multiple texts while respecting API rate limits
        and providing progress feedback.
        
        Args:
            texts (list): List of texts to analyze
            delay_between_calls (float): Seconds to wait between API calls
            
        Returns:
            list: List of sentiment analysis results
        """
        results = []
        
        logger.info(f"Starting batch sentiment analysis for {len(texts)} texts")
        
        for i, text in enumerate(texts):
            try:
                result = self.analyze_sentiment(text)
                results.append(result)
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(texts)} texts")
                
                # Rate limiting delay
                if delay_between_calls > 0 and i < len(texts) - 1:
                    time.sleep(delay_between_calls)
                    
            except Exception as e:
                logger.error(f"Error in batch processing text {i}: {str(e)}")
                results.append({
                    'sentiment': 'neutral',
                    'confidence': 0,
                    'reason': 'batch processing error',
                    'error': str(e)
                })
        
        logger.info(f"Completed batch analysis. Processed {len(results)} texts")
        return results
    
    def get_sentiment_summary(self, results: list) -> Dict[str, any]:
        """
        Generate a summary of sentiment analysis results.
        
        This method provides aggregate statistics about a collection of
        sentiment analysis results, useful for dashboard displays.
        
        Args:
            results (list): List of sentiment analysis result dictionaries
            
        Returns:
            Dict: Summary statistics including counts and averages
        """
        if not results:
            return {
                'total_analyzed': 0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'average_confidence': 0,
                'error_count': 0
            }
        
        # Count sentiments
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        confidence_scores = []
        error_count = 0
        
        for result in results:
            if 'error' in result:
                error_count += 1
                continue
                
            sentiment = result.get('sentiment', 'neutral')
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            
            confidence = result.get('confidence', 0)
            if isinstance(confidence, (int, float)) and 0 <= confidence <= 100:
                confidence_scores.append(confidence)
        
        # Calculate averages
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            'total_analyzed': len(results),
            'sentiment_distribution': sentiment_counts,
            'average_confidence': round(avg_confidence, 2),
            'error_count': error_count,
            'success_rate': round((len(results) - error_count) / len(results) * 100, 2) if results else 0
        }

# Convenience function for easy importing
def analyze_sentiment(text: str) -> Dict[str, any]:
    """
    Simple function interface for sentiment analysis.
    
    This function provides a simple way to analyze sentiment without
    needing to instantiate the SentimentAnalyzer class directly.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        Dict: Sentiment analysis result
    """
    analyzer = SentimentAnalyzer()
    return analyzer.analyze_sentiment(text)
                