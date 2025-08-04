"""
LLM Sentiment Analysis Service

This module provides sentiment analysis capabilities using various LLM providers
including OpenAI GPT models and Hugging Face transformers.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
import asyncio

import openai
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()


class SentimentLabel(str, Enum):
    """Sentiment classification labels"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    label: SentimentLabel
    confidence: float
    reasoning: Optional[str] = None
    processing_time: float = 0.0
    model_used: str = ""


class BaseLLMProvider(ABC):
    """Abstract base class for LLM sentiment analysis providers"""
    
    @abstractmethod
    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment of given text"""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT-based sentiment analysis"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        
    def get_provider_name(self) -> str:
        return f"openai-{self.model}"
    
    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment using OpenAI GPT"""
        start_time = time.time()
        
        try:
            # Construct the prompt for sentiment analysis
            system_prompt = """You are an expert sentiment analyzer. Analyze the sentiment of the given text and respond with ONLY a JSON object in this exact format:

{
    "sentiment": "positive" | "negative" | "neutral",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}

Guidelines:
- positive: clearly expresses happiness, satisfaction, excitement, or approval
- negative: clearly expresses dissatisfaction, anger, sadness, or disapproval  
- neutral: factual, objective, or mixed emotions that don't lean clearly positive or negative
- confidence: how certain you are (0.0 = not certain, 1.0 = very certain)
- reasoning: 1-2 sentences explaining your classification"""

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this text: {text[:1000]}"}  # Limit text length
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=200
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            try:
                # Find JSON object in response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError):
                # Fallback parsing
                response_lower = response_text.lower()
                if "positive" in response_lower:
                    sentiment = SentimentLabel.POSITIVE
                elif "negative" in response_lower:
                    sentiment = SentimentLabel.NEGATIVE
                else:
                    sentiment = SentimentLabel.NEUTRAL
                
                result = {
                    "sentiment": sentiment.value,
                    "confidence": 0.7,
                    "reasoning": "Fallback classification based on keyword detection"
                }
            
            processing_time = time.time() - start_time
            
            return SentimentResult(
                label=SentimentLabel(result["sentiment"]),
                confidence=float(result["confidence"]),
                reasoning=result.get("reasoning", ""),
                processing_time=processing_time,
                model_used=self.get_provider_name()
            )
            
        except Exception as e:
            logger.error("OpenAI sentiment analysis error", error=str(e), text_length=len(text))
            
            # Return neutral sentiment as fallback
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                reasoning=f"Error in analysis: {str(e)}",
                processing_time=time.time() - start_time,
                model_used=self.get_provider_name()
            )


class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face transformer-based sentiment analysis"""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        
        try:
            # Initialize the sentiment analysis pipeline
            self.pipeline = pipeline(
                "sentiment-analysis", 
                model=model_name, 
                device=self.device,
                return_all_scores=True
            )
            logger.info("Hugging Face model loaded", model=model_name, device=self.device)
            
        except Exception as e:
            logger.error("Failed to load Hugging Face model", error=str(e), model=model_name)
            raise
    
    def get_provider_name(self) -> str:
        return f"huggingface-{self.model_name.split('/')[-1]}"
    
    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment using Hugging Face transformers"""
        start_time = time.time()
        
        try:
            # Truncate text to model's max length (typically 512 tokens)
            text = text[:500]  # Conservative truncation
            
            # Run inference
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self.pipeline, text)
            
            # Parse results - results[0] contains all scores
            all_scores = results[0]
            
            # Find the highest confidence prediction
            best_result = max(all_scores, key=lambda x: x['score'])
            
            # Map model labels to our standard labels
            label_mapping = {
                'POSITIVE': SentimentLabel.POSITIVE,
                'NEGATIVE': SentimentLabel.NEGATIVE,
                'NEUTRAL': SentimentLabel.NEUTRAL,
                'positive': SentimentLabel.POSITIVE,
                'negative': SentimentLabel.NEGATIVE,
                'neutral': SentimentLabel.NEUTRAL,
                'LABEL_0': SentimentLabel.NEGATIVE,  # RoBERTa specific
                'LABEL_1': SentimentLabel.NEUTRAL,
                'LABEL_2': SentimentLabel.POSITIVE,
            }
            
            raw_label = best_result['label']
            sentiment_label = label_mapping.get(raw_label, SentimentLabel.NEUTRAL)
            confidence = float(best_result['score'])
            
            processing_time = time.time() - start_time
            
            return SentimentResult(
                label=sentiment_label,
                confidence=confidence,
                reasoning=f"Model prediction: {raw_label} with {confidence:.3f} confidence",
                processing_time=processing_time,
                model_used=self.get_provider_name()
            )
            
        except Exception as e:
            logger.error("Hugging Face sentiment analysis error", error=str(e), text_length=len(text))
            
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                reasoning=f"Error in analysis: {str(e)}",
                processing_time=time.time() - start_time,
                model_used=self.get_provider_name()
            )


class LLMSentimentService:
    """Main service for LLM-based sentiment analysis with multiple providers"""
    
    def __init__(self, openai_api_key: Optional[str] = None, huggingface_api_key: Optional[str] = None):
        self.providers: List[BaseLLMProvider] = []
        self.primary_provider: Optional[BaseLLMProvider] = None
        
        # Initialize OpenAI provider
        if openai_api_key:
            try:
                openai_provider = OpenAIProvider(openai_api_key)
                self.providers.append(openai_provider)
                if not self.primary_provider:
                    self.primary_provider = openai_provider
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.error("Failed to initialize OpenAI provider", error=str(e))
        
        # Initialize Hugging Face provider
        try:
            hf_provider = HuggingFaceProvider()
            self.providers.append(hf_provider)
            if not self.primary_provider:
                self.primary_provider = hf_provider
            logger.info("Hugging Face provider initialized")
        except Exception as e:
            logger.error("Failed to initialize Hugging Face provider", error=str(e))
        
        if not self.providers:
            raise ValueError("No sentiment analysis providers available")
    
    async def analyze_sentiment(self, text: str, provider_name: Optional[str] = None) -> SentimentResult:
        """
        Analyze sentiment using specified provider or primary provider
        
        Args:
            text: Text to analyze
            provider_name: Specific provider to use (optional)
            
        Returns:
            SentimentResult with analysis
        """
        if not text or not text.strip():
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                reasoning="Empty text provided",
                processing_time=0.0,
                model_used="none"
            )
        
        # Select provider
        provider = self.primary_provider
        if provider_name:
            # Find specific provider
            for p in self.providers:
                if provider_name in p.get_provider_name():
                    provider = p
                    break
        
        if not provider:
            raise ValueError("No suitable provider found")
        
        try:
            result = await provider.analyze_sentiment(text)
            logger.debug("Sentiment analysis completed", 
                        provider=provider.get_provider_name(),
                        sentiment=result.label.value,
                        confidence=result.confidence,
                        processing_time=result.processing_time)
            return result
            
        except Exception as e:
            logger.error("Sentiment analysis failed", 
                        provider=provider.get_provider_name(),
                        error=str(e))
            
            # Try fallback provider if available
            for fallback_provider in self.providers:
                if fallback_provider != provider:
                    try:
                        logger.info("Trying fallback provider", 
                                   fallback=fallback_provider.get_provider_name())
                        return await fallback_provider.analyze_sentiment(text)
                    except Exception as fallback_error:
                        logger.error("Fallback provider failed", 
                                   provider=fallback_provider.get_provider_name(),
                                   error=str(fallback_error))
            
            # All providers failed
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                reasoning=f"All providers failed: {str(e)}",
                processing_time=0.0,
                model_used="error"
            )
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return [provider.get_provider_name() for provider in self.providers]
    
    async def batch_analyze(self, texts: List[str], provider_name: Optional[str] = None) -> List[SentimentResult]:
        """
        Analyze multiple texts in batch
        
        Args:
            texts: List of texts to analyze
            provider_name: Specific provider to use (optional)
            
        Returns:
            List of SentimentResult objects
        """
        tasks = [self.analyze_sentiment(text, provider_name) for text in texts]
        return await asyncio.gather(*tasks)
