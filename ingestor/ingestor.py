"""
Real-time Social Media Data Ingestor

This module streams data from social media APIs (Twitter/X, Reddit) and publishes
messages to RabbitMQ for sentiment analysis processing.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import signal
import sys

import pika
import redis
import tweepy
import praw
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    rabbitmq_url: str = "amqp://admin:password@localhost:5672/"
    redis_url: str = "redis://localhost:6379/0"
    
    # Twitter/X API credentials
    twitter_bearer_token: Optional[str] = None
    
    # Reddit API credentials
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: str = "SentimentTracker/1.0"
    
    # Streaming configuration
    max_tweets_per_minute: int = 100
    max_reddit_posts_per_minute: int = 50
    
    # Keywords to track
    tracking_keywords: List[str] = ["AI", "technology", "python", "programming"]
    
    class Config:
        env_file = ".env"


class SocialMediaPost(BaseModel):
    """Data model for social media posts"""
    id: str
    platform: str  # "twitter" or "reddit"
    content: str
    author: str
    created_at: datetime
    url: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TwitterStreamListener(tweepy.StreamingClient):
    """Custom Twitter streaming client"""
    
    def __init__(self, bearer_token: str, message_publisher, **kwargs):
        super().__init__(bearer_token, **kwargs)
        self.message_publisher = message_publisher
        self.posts_count = 0
        self.start_time = time.time()
        
    def on_tweet(self, tweet):
        """Handle incoming tweets"""
        try:
            # Rate limiting check
            current_time = time.time()
            if current_time - self.start_time >= 60:  # Reset counter every minute
                self.posts_count = 0
                self.start_time = current_time
                
            if self.posts_count >= 100:  # Max 100 tweets per minute
                return
                
            # Create structured post data
            post = SocialMediaPost(
                id=str(tweet.id),
                platform="twitter",
                content=tweet.text,
                author=tweet.author_id or "unknown",
                created_at=tweet.created_at or datetime.utcnow(),
                url=f"https://twitter.com/i/status/{tweet.id}",
                metrics={
                    "retweet_count": getattr(tweet, 'retweet_count', 0),
                    "like_count": getattr(tweet, 'like_count', 0),
                    "reply_count": getattr(tweet, 'reply_count', 0),
                },
                metadata={
                    "lang": getattr(tweet, 'lang', 'en'),
                    "possibly_sensitive": getattr(tweet, 'possibly_sensitive', False),
                }
            )
            
            # Publish to message queue
            self.message_publisher.publish_message(post.model_dump())
            self.posts_count += 1
            
            logger.info("Tweet processed", 
                       tweet_id=tweet.id, 
                       author=post.author, 
                       content_length=len(post.content))
            
        except Exception as e:
            logger.error("Error processing tweet", error=str(e), tweet_id=getattr(tweet, 'id', 'unknown'))

    def on_errors(self, errors):
        """Handle stream errors"""
        logger.error("Twitter stream error", errors=errors)
        return True  # Continue streaming


class MessagePublisher:
    """Handles publishing messages to RabbitMQ"""
    
    def __init__(self, rabbitmq_url: str):
        self.rabbitmq_url = rabbitmq_url
        self.connection = None
        self.channel = None
        self.connect()
        
    def connect(self):
        """Establish connection to RabbitMQ"""
        try:
            self.connection = pika.BlockingConnection(pika.URLParameters(self.rabbitmq_url))
            self.channel = self.connection.channel()
            
            # Declare exchanges and queues
            self.channel.exchange_declare(exchange='social_media', exchange_type='topic', durable=True)
            self.channel.queue_declare(queue='sentiment_analysis', durable=True)
            self.channel.queue_bind(exchange='social_media', queue='sentiment_analysis', routing_key='post.*')
            
            logger.info("Connected to RabbitMQ")
            
        except Exception as e:
            logger.error("Failed to connect to RabbitMQ", error=str(e))
            raise
            
    def publish_message(self, post_data: Dict[str, Any]):
        """Publish post data to message queue"""
        try:
            if not self.connection or self.connection.is_closed:
                self.connect()
                
            message = json.dumps(post_data, default=str)
            routing_key = f"post.{post_data['platform']}"
            
            self.channel.basic_publish(
                exchange='social_media',
                routing_key=routing_key,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    timestamp=int(time.time()),
                    content_type='application/json'
                )
            )
            
            logger.debug("Message published", 
                        platform=post_data['platform'], 
                        post_id=post_data['id'])
            
        except Exception as e:
            logger.error("Failed to publish message", error=str(e))
            self.connect()  # Reconnect on failure
            
    def close(self):
        """Close RabbitMQ connection"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("RabbitMQ connection closed")


class SocialMediaIngestor:
    """Main ingestion orchestrator"""
    
    def __init__(self):
        self.settings = Settings()
        self.message_publisher = MessagePublisher(self.settings.rabbitmq_url)
        self.redis_client = redis.from_url(self.settings.redis_url)
        self.running = False
        
        # Initialize API clients
        self.twitter_client = None
        self.reddit_client = None
        self.setup_api_clients()
        
    def setup_api_clients(self):
        """Initialize social media API clients"""
        # Twitter/X client
        if self.settings.twitter_bearer_token:
            try:
                self.twitter_client = TwitterStreamListener(
                    bearer_token=self.settings.twitter_bearer_token,
                    message_publisher=self.message_publisher
                )
                logger.info("Twitter client initialized")
            except Exception as e:
                logger.error("Failed to initialize Twitter client", error=str(e))
        
        # Reddit client
        if self.settings.reddit_client_id and self.settings.reddit_client_secret:
            try:
                self.reddit_client = praw.Reddit(
                    client_id=self.settings.reddit_client_id,
                    client_secret=self.settings.reddit_client_secret,
                    user_agent=self.settings.reddit_user_agent
                )
                logger.info("Reddit client initialized")
            except Exception as e:
                logger.error("Failed to initialize Reddit client", error=str(e))
    
    async def stream_twitter_data(self):
        """Stream data from Twitter/X"""
        if not self.twitter_client:
            logger.warning("Twitter client not available")
            return
            
        try:
            # Delete existing rules
            rules = self.twitter_client.get_rules()
            if rules.data:
                rule_ids = [rule.id for rule in rules.data]
                self.twitter_client.delete_rules(rule_ids)
            
            # Add new rules for keywords
            rules = []
            for keyword in self.settings.tracking_keywords:
                rules.append(tweepy.StreamRule(f"{keyword} lang:en -is:retweet"))
            
            self.twitter_client.add_rules(rules)
            logger.info("Twitter streaming rules added", keywords=self.settings.tracking_keywords)
            
            # Start streaming
            self.twitter_client.filter(threaded=True)
            
        except Exception as e:
            logger.error("Twitter streaming error", error=str(e))
    
    async def stream_reddit_data(self):
        """Stream data from Reddit"""
        if not self.reddit_client:
            logger.warning("Reddit client not available")
            return
            
        try:
            # Monitor multiple subreddits for new posts
            subreddits = ["technology", "programming", "MachineLearning", "artificial"]
            subreddit = self.reddit_client.subreddit("+".join(subreddits))
            
            post_count = 0
            start_time = time.time()
            
            # Stream new submissions
            for submission in subreddit.stream.submissions(skip_existing=True):
                if not self.running:
                    break
                    
                # Rate limiting
                current_time = time.time()
                if current_time - start_time >= 60:
                    post_count = 0
                    start_time = current_time
                    
                if post_count >= self.settings.max_reddit_posts_per_minute:
                    await asyncio.sleep(1)
                    continue
                
                # Filter by keywords
                content = f"{submission.title} {submission.selftext}"
                if not any(keyword.lower() in content.lower() for keyword in self.settings.tracking_keywords):
                    continue
                
                # Create post object
                post = SocialMediaPost(
                    id=str(submission.id),
                    platform="reddit",
                    content=content,
                    author=str(submission.author) if submission.author else "unknown",
                    created_at=datetime.fromtimestamp(submission.created_utc),
                    url=f"https://reddit.com{submission.permalink}",
                    metrics={
                        "score": submission.score,
                        "upvote_ratio": submission.upvote_ratio,
                        "num_comments": submission.num_comments,
                    },
                    metadata={
                        "subreddit": str(submission.subreddit),
                        "is_self": submission.is_self,
                        "over_18": submission.over_18,
                    }
                )
                
                # Publish message
                self.message_publisher.publish_message(post.model_dump())
                post_count += 1
                
                logger.info("Reddit post processed", 
                           post_id=submission.id, 
                           subreddit=str(submission.subreddit),
                           content_length=len(content))
                
                await asyncio.sleep(0.1)  # Small delay to prevent overwhelming
                
        except Exception as e:
            logger.error("Reddit streaming error", error=str(e))
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info("Shutdown signal received", signal=signum)
            self.running = False
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def start_ingestion(self):
        """Start the data ingestion process"""
        self.running = True
        self.setup_signal_handlers()
        
        logger.info("Starting social media data ingestion", 
                   keywords=self.settings.tracking_keywords)
        
        # Store ingestion status in Redis
        self.redis_client.hset("ingestor:status", mapping={
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
            "keywords": json.dumps(self.settings.tracking_keywords)
        })
        
        try:
            # Start streaming tasks
            tasks = []
            
            if self.twitter_client:
                tasks.append(asyncio.create_task(self.stream_twitter_data()))
                
            if self.reddit_client:
                tasks.append(asyncio.create_task(self.stream_reddit_data()))
            
            if not tasks:
                logger.error("No API clients available - check configuration")
                return
            
            # Run until shutdown
            while self.running:
                await asyncio.sleep(1)
                
                # Update status
                self.redis_client.hset("ingestor:status", "last_seen", datetime.utcnow().isoformat())
                
        except Exception as e:
            logger.error("Ingestion error", error=str(e))
        finally:
            # Cleanup
            if self.twitter_client:
                self.twitter_client.disconnect()
            
            self.message_publisher.close()
            
            # Update status
            self.redis_client.hset("ingestor:status", mapping={
                "status": "stopped",
                "stopped_at": datetime.utcnow().isoformat()
            })
            
            logger.info("Data ingestion stopped")


async def main():
    """Main entry point"""
    ingestor = SocialMediaIngestor()
    await ingestor.start_ingestion()


if __name__ == "__main__":
    asyncio.run(main())
