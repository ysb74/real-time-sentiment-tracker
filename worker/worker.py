"""
Sentiment Analysis Worker

This module consumes social media posts from RabbitMQ, performs sentiment analysis
using LLMs, and stores results in Redis for real-time dashboard consumption.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import pika
import redis
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import structlog

from .llm_service import LLMSentimentService, SentimentResult

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

from multimodal.image import ImageSentimentAnalyzer
from multimodal.audio import AudioSentimentAnalyzer
from multimodal.fusion import fuse_modalities

# Inside your worker's processing routine, add:
def process_post(post):
    results = {}
    # Assume post dict has keys: text, image_url, audio_url
    if post.get("text"):
        results["text"] = analyze_text_sentiment(post["text"])  # Use existing LLM service
    if post.get("image_url"):
        img_analyzer = ImageSentimentAnalyzer()
        results["image"] = img_analyzer.analyze(post["image_url"])
    if post.get("audio_url"):
        aud_analyzer = AudioSentimentAnalyzer()
        results["audio"] = aud_analyzer.analyze(post["audio_url"])
    fused = fuse_modalities(results)
    return fused, results


class Settings(BaseSettings):
    """Worker settings"""
    rabbitmq_url: str = "amqp://admin:password@localhost:5672/"
    redis_url: str = "redis://localhost:6379/0"
    
    # LLM API keys
    openai_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    
    # Processing configuration
    max_concurrent_tasks: int = 10
    batch_size: int = 5
    processing_timeout: int = 30
    
    # Data retention
    sentiment_data_retention_hours: int = 24
    
    class Config:
        env_file = ".env"


class ProcessedPost(BaseModel):
    """Processed social media post with sentiment"""
    # Original post data
    id: str
    platform: str
    content: str
    author: str
    created_at: datetime
    url: Optional[str] = None
    metrics: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    
    # Sentiment analysis results
    sentiment_label: str
    sentiment_confidence: float
    sentiment_reasoning: Optional[str] = None
    model_used: str
    processing_time: float
    processed_at: datetime


class SentimentWorker:
    """Worker that processes social media posts for sentiment analysis"""
    
    def __init__(self):
        self.settings = Settings()
        self.running = False
        self.connection = None
        self.channel = None
        self.redis_client = redis.from_url(self.settings.redis_url)
        
        # Initialize LLM service
        self.llm_service = LLMSentimentService(
            openai_api_key=self.settings.openai_api_key,
            huggingface_api_key=self.settings.huggingface_api_key
        )
        
        # Processing statistics
        self.stats = {
            "processed_count": 0,
            "error_count": 0,
            "start_time": time.time(),
            "last_processed": None
        }
        
    def setup_rabbitmq_connection(self):
        """Setup RabbitMQ connection and channel"""
        try:
            self.connection = pika.BlockingConnection(pika.URLParameters(self.settings.rabbitmq_url))
            self.channel = self.connection.channel()
            
            # Ensure queue exists
            self.channel.queue_declare(queue='sentiment_analysis', durable=True)
            
            # Set QoS to process messages one at a time for load balancing
            self.channel.basic_qos(prefetch_count=1)
            
            logger.info("RabbitMQ connection established")
            
        except Exception as e:
            logger.error("Failed to connect to RabbitMQ", error=str(e))
            raise
    
    async def process_message(self, ch, method, properties, body):
        """Process a single message from the queue"""
        try:
            # Parse message
            message_data = json.loads(body.decode('utf-8'))
            post_id = message_data.get('id', 'unknown')
            
            logger.info("Processing message", post_id=post_id, platform=message_data.get('platform'))
            
            # Extract content for sentiment analysis
            content = message_data.get('content', '')
            if not content or len(content.strip()) < 5:
                logger.warning("Skipping message with insufficient content", post_id=post_id)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            
            # Perform sentiment analysis
            start_time = time.time()
            sentiment_result = await self.llm_service.analyze_sentiment(content)
            
            # Create processed post object
            processed_post = ProcessedPost(
                # Original data
                id=message_data['id'],
                platform=message_data['platform'],
                content=content,
                author=message_data.get('author', 'unknown'),
                created_at=datetime.fromisoformat(message_data['created_at'].replace('Z', '+00:00')),
                url=message_data.get('url'),
                metrics=message_data.get('metrics', {}),
                metadata=message_data.get('metadata', {}),
                
                # Sentiment data
                sentiment_label=sentiment_result.label.value,
                sentiment_confidence=sentiment_result.confidence,
                sentiment_reasoning=sentiment_result.reasoning,
                model_used=sentiment_result.model_used,
                processing_time=sentiment_result.processing_time,
                processed_at=datetime.utcnow()
            )
            
            # Store results in Redis
            await self.store_sentiment_result(processed_post)
            
            # Update statistics
            self.stats["processed_count"] += 1
            self.stats["last_processed"] = time.time()
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
            logger.info("Message processed successfully", 
                       post_id=post_id,
                       sentiment=sentiment_result.label.value,
                       confidence=sentiment_result.confidence,
                       processing_time=sentiment_result.processing_time)
            
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in message", error=str(e))
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)  # Don't requeue invalid messages
            self.stats["error_count"] += 1
            
        except Exception as e:
            logger.error("Error processing message", error=str(e))
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)  # Requeue for retry
            self.stats["error_count"] += 1
    
    async def store_sentiment_result(self, processed_post: ProcessedPost):
        """Store sentiment analysis result in Redis"""
        try:
            # Store individual post data
            post_key = f"sentiment:post:{processed_post.id}"
            post_data = processed_post.model_dump(mode='json')
            
            # Convert datetime objects to ISO strings for JSON serialization
            for key, value in post_data.items():
                if isinstance(value, datetime):
                    post_data[key] = value.isoformat()
            
            # Store with expiration
            expiration_seconds = self.settings.sentiment_data_retention_hours * 3600
            self.redis_client.setex(post_key, expiration_seconds, json.dumps(post_data))
            
            # Add to platform-specific sorted sets for time-based queries
            platform_key = f"sentiment:platform:{processed_post.platform}"
            timestamp = processed_post.processed_at.timestamp()
            self.redis_client.zadd(platform_key, {processed_post.id: timestamp})
            
            # Add to sentiment-specific sorted sets
            sentiment_key = f"sentiment:label:{processed_post.sentiment_label}"
            self.redis_client.zadd(sentiment_key, {processed_post.id: timestamp})
            
            # Update real-time statistics
            current_hour = datetime.utcnow().strftime("%Y-%m-%d:%H")
            stats_key = f"sentiment:stats:{current_hour}"
            
            pipe = self.redis_client.pipeline()
            pipe.hincrby(stats_key, f"total", 1)
            pipe.hincrby(stats_key, f"platform:{processed_post.platform}", 1)
            pipe.hincrby(stats_key, f"sentiment:{processed_post.sentiment_label}", 1)
            pipe.expire(stats_key, expiration_seconds)
            pipe.execute()
            
            # Update recent posts list for dashboard
            recent_key = "sentiment:recent"
            self.redis_client.lpush(recent_key, processed_post.id)
            self.redis_client.ltrim(recent_key, 0, 99)  # Keep only last 100 posts
            
            logger.debug("Sentiment result stored", 
                        post_id=processed_post.id,
                        platform=processed_post.platform,
                        sentiment=processed_post.sentiment_label)
            
        except Exception as e:
            logger.error("Failed to store sentiment result", error=str(e), post_id=processed_post.id)
    
    def cleanup_expired_data(self):
        """Clean up expired data from Redis"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=self.settings.sentiment_data_retention_hours)
            cutoff_timestamp = cutoff_time.timestamp()
            
            # Clean up sorted sets
            for key_pattern in ["sentiment:platform:*", "sentiment:label:*"]:
                for key in self.redis_client.scan_iter(match=key_pattern):
                    self.redis_client.zremrangebyscore(key, 0, cutoff_timestamp)
            
            logger.debug("Expired data cleanup completed", cutoff_time=cutoff_time.isoformat())
            
        except Exception as e:
            logger.error("Error during data cleanup", error=str(e))
    
    def update_worker_status(self):
        """Update worker status in Redis"""
        try:
            status_data = {
                "worker_id": os.getenv("HOSTNAME", "worker-1"),
                "status": "running" if self.running else "stopped",
                "processed_count": self.stats["processed_count"],
                "error_count": self.stats["error_count"],
                "uptime": time.time() - self.stats["start_time"],
                "last_seen": datetime.utcnow().isoformat(),
                "available_providers": self.llm_service.get_available_providers()
            }
            
            worker_key = f"worker:status:{status_data['worker_id']}"
            self.redis_client.hset(worker_key, mapping=status_data)
            self.redis_client.expire(worker_key, 300)  # 5 minutes expiration
            
        except Exception as e:
            logger.error("Failed to update worker status", error=str(e))
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info("Shutdown signal received", signal=signum)
            self.running = False
            if self.connection and not self.connection.is_closed:
                self.connection.close()
                
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def start_processing(self):
        """Start processing messages from the queue"""
        self.running = True
        self.setup_signal_handlers()
        
        logger.info("Starting sentiment analysis worker",
                   available_providers=self.llm_service.get_available_providers())
        
        # Setup RabbitMQ connection
        self.setup_rabbitmq_connection()
        
        # Create async wrapper for message processing
        def callback_wrapper(ch, method, properties, body):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.process_message(ch, method, properties, body))
            finally:
                loop.close()
        
        # Start consuming messages
        self.channel.basic_consume(
            queue='sentiment_analysis',
            on_message_callback=callback_wrapper
        )
        
        logger.info("Worker ready, waiting for messages")
        
        # Main processing loop
        last_cleanup = time.time()
        last_status_update = time.time()
        
        try:
            while self.running:
                # Process messages (non-blocking)
                self.connection.process_data_events(time_limit=1)
                
                # Periodic maintenance tasks
                current_time = time.time()
                
                # Cleanup expired data every hour
                if current_time - last_cleanup > 3600:
                    self.cleanup_expired_data()
                    last_cleanup = current_time
                
                # Update worker status every 30 seconds
                if current_time - last_status_update > 30:
                    self.update_worker_status()
                    last_status_update = current_time
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error("Worker error", error=str(e))
        finally:
            # Cleanup
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            
            self.update_worker_status()
            logger.info("Worker stopped", 
                       processed_count=self.stats["processed_count"],
                       error_count=self.stats["error_count"])


async def main():
    """Main entry point"""
    worker = SentimentWorker()
    await worker.start_processing()


if __name__ == "__main__":
    asyncio.run(main())
