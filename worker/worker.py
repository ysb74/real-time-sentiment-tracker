#!/usr/bin/env python3
"""
Real-time Social Media Sentiment Tracker - Worker Service

This service processes messages from the Redis queue, performs sentiment analysis
using the Gemini API through the llm_service module, and stores the results
for consumption by the dashboard service.

The worker acts as the processing engine of our pipeline, handling:
- Message queue consumption and management
- Sentiment analysis orchestration
- Result storage and caching
- Error handling and recovery

Author: Your Name
Date: 2024
"""

import os
import json
import time
import logging
import redis
from datetime import datetime, timedelta
from dotenv import load_dotenv
from llm_service import SentimentAnalyzer

# Load environment variables from .env file
load_dotenv()

# Configure logging for monitoring and debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentWorker:
    """
    A worker service that processes Reddit posts from a message queue
    and performs sentiment analysis using AI.
    
    This class handles:
    1. Redis queue consumption
    2. Message parsing and validation
    3. Sentiment analysis coordination
    4. Result storage and management
    5. Error handling and recovery
    """
    
    def __init__(self):
        """
        Initialize the sentiment worker with Redis connection and AI analyzer.
        
        Sets up all necessary connections and validates configuration.
        """
        # Load configuration from environment variables
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis_db = int(os.getenv('REDIS_DB', 0))
        
        # Processing configuration
        self.batch_size = int(os.getenv('WORKER_BATCH_SIZE', 1))
        self.processing_timeout = int(os.getenv('WORKER_TIMEOUT', 30))
        
        # Initialize connections
        self.redis_client = self._setup_redis_connection()
        self.sentiment_analyzer = self._setup_sentiment_analyzer()
        
        # Performance tracking
        self.processed_count = 0
        self.error_count = 0
        self.start_time = datetime.utcnow()
        
        logger.info("Sentiment Worker initialized successfully")
    
    def _setup_redis_connection(self):
        """
        Create and test the Redis connection for message queue operations.
        
        Returns:
            redis.Redis: Connected Redis client instance
        """
        try:
            redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True
            )
            
            # Test the connection
            redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
            return redis_client
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
    
    def _setup_sentiment_analyzer(self):
        """
        Initialize the sentiment analysis service.
        
        Returns:
            SentimentAnalyzer: Configured sentiment analyzer instance
        """
        try:
            analyzer = SentimentAnalyzer()
            logger.info("Sentiment analyzer initialized successfully")
            return analyzer
            
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {str(e)}")
            raise
    
    def _parse_message(self, message: str) -> dict:
        """
        Parse and validate a JSON message from the queue.
        
        This method handles message deserialization and validates that
        the message contains all required fields for processing.
        
        Args:
            message (str): JSON string from Redis queue
            
        Returns:
            dict: Parsed message data, or None if invalid
        """
        try:
            data = json.loads(message)
            
            # Validate required fields
            required_fields = ['id', 'title', 'content', 'subreddit']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                logger.warning(f"Message missing required fields: {missing_fields}")
                return None
            
            # Ensure content is not empty
            if not data.get('content', '').strip():
                logger.warning(f"Message {data.get('id', 'unknown')} has empty content")
                return None
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON message: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing message: {str(e)}")
            return None
    
    def _process_post(self, post_data: dict) -> dict:
        """
        Process a single post through sentiment analysis.
        
        This method coordinates the sentiment analysis process and
        combines the original post data with the analysis results.
        
        Args:
            post_data (dict): Original post data from ingestor
            
        Returns:
            dict: Combined post data with sentiment analysis results
        """
        try:
            # Extract text content for analysis
            content = post_data['content']
            
            # Perform sentiment analysis
            logger.info(f"Analyzing sentiment for post: {post_data['id']}")
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(content)
            
            # Combine original data with sentiment results  
            processed_data = {
                **post_data,  # Original post data
                'sentiment_analysis': sentiment_result,
                'processed_timestamp': datetime.utcnow().isoformat(),
                'processing_duration': sentiment_result.get('processing_time', 0)
            }
            
            # Add convenience fields for dashboard
            processed_data['sentiment'] = sentiment_result.get('sentiment', 'neutral')
            processed_data['confidence'] = sentiment_result.get('confidence', 0)
            processed_data['sentiment_reason'] = sentiment_result.get('reason', 'unknown')
            
            logger.info(f"Successfully processed post {post_data['id']}: "
                       f"{processed_data['sentiment']} ({processed_data['confidence']}%)")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing post {post_data.get('id', 'unknown')}: {str(e)}")
            
            # Return post with error information
            return {
                **post_data,
                'sentiment_analysis': {
                    'sentiment': 'neutral',
                    'confidence': 0,
                    'reason': 'processing error',
                    'error': str(e)
                },
                'sentiment': 'neutral',
                'confidence': 0,
                'sentiment_reason': 'processing error',
                'processed_timestamp': datetime.utcnow().isoformat(),
                'processing_error': True
            }
    
    def _store_result(self, processed_data: dict):
        """
        Store processed results in Redis for dashboard consumption.
        
        This method handles multiple storage patterns:
        1. Individual post storage with expiration
        2. Aggregate data for dashboard displays
        3. Recent posts list for real-time updates
        
        Args:
            processed_data (dict): Processed post with sentiment analysis
        """
        try:
            post_id = processed_data['id']
            
            # Store individual post data (expires after 24 hours)
            self.redis_client.hset(
                f"post:{post_id}",
                mapping=processed_data
            )
            self.redis_client.expire(f"post:{post_id}", 86400)  # 24 hours
            
            # Add to recent posts list (for dashboard table)
            recent_posts_data = {
                'id': post_id,
                'title': processed_data['title'][:100],  # Truncate long titles
                'subreddit': processed_data['subreddit'],
                'sentiment': processed_data['sentiment'],
                'confidence': processed_data['confidence'],
                'timestamp': processed_data['processed_timestamp'],
                'permalink': processed_data.get('permalink', ''),
                'score': processed_data.get('score', 0)
            }
            
            # Push to recent posts list and limit size
            self.redis_client.lpush('recent_posts', json.dumps(recent_posts_data))
            self.redis_client.ltrim('recent_posts', 0, 199)  # Keep last 200 posts
            
            # Update aggregate statistics
            self._update_statistics(processed_data)
            
            logger.debug(f"Stored results for post {post_id}")
            
        except Exception as e:
            logger.error(f"Failed to store results for post {processed_data.get('id', 'unknown')}: {str(e)}")
    
    def _update_statistics(self, processed_data: dict):
        """
        Update aggregate statistics for dashboard displays.
        
        This method maintains running totals and recent sentiment trends
        that are used by the dashboard for real-time visualization.
        
        Args:
            processed_data (dict): Processed post with sentiment analysis
        """
        try:
            sentiment = processed_data['sentiment']
            confidence = processed_data['confidence']
            timestamp = processed_data['processed_timestamp']
            
            # Increment sentiment counters
            self.redis_client.hincrby('sentiment_stats', f'{sentiment}_count', 1)
            self.redis_client.hincrby('sentiment_stats', 'total_count', 1)
            
            # Update confidence tracking
            current_avg = float(self.redis_client.hget('sentiment_stats', 'avg_confidence') or 0)
            total_count = int(self.redis_client.hget('sentiment_stats', 'total_count') or 1)
            
            # Calculate new rolling average
            new_avg = ((current_avg * (total_count - 1)) + confidence) / total_count
            self.redis_client.hset('sentiment_stats', 'avg_confidence', round(new_avg, 2))
            
            # Add to time series data for trend charts (last 24 hours)
            hour_key = datetime.fromisoformat(timestamp.replace('Z', '')).strftime('%Y-%m-%d-%H')
            self.redis_client.hincrby(f'hourly_sentiment:{hour_key}', sentiment, 1)
            self.redis_client.expire(f'hourly_sentiment:{hour_key}', 86400)  # 24 hour expiry
            
            # Update last processed timestamp
            self.redis_client.hset('sentiment_stats', 'last_updated', timestamp)
            
        except Exception as e:
            logger.error(f"Failed to update statistics: {str(e)}")
    
    def _get_queue_message(self, timeout: int = 1) -> str:
        """
        Get the next message from the Redis queue with timeout.
        
        This method handles queue operations with proper timeout handling
        to prevent the worker from blocking indefinitely.
        
        Args:
            timeout (int): Timeout in seconds for queue operations
            
        Returns:
            str: Message from queue, or None if timeout/empty
        """
        try:
            # Use BRPOP for blocking pop with timeout
            result = self.redis_client.brpop('reddit_posts', timeout=timeout)
            
            if result:
                queue_name, message = result
                return message
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting message from queue: {str(e)}")
            return None
    
    def _log_performance_stats(self):
        """
        Log performance statistics for monitoring.
        
        This method provides periodic performance updates that help
        with monitoring and optimization of the worker service.
        """
        uptime = datetime.utcnow() - self.start_time
        rate = self.processed_count / uptime.total_seconds() if uptime.total_seconds() > 0 else 0
        error_rate = (self.error_count / self.processed_count * 100) if self.processed_count > 0 else 0
        
        logger.info(f"Performance Stats - Processed: {self.processed_count}, "
                   f"Errors: {self.error_count}, Rate: {rate:.2f}/sec, "
                   f"Error Rate: {error_rate:.1f}%, Uptime: {uptime}")
    
    def start_processing(self):
        """
        Start the main processing loop.
        
        This is the main method that runs continuously, processing messages
        from the queue and coordinating sentiment analysis. It includes
        comprehensive error handling and performance monitoring.
        """
        logger.info("Starting sentiment processing worker...")
        
        last_stats_log = datetime.utcnow()
        stats_interval = timedelta(minutes=5)  # Log stats every 5 minutes
        
        while True:
            try:
                # Get message from queue with timeout
                message = self._get_queue_message(timeout=5)
                
                if message:
                    # Parse the message
                    post_data = self._parse_message(message)
                    
                    if post_data:
                        # Process the post
                        start_time = time.time()
                        processed_data = self._process_post(post_data)
                        processing_time = time.time() - start_time
                        
                        # Store the results
                        self._store_result(processed_data)
                        
                        # Update counters
                        self.processed_count += 1
                        
                        if processed_data.get('processing_error'):
                            self.error_count += 1
                        
                        logger.debug(f"Processed post in {processing_time:.2f}s")
                        
                    else:
                        logger.warning("Skipped invalid message")
                        self.error_count += 1
                
                else:
                    # No message received (timeout) - this is normal
                    logger.debug("No messages in queue, waiting...")
                
                # Periodic performance logging
                if datetime.utcnow() - last_stats_log > stats_interval:
                    self._log_performance_stats()
                    last_stats_log = datetime.utcnow()
                    
            except KeyboardInterrupt:
                logger.info("Worker stopped by user")
                break
                
            except Exception as e:
                logger.error(f"Unexpected error in processing loop: {str(e)}")
                self.error_count += 1
                
                # Brief pause to prevent tight error loops
                time.sleep(5)
                
                # Attempt to reconnect services
                try:
                    self.redis_client = self._setup_redis_connection()
                    self.sentiment_analyzer = self._setup_sentiment_analyzer()
                except Exception as reconnect_error:
                    logger.error(f"Reconnection failed: {str(reconnect_error)}")
                    time.sleep(30)  # Longer pause on reconnection failure
        
        # Final performance stats
        self._log_performance_stats()
        logger.info("Sentiment processing worker stopped")

def main():
    """
    Main entry point for the worker service.
    
    This function initializes the worker and starts the processing loop.
    It's designed to run continuously as a service.
    """
    try:
        logger.info("Starting Sentiment Worker Service...")
        worker = SentimentWorker()
        worker.start_processing()
        
    except KeyboardInterrupt:
        logger.info("Worker service stopped by user")
    except Exception as e:
        logger.error(f"Fatal error in worker service: {str(e)}")
        raise

if __name__ == "__main__":
    main()