#!/usr/bin/env python3
"""
Real-time Social Media Sentiment Tracker - Data Ingestor Service

This service connects to Reddit's API using PRAW (Python Reddit API Wrapper) 
and streams live posts from specified subreddits. Each post is processed and 
sent to a Redis message queue for further analysis by worker services.

The ingestor acts as the entry point for our data pipeline, handling:
- Reddit API authentication and connection management
- Real-time post streaming with error handling
- Data extraction and formatting
- Message queue publishing with proper serialization

Author: Your Name
Date: 2024
"""

import os
import json
import time
import logging
import praw
import redis
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging to help with debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RedditIngestor:
    """
    A service class that handles Reddit API connections and streams posts 
    to a Redis message queue for processing.
    
    This class encapsulates all the logic needed to:
    1. Authenticate with Reddit's API
    2. Monitor specified subreddits for new posts
    3. Extract relevant data from posts
    4. Publish formatted data to Redis queue
    """
    
    def __init__(self):
        """
        Initialize the Reddit ingestor with API credentials and Redis connection.
        
        This constructor sets up the necessary connections and validates that
        all required environment variables are present.
        """
        # Load configuration from environment variables
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'SentimentTracker/1.0')
        self.subreddit_name = os.getenv('SUBREDDIT', 'python')
        
        # Redis connection parameters
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis_db = int(os.getenv('REDIS_DB', 0))
        
        # Validate that we have all required credentials
        self._validate_credentials()
        
        # Initialize connections
        self.reddit = self._setup_reddit_connection()
        self.redis_client = self._setup_redis_connection()
        
        logger.info(f"Reddit Ingestor initialized for subreddit: r/{self.subreddit_name}")
    
    def _validate_credentials(self):
        """
        Validate that all required environment variables are set.
        
        This method checks for the presence of critical configuration values
        and raises informative errors if any are missing.
        """
        required_vars = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _setup_reddit_connection(self):
        """
        Create and configure the Reddit API connection using PRAW.
        
        Returns:
            praw.Reddit: Configured Reddit instance ready for API calls
        """
        try:
            reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent=self.reddit_user_agent
            )
            
            # Test the connection by getting basic info
            # This will raise an exception if credentials are invalid
            logger.info(f"Connected to Reddit API. Read-only: {reddit.read_only}")
            return reddit
            
        except Exception as e:
            logger.error(f"Failed to connect to Reddit API: {str(e)}")
            raise
    
    def _setup_redis_connection(self):
        """
        Create and test the Redis connection for message queuing.
        
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
    
    def _extract_post_data(self, submission):
        """
        Extract relevant data from a Reddit submission for sentiment analysis.
        
        This method carefully extracts the most important information from each
        post while handling potential missing or malformed data gracefully.
        
        Args:
            submission (praw.models.Submission): Reddit post object
            
        Returns:
            dict: Structured post data ready for analysis
        """
        try:
            # Extract basic post information
            post_data = {
                'id': submission.id,
                'title': submission.title,
                'author': str(submission.author) if submission.author else '[deleted]',
                'subreddit': str(submission.subreddit),
                'url': submission.url,
                'permalink': f"https://reddit.com{submission.permalink}",
                'created_utc': submission.created_utc,
                'timestamp': datetime.utcnow().isoformat(),
                'score': submission.score,
                'num_comments': submission.num_comments,
                'upvote_ratio': submission.upvote_ratio
            }
            
            # Handle post content - some posts only have titles, others have text
            if hasattr(submission, 'selftext') and submission.selftext:
                # Text post - combine title and body for analysis
                post_data['content'] = f"{submission.title}. {submission.selftext[:500]}"
                post_data['post_type'] = 'text'
            else:
                # Link post - use title and any available description
                post_data['content'] = submission.title
                post_data['post_type'] = 'link'
            
            # Ensure content is not empty and not too long for API calls
            if not post_data['content'].strip():
                post_data['content'] = submission.title
            
            # Limit content length to avoid API limits and improve processing speed
            if len(post_data['content']) > 1000:
                post_data['content'] = post_data['content'][:1000] + "..."
            
            return post_data
            
        except Exception as e:
            logger.error(f"Error extracting post data from submission {submission.id}: {str(e)}")
            return None
    
    def _publish_to_queue(self, post_data):
        """
        Publish post data to Redis queue for worker processing.
        
        This method handles the serialization and queue management, including
        error handling for connection issues.
        
        Args:
            post_data (dict): Structured post data to publish
        """
        try:
            # Serialize post data as JSON
            message = json.dumps(post_data)
            
            # Push to Redis list (queue)
            # Using LPUSH to add to the left (newest items first)
            self.redis_client.lpush('reddit_posts', message)
            
            # Optionally limit queue size to prevent memory issues
            # Keep only the most recent 1000 posts in queue
            self.redis_client.ltrim('reddit_posts', 0, 999)
            
            logger.info(f"Published post to queue: {post_data['id']} - {post_data['title'][:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to publish post {post_data['id']} to queue: {str(e)}")
    
    def start_streaming(self):
        """
        Begin streaming posts from the specified subreddit.
        
        This is the main method that runs continuously, monitoring Reddit
        for new posts and sending them to the processing queue. It includes
        comprehensive error handling and retry logic for robust operation.
        """
        logger.info(f"Starting to stream posts from r/{self.subreddit_name}")
        
        while True:
            try:
                # Get the subreddit object
                subreddit = self.reddit.subreddit(self.subreddit_name)
                
                # Stream new posts as they are submitted
                # The stream method provides real-time posts
                for submission in subreddit.stream.submissions(skip_existing=True):
                    try:
                        # Extract relevant data from the post
                        post_data = self._extract_post_data(submission)
                        
                        if post_data:
                            # Send to processing queue
                            self._publish_to_queue(post_data)
                        else:
                            logger.warning(f"Skipped post {submission.id} - failed to extract data")
                            
                    except Exception as e:
                        logger.error(f"Error processing submission {submission.id}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
                logger.info("Attempting to reconnect in 30 seconds...")
                time.sleep(30)
                
                # Attempt to reconnect to both services
                try:
                    self.reddit = self._setup_reddit_connection()
                    self.redis_client = self._setup_redis_connection()
                except Exception as reconnect_error:
                    logger.error(f"Reconnection failed: {str(reconnect_error)}")
                    logger.info("Will retry in 60 seconds...")
                    time.sleep(60)

def main():
    """
    Main entry point for the ingestor service.
    
    This function initializes the ingestor and starts the streaming process.
    It's designed to run continuously as a service.
    """
    try:
        logger.info("Starting Reddit Ingestor Service...")
        ingestor = RedditIngestor()
        ingestor.start_streaming()
        
    except KeyboardInterrupt:
        logger.info("Ingestor service stopped by user")
    except Exception as e:
        logger.error(f"Fatal error in ingestor service: {str(e)}")
        raise

if __name__ == "__main__":
    main()