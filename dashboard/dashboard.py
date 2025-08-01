#!/usr/bin/env python3
"""
Real-time Social Media Sentiment Tracker - Dashboard Service

This service provides a real-time web dashboard using Streamlit to visualize
sentiment analysis results. It connects to Redis to fetch processed data
and displays interactive charts, statistics, and recent posts.

The dashboard provides:
- Real-time sentiment distribution charts
- Historical sentiment trends
- Recent posts table with sentiment scores
- System statistics and performance metrics
- Configuration controls

Author: Your Name
Date: 2024
"""

import os
import json
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import redis
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentDashboard:
    """
    A Streamlit-based dashboard for visualizing real-time sentiment analysis.
    
    This class handles:
    1. Redis connection and data retrieval
    2. Data formatting and preparation
    3. Interactive chart generation
    4. Real-time updates and caching
    """
    
    def __init__(self):
        """Initialize the dashboard with Redis connection and configuration."""
        # Load configuration
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        self.redis_db = int(os.getenv('REDIS_DB', 0))
        self.max_posts_display = int(os.getenv('MAX_POSTS_DISPLAY', 100))
        self.refresh_interval = int(os.getenv('REFRESH_INTERVAL', 5))
        
        # Initialize Redis connection
        self.redis_client = self._setup_redis_connection()
        
        # Configure Streamlit page
        st.set_page_config(
            page_title="Social Media Sentiment Tracker",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        self._inject_custom_css()
    
    def _setup_redis_connection(self):
        """Create and test Redis connection."""
        try:
            redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True
            )
            redis_client.ping()
            logger.info(f"Dashboard connected to Redis at {self.redis_host}:{self.redis_port}")
            return redis_client
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            st.error(f"Cannot connect to Redis: {str(e)}")
            st.stop()
    
    def _inject_custom_css(self):
        """Inject custom CSS for improved dashboard styling."""
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .sentiment-positive {
            color: #28a745;
            font-weight: bold;
        }
        .sentiment-negative {
            color: #dc3545;
            font-weight: bold;
        }
        .sentiment-neutral {
            color: #6c757d;
            font-weight: bold;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online {
            background-color: #28a745;
        }
        .status-offline {
            background-color: #dc3545;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @st.cache_data(ttl=30)  # Cache for 30 seconds
    def _get_sentiment_stats(self) -> Dict:
        """Get current sentiment statistics from Redis."""
        try:
            stats = self.redis_client.hgetall('sentiment_stats')
            if not stats:
                return {
                    'total_count': 0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0,
                    'avg_confidence': 0,
                    'last_updated': None
                }
            
            # Convert string values to appropriate types
            processed_stats = {}
            for key, value in stats.items():
                if key in ['total_count', 'positive_count', 'negative_count', 'neutral_count']:
                    processed_stats[key] = int(value) if value else 0
                elif key == 'avg_confidence':
                    processed_stats[key] = float(value) if value else 0
                else:
                    processed_stats[key] = value
            
            return processed_stats
        except Exception as e:
            logger.error(f"Error getting sentiment stats: {str(e)}")
            return {}
    
    @st.cache_data(ttl=60)  # Cache for 1 minute
    def _get_hourly_trends(self, hours_back: int = 24) -> pd.DataFrame:
        """Get hourly sentiment trends for the specified time period."""
        try:
            now = datetime.utcnow()
            trend_data = []
            
            for i in range(hours_back):
                hour = now - timedelta(hours=i)
                hour_key = hour.strftime('%Y-%m-%d-%H')
                
                hourly_data = self.redis_client.hgetall(f'hourly_sentiment:{hour_key}')
                
                if hourly_data:
                    trend_data.append({
                        'hour': hour,
                        'positive': int(hourly_data.get('positive', 0)),
                        'negative': int(hourly_data.get('negative', 0)),
                        'neutral': int(hourly_data.get('neutral', 0))
                    })
                else:
                    trend_data.append({
                        'hour': hour,
                        'positive': 0,
                        'negative': 0,
                        'neutral': 0
                    })
            
            df = pd.DataFrame(trend_data)
            df = df.sort_values('hour')
            return df
            
        except Exception as e:
            logger.error(f"Error getting hourly trends: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=10)  # Cache for 10 seconds
    def _get_recent_posts(self, limit: int = None) -> List[Dict]:
        """Get recent processed posts from Redis."""
        try:
            if limit is None:
                limit = self.max_posts_display
            
            # Get recent posts from Redis list
            recent_posts_raw = self.redis_client.lrange('recent_posts', 0, limit - 1)
            
            posts = []
            for post_raw in recent_posts_raw:
                try:
                    post_data = json.loads(post_raw)
                    posts.append(post_data)
                except json.JSONDecodeError:
                    continue
            
            return posts
            
        except Exception as e:
            logger.error(f"Error getting recent posts: {str(e)}")
            return []
    
    def _create_sentiment_distribution_chart(self, stats: Dict) -> go.Figure:
        """Create a pie chart showing sentiment distribution."""
        if not stats or stats.get('total_count', 0) == 0:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(title="Sentiment Distribution")
            return fig
        
        sentiments = ['Positive', 'Negative', 'Neutral']
        values = [
            stats.get('positive_count', 0),
            stats.get('negative_count', 0),
            stats.get('neutral_count', 0)
        ]
        colors = ['#28a745', '#dc3545', '#6c757d']
        
        fig = go.Figure(data=[go.Pie(
            labels=sentiments,
            values=values,
            marker_colors=colors,
            hole=0.4
        )])
        
        fig.update_layout(
            title={
                'text': "Real-time Sentiment Distribution",
                'x': 0.5,
                'xanchor': 'center'
            },
            font=dict(size=14),
            showlegend=True,
            height=400
        )
        
        return fig
    
    def _create_trend_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create a line chart showing sentiment trends over time."""
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No trend data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(title="Sentiment Trends")
            return fig
        
        fig = go.Figure()
        
        # Add lines for each sentiment
        fig.add_trace(go.Scatter(
            x=df['hour'],
            y=df['positive'],
            mode='lines+markers',
            name='Positive',
            line=dict(color='#28a745', width=3),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['hour'],
            y=df['negative'],
            mode='lines+markers',
            name='Negative',
            line=dict(color='#dc3545', width=3),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x