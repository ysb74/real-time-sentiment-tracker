"""
Real-time Sentiment Analysis Dashboard

Interactive dashboard built with Streamlit and Plotly for visualizing
real-time social media sentiment analysis results.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import redis
from pydantic_settings import BaseSettings
import structlog

# Configure page settings
st.set_page_config(
    page_title="Real-time Social Media Sentiment Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logger = structlog.get_logger()


class Settings(BaseSettings):
    """Dashboard settings"""
    redis_url: str = "redis://localhost:6379/0"
    rabbitmq_url: str = "amqp://admin:password@localhost:5672/"
    
    # Dashboard configuration
    refresh_interval: int = 5  # seconds
    max_recent_posts: int = 50
    default_time_range: int = 24  # hours
    
    class Config:
        env_file = ".env"


class DashboardData:
    """Data access layer for dashboard"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def get_recent_posts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent processed posts"""
        try:
            recent_ids = self.redis.lrange("sentiment:recent", 0, limit - 1)
            posts = []
            
            for post_id in recent_ids:
                post_key = f"sentiment:post:{post_id.decode()}"
                post_data = self.redis.get(post_key)
                
                if post_data:
                    post = json.loads(post_data)
                    posts.append(post)
            
            return posts
            
        except Exception as e:
            logger.error("Error fetching recent posts", error=str(e))
            return []
    
    def get_sentiment_stats(self, hours: int = 24) -> Dict[str, int]:
        """Get sentiment statistics for the specified time period"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            stats = {"positive": 0, "negative": 0, "neutral": 0}
            
            for sentiment in stats.keys():
                sentiment_key = f"sentiment:label:{sentiment}"
                count = self.redis.zcount(sentiment_key, cutoff_time.timestamp(), "+inf")
                stats[sentiment] = count
            
            return stats
            
        except Exception as e:
            logger.error("Error fetching sentiment stats", error=str(e))
            return {"positive": 0, "negative": 0, "neutral": 0}
    
    def get_platform_stats(self, hours: int = 24) -> Dict[str, int]:
        """Get platform statistics"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            platforms = ["twitter", "reddit"]
            stats = {}
            
            for platform in platforms:
                platform_key = f"sentiment:platform:{platform}"
                count = self.redis.zcount(platform_key, cutoff_time.timestamp(), "+inf")
                stats[platform] = count
            
            return stats
            
        except Exception as e:
            logger.error("Error fetching platform stats", error=str(e))
            return {}
    
    def get_hourly_trends(self, hours: int = 24) -> Dict[str, List[Dict]]:
        """Get hourly sentiment trends"""
        try:
            trends = {"positive": [], "negative": [], "neutral": []}
            current_time = datetime.utcnow()
            
            for i in range(hours):
                hour_time = current_time - timedelta(hours=i)
                hour_key = hour_time.strftime("%Y-%m-%d:%H")
                stats_key = f"sentiment:stats:{hour_key}"
                
                hour_stats = self.redis.hgetall(stats_key)
                
                for sentiment in trends.keys():
                    count = int(hour_stats.get(f"sentiment:{sentiment}".encode(), 0))
                    trends[sentiment].append({
                        "hour": hour_time,
                        "count": count
                    })
            
            # Reverse to get chronological order
            for sentiment in trends.keys():
                trends[sentiment].reverse()
            
            return trends
            
        except Exception as e:
            logger.error("Error fetching hourly trends", error=str(e))
            return {"positive": [], "negative": [], "neutral": []}

    def search_posts(self, keywords=None, sentiment=None, platform=None, since=None, until=None, modality=None):
        # keywords: list, sentiment: str/list, platform:str/list, since/until datetime
        posts = self.get_recent_posts(200)
        # Filtering logic
        if keywords:
            posts = [p for p in posts if any(kw.lower() in p['content'].lower() for kw in keywords)]
        if sentiment:
            if isinstance(sentiment, str): sentiment = [sentiment]
            posts = [p for p in posts if p.get("sentiment_label") in sentiment]
        if platform:
            if isinstance(platform, str): platform = [platform]
            posts = [p for p in posts if p.get("platform") in platform]
        # Date filtering
        if since or until:
            def in_range(p):
                ts = datetime.fromisoformat(p.get("processed_at"))
                if since and ts < since: return False
                if until and ts > until: return False
                return True
            posts = [p for p in posts if in_range(p)]
        if modality:
            posts = [p for p in posts if p.get(modality)]
        return posts

    
    
    def get_worker_status(self) -> List[Dict[str, Any]]:
        """Get status of all workers"""
        try:
            worker_keys = list(self.redis.scan_iter(match="worker:status:*"))
            workers = []
            
            for key in worker_keys:
                worker_data = self.redis.hgetall(key)
                if worker_data:
                    worker = {}
                    for k, v in worker_data.items():
                        try:
                            # Try to parse as JSON for lists
                            if k.decode() == 'available_providers':
                                worker[k.decode()] = json.loads(v.decode())
                            else:
                                worker[k.decode()] = v.decode()
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            worker[k.decode()] = str(v)
                    
                    workers.append(worker)
            
            return workers
            
        except Exception as e:
            logger.error("Error fetching worker status", error=str(e))
            return []


def create_sentiment_pie_chart(stats: Dict[str, int]) -> go.Figure:
    """Create a pie chart for sentiment distribution"""
    labels = list(stats.keys())
    values = list(stats.values())
    colors = {'positive': '#2E8B57', 'negative': '#DC143C', 'neutral': '#4682B4'}
    
    fig = go.Figure(data=[go.Pie(
        labels=[label.title() for label in labels],
        values=values,
        hole=0.4,
        marker_colors=[colors.get(label, '#808080') for label in labels],
        textinfo='label+percent+value',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Sentiment Distribution",
        font=dict(size=14),
        showlegend=True,
        height=400
    )
    
    return fig


def create_platform_bar_chart(stats: Dict[str, int]) -> go.Figure:
    """Create a bar chart for platform distribution"""
    platforms = list(stats.keys())
    counts = list(stats.values())
    
    fig = go.Figure(data=[go.Bar(
        x=[platform.title() for platform in platforms],
        y=counts,
        marker_color=['#1DA1F2', '#FF4500']  # Twitter blue, Reddit orange
    )])
    
    fig.update_layout(
        title="Posts by Platform",
        xaxis_title="Platform",
        yaxis_title="Number of Posts",
        height=400
    )
    
    return fig


def create_sentiment_timeline(trends: Dict[str, List[Dict]]) -> go.Figure:
    """Create a timeline chart for sentiment trends"""
    fig = go.Figure()
    
    colors = {'positive': '#2E8B57', 'negative': '#DC143C', 'neutral': '#4682B4'}
    
    for sentiment, data in trends.items():
        if data:  # Only add if there's data
            hours = [item['hour'] for item in data]
            counts = [item['count'] for item in data]
            
            fig.add_trace(go.Scatter(
                x=hours,
                y=counts,
                mode='lines+markers',
                name=sentiment.title(),
                line=dict(color=colors.get(sentiment, '#808080'), width=3),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title="Sentiment Trends Over Time",
        xaxis_title="Time",
        yaxis_title="Number of Posts",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_recent_posts_table(posts: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a table of recent posts"""
    if not posts:
        return pd.DataFrame()
    
    # Process posts for display
    display_posts = []
    for post in posts:
        # Truncate content for display
        content = post.get('content', '')[:100] + "..." if len(post.get('content', '')) > 100 else post.get('content', '')
        
        display_posts.append({
            'Platform': post.get('platform', '').title(),
            'Content': content,
            'Author': post.get('author', 'Unknown'),
            'Sentiment': post.get('sentiment_label', 'Unknown').title(),
            'Confidence': f"{post.get('sentiment_confidence', 0):.2f}",
            'Model': post.get('model_used', 'Unknown'),
            'Time': post.get('processed_at', '')[:19].replace('T', ' ') if post.get('processed_at') else 'Unknown'
        })
    
    return pd.DataFrame(display_posts)


def main():
    """Main dashboard application"""
    # Initialize settings and data connection
    settings = Settings()
    
    try:
        redis_client = redis.from_url(settings.redis_url)
        redis_client.ping()  # Test connection
        data_handler = DashboardData(redis_client)
    except Exception as e:
        st.error(f"❌ Could not connect to Redis: {e}")
        st.stop()
    
    # Dashboard header
    st.title("📊 Real-time Social Media Sentiment Tracker")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("Advanced Search & Filter") # **Usability Add**
    search_query = st.sidebar.text_input("Search keywords")
    search_modality = st.sidebar.selectbox("Modality", ["All", "text", "image", "audio"])
    search_sentiment = st.sidebar.selectbox("Sentiment", ["All", "positive", "negative", "neutral"])
    search_platform = st.sidebar.selectbox("Platform", ["All", "twitter", "reddit"])
    from_dt = st.sidebar.date_input("From date")
    to_dt = st.sidebar.date_input("To date")

if st.sidebar.button("Apply Filters"):
    results = data_handler.search_posts(
        keywords=[search_query] if search_query else None,
        modality=None if search_modality == "All" else search_modality,
        sentiment=None if search_sentiment == "All" else search_sentiment,
        platform=None if search_platform == "All" else search_platform,
        since=datetime.combine(from_dt, datetime.min.time()) if from_dt else None,
        until=datetime.combine(to_dt, datetime.max.time()) if to_dt else None
    )
    # Display these results (as table, charts etc)

    
    # Time range selector
    time_range = st.sidebar.selectbox(
        "Time Range",
        options=[1, 6, 12, 24, 48],
        index=3,  # Default to 24 hours
        format_func=lambda x: f"Last {x} hours"
    )
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 60, settings.refresh_interval)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.experimental_rerun()
    
    # Data fetch
    with st.spinner("Loading data..."):
        sentiment_stats = data_handler.get_sentiment_stats(time_range)
        platform_stats = data_handler.get_platform_stats(time_range)
        recent_posts = data_handler.get_recent_posts(settings.max_recent_posts)
        hourly_trends = data_handler.get_hourly_trends(min(time_range, 24))  # Max 24 hours for trends
        worker_status = data_handler.get_worker_status()
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    total_posts = sum(sentiment_stats.values())
    
    with col1:
        st.metric(
            label="Total Posts",
            value=total_posts,
            delta=f"Last {time_range}h"
        )
    
    with col2:
        positive_pct = (sentiment_stats['positive'] / total_posts * 100) if total_posts > 0 else 0
        st.metric(
            label="Positive Sentiment",
            value=f"{positive_pct:.1f}%",
            delta=f"{sentiment_stats['positive']} posts"
        )
    
    with col3:
        negative_pct = (sentiment_stats['negative'] / total_posts * 100) if total_posts > 0 else 0
        st.metric(
            label="Negative Sentiment", 
            value=f"{negative_pct:.1f}%",
            delta=f"{sentiment_stats['negative']} posts"
        )
    
    with col4:
        active_workers = len([w for w in worker_status if w.get('status') == 'running'])
        st.metric(
            label="Active Workers",
            value=active_workers,
            delta=f"{len(worker_status)} total"
        )
    
    st.markdown("---")
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        if sum(sentiment_stats.values()) > 0:
            fig_pie = create_sentiment_pie_chart(sentiment_stats)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No sentiment data available for the selected time range.")
    
    with col2:
        if sum(platform_stats.values()) > 0:
            fig_bar = create_platform_bar_chart(platform_stats)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No platform data available for the selected time range.")
    
    # Timeline Chart
    st.subheader("📈 Sentiment Trends")
    if any(trends for trends in hourly_trends.values()):
        fig_timeline = create_sentiment_timeline(hourly_trends)
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No trend data available for the selected time range.")
    
    st.markdown("---")

    st.subheader("Real-Time Alerts")
    alerts_list = data_handler.redis.lrange("alerts:recent", 0, 9)  # store recent triggered alerts as Redis list in monitor.py
    if alerts_list:
        for alert_json in alerts_list:
            alert = json.loads(alert_json)
            st.error(f"🔔 {alert['time']}: {alert['message']}")
    else:
        st.success("No critical alerts in last hour.")

    
    # Recent Posts Section
    st.subheader("📝 Recent Posts")
    
    if recent_posts:
        # Display count and filters
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"Showing {len(recent_posts)} most recent posts")
        
        with col2:
            platform_filter = st.selectbox(
                "Filter by Platform",
                options=["All"] + list(set(post.get('platform', '').title() for post in recent_posts)),
                key="platform_filter"
            )
        
        with col3:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment",
                options=["All"] + list(set(post.get('sentiment_label', '').title() for post in recent_posts)),
                key="sentiment_filter"
            )
        
        # Apply filters
        filtered_posts = recent_posts
        if platform_filter != "All":
            filtered_posts = [p for p in filtered_posts if p.get('platform', '').title() == platform_filter]
        if sentiment_filter != "All":
            filtered_posts = [p for p in filtered_posts if p.get('sentiment_label', '').title() == sentiment_filter]
        
        # Create and display table
        if filtered_posts:
            df = create_recent_posts_table(filtered_posts)
            st.dataframe(df, use_container_width=True, height=400)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"sentiment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No posts match the selected filters.")
    else:
        st.info("No recent posts available.")
    
    st.markdown("---")
    
    # System Status
    with st.expander("🔧 System Status", expanded=False):
        if worker_status:
            st.subheader("Worker Status")
            
            worker_df = pd.DataFrame(worker_status)
            if not worker_df.empty:
                # Clean up worker dataframe for display
                display_cols = ['worker_id', 'status', 'processed_count', 'error_count', 'uptime', 'last_seen']
                available_cols = [col for col in display_cols if col in worker_df.columns]
                
                if available_cols:
                    st.dataframe(worker_df[available_cols], use_container_width=True)
            
            # Show available LLM providers
            all_providers = set()
            for worker in worker_status:
                providers = worker.get('available_providers', [])
                if isinstance(providers, list):
                    all_providers.update(providers)
            
            if all_providers:
                st.write("**Available LLM Providers:**")
                st.write(", ".join(sorted(all_providers)))
        else:
            st.warning("No worker status information available.")
        
        # Redis connection status
        try:
            redis_info = redis_client.info()
            st.success(f"✅ Redis Connected - {redis_info.get('redis_version', 'Unknown version')}")
        except Exception as e:
            st.error(f"❌ Redis Connection Error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Auto-refresh: {'On' if auto_refresh else 'Off'}"
    )
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_rate)
        st.experimental_rerun()


if __name__ == "__main__":
    main()
