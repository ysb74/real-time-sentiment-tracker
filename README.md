# Real-time Social Media Sentiment Tracker 🚀

A comprehensive real-time sentiment analysis system that monitors social media posts, analyzes their emotional tone using AI, and visualizes the results in an interactive dashboard.

## 🏗️ Architecture Overview

```

This project demonstrates a microservices architecture with the following components:


┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Reddit    │───▶│  Ingestor   │───▶│    Redis    │───▶│   Worker    │
│     API     │    │  Service    │    │    Queue    │    │  Service    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                 │
                                                                 ▼
┌─────────────┐    ┌─────────────┐                    ┌─────────────┐
│  Streamlit  │◀───│  Dashboard  │◀───────────────────│   Gemini    │
│  Dashboard  │    │   Service   │                    │     API     │
└─────────────┘    └─────────────┘                    └─────────────┘

```
### Components:

- **Ingestor**: Streams live Reddit posts and pushes them to a message queue
- **Worker**: Processes queued posts through AI sentiment analysis
- **Dashboard**: Real-time visualization of sentiment data
- **Redis**: Message queue and data storage
- **Gemini API**: AI-powered sentiment analysis

## 🚀 Features

- **Real-time Data Streaming**: Live Reddit post monitoring
- **AI Sentiment Analysis**: Powered by Google's Gemini API
- **Interactive Dashboard**: Real-time charts and analytics
- **Scalable Architecture**: Microservices with message queues
- **Docker Deployment**: Easy containerized deployment
- **Configurable**: Environment-based configuration

## 📋 Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Redis server
- Reddit API credentials
- Google Gemini API key

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/sentiment-tracker.git
cd sentiment-tracker
```
