
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

## Project Structure
```
real-time-sentiment-tracker/
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── README.md
├── requirements.txt
├── ingestor/
│   ├── __init__.py
│   └── ingestor.py
├── multimodal/
│   ├── __init__.py
│   ├── image.py          # Image feature extraction & sentiment
│   ├── audio.py          # Audio-to-text & audio sentiment/emotion
│   └── fusion.py         # Fusion logic for multimodal aggregation
├── worker/
│   ├── __init__.py
│   ├── worker.py
│   └── llm_service.py
├── dashboard/
│   ├── __init__.py
│   └── dashboard.py
└── utils/
    ├── __init__.py
    └── helpers.py        # Common utility functions
```
    
# Real-time Social Media Sentiment Tracker

A comprehensive real-time sentiment analysis system that monitors social media platforms, processes content through Large Language Models (LLMs), and provides live visualization through an interactive dashboard.

## 🏗️ Architecture

The system follows a microservices architecture with the following components:

- **Data Ingestor**: Streams real-time data from Twitter/X and Reddit APIs
- **Message Queue**: RabbitMQ handles asynchronous message processing between services
- **Sentiment Workers**: Processes messages using LLMs (OpenAI GPT, Hugging Face transformers)
- **Redis Cache**: Stores processed results and provides fast data access
- **Dashboard**: Real-time Streamlit web application with Plotly visualizations

## 🚀 Features

### Data Ingestion
- Real-time streaming from Twitter/X API using Tweepy
- Reddit data collection via PRAW library
- Configurable keyword tracking and filtering
- Rate limiting and error handling
- Robust connection management with auto-reconnection

### Sentiment Analysis
- Multi-provider LLM support (OpenAI GPT, Hugging Face)
- Automatic fallback between providers
- Batch processing capabilities
- Confidence scoring and reasoning
- Performance monitoring and optimization

### Real-time Dashboard
- Live sentiment metrics and KPIs
- Interactive charts and visualizations
- Platform and time-based filtering
- Recent posts table with search functionality
- System health monitoring
- Data export capabilities

### Infrastructure
- Docker containerization for all services
- Docker Compose orchestration
- Horizontal scaling support
- Health checks and service discovery
- Structured logging and monitoring

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.11+
- API credentials for:
  - Twitter/X Developer Account (Bearer Token)
  - Reddit App (Client ID, Secret)
  - OpenAI API Key (optional)
  - Hugging Face API Key (optional)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/real-time-sentiment-tracker.gitcd real-time-sentiment-tracker
```
2. **Set up environment variables**
```bash
cp .env.example .env
```

Required environment variables:
Twitter/X API
```bash
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
```
Reddit API
```bash
REDDIT_CLIENT_ID=your_reddit_client_idREDDIT_CLIENT_SECRET=your_reddit_client_secretREDDIT_USER_AGENT=SentimentTracker/1.0
```
LLM APIs (at least one required)
```bash
OPENAI_API_KEY=your_openai_api_keyHUGGINGFACE_API_KEY=your_huggingface_api_key
```


3. **Build and start the services**
```bash
docker-compose up –build
```

4. **Access the dashboard**
Open your browser and navigate to `http://localhost:8501`

## 🔧 Configuration

### Tracking Keywords
Edit the keywords in `ingestor/ingestor.py`:
```bash
tracking_keywords: Liststr = “AI”, “technology”, “python”, “programming”
```

### Sentiment Models
Configure LLM providers in `worker/llm_service.py`:
- OpenAI models: `gpt-3.5-turbo`, `gpt-4`
- Hugging Face models: Pre-trained sentiment analysis models

### Dashboard Settings
Customize dashboard behavior in `dashboard/dashboard.py`:
- Refresh intervals
- Data retention periods
- Display limits

## 📊 Usage

### Starting the System
Start all services
```bash 
docker-compose up -d
```
View logs
```bash
docker-compose logs -f
```
Scale workers
```bash
docker-compose up -d –scale worker=3
```

### Monitoring
- **Dashboard**: `http://localhost:8501`
- **RabbitMQ Management**: `http://localhost:15672` (admin/password)
- **Worker Logs**: `docker-compose logs worker`

### API Endpoints
The system exposes several internal endpoints for monitoring:
- Redis: `localhost:6379`
- RabbitMQ: `localhost:5672`

## 🧪 Development

### Local Development Setup
Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```
Install dependencies
```bash
pip install -r requirements.txt
```
Run individual components
```bash
python ingestor/ingestor.py
python worker/worker.py
streamlit run dashboard/dashboard.py
```

## testing

Run Tests
```bash
pytest tests/
```
Run with coverage
```bash
pytest –cov=. tests/
```
### Code Quality
Format code
```bash
black .
isort .
```
Lint
```bash
flake8 .
```
