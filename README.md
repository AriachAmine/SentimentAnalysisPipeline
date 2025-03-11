# Sentiment Analysis Pipeline with Multiple Data Sources

A comprehensive data pipeline that collects content from multiple sources (Reddit, News APIs), processes them using natural language processing (NLP) techniques, and performs sentiment analysis to classify each post as positive, negative, or neutral.

## Features

- Multiple data sources:
  - Reddit posts from configurable subreddits
  - News articles from NewsAPI.org
- Text preprocessing pipeline with NLTK
- Sentiment analysis using machine learning (Logistic Regression with TF-IDF)
- Data storage in SQLite database
- Interactive visualization dashboard
- Continuous processing and classification

## Project Structure

```
/SentimentAnalysisPipeline/
├── config/              # Configuration files
├── data/                # Data storage
│   └── raw/             # Raw downloaded datasets for model training
├── models/              # ML model storage
│   └── saved_models/    # Saved trained models
├── src/                 # Source code
│   ├── data/            # Data collection and processing
│   │   ├── reddit_collector.py    # Reddit data collector
│   │   ├── news_collector.py      # News API collector
│   │   ├── database.py            # Database operations
│   │   └── preprocessor.py        # Text preprocessing
│   ├── models/          # Model training and classification
│   │   ├── trainer.py             # Model training
│   │   └── classifier.py          # Sentiment classification
│   └── visualization/   # Visualization and dashboard
│       ├── dashboard.py           # Interactive web dashboard
│       └── plots.py               # Data visualization
├── app.py               # Main application
└── requirements.txt     # Dependencies
```

## Prerequisites

- Python 3.7+
- Reddit API credentials (Client ID, Client Secret)
- NewsAPI.org API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AriachAmine/SentimentAnalysisPipeline.git
   cd SentimentAnalysisPipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required NLTK data:
   ```bash
   python app.py --setup
   ```

## Configuration

1. Open `config/config.yaml` and update it with your API credentials:
   ```yaml
   # Reddit API Configuration
   reddit_api:
     client_id: "YOUR_REDDIT_CLIENT_ID"
     client_secret: "YOUR_REDDIT_CLIENT_SECRET"
     user_agent: "Sentiment Analysis App by /u/YOUR_USERNAME"

   # News API Configuration
   news_api:
     api_key: "YOUR_NEWS_API_KEY"
   ```

2. Configure data collection parameters:
   ```yaml
   collection:
     keywords: ["technology", "AI", "machinelearning"]  # For news queries
     subreddits: ["technology", "artificial", "MachineLearning"]  # Reddit sources
     languages: ["en"]
     interval_seconds: 60
   ```

## Usage

### Running the Complete Pipeline

The simplest way to run the full application is:

```bash
python app.py
```

This will:
1. Start Reddit and News API collectors
2. Process collected content
3. Classify sentiment (if a model has been trained)
4. Launch the visualization dashboard

### Model Training

Before classification can work, you need to train a sentiment model:

1. Download a sentiment dataset like Sentiment140 from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
2. Place it in the `data/raw` directory
3. Run the training command:

```bash
python app.py --train
```

### Dashboard Only

To run just the visualization dashboard:

```bash
python app.py --dashboard-only
```

## Command Line Options

The application supports several command line options:

- `--train`: Train the sentiment model
- `--dashboard-only`: Run only the dashboard
- `--setup`: Download required NLTK data

## Dashboard Features

The dashboard is accessible at http://localhost:5000 and includes:
- Sentiment distribution pie chart
- Sentiment trend line chart over time
- Recent content with sentiment classification

## License

This project is licensed under the MIT License.

## Acknowledgements

- [PRAW](https://praw.readthedocs.io/) for Reddit API interaction
- [NewsAPI](https://newsapi.org/) for news content
- [NLTK](https://www.nltk.org/) for natural language processing
- [scikit-learn](https://scikit-learn.org/) for machine learning
- [Flask](https://flask.palletsprojects.com/) for the web dashboard
