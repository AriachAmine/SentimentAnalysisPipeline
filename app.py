import os
import time
import threading
import argparse
import nltk
from src.data.reddit_collector import RedditCollector
from src.data.news_collector import NewsCollector
from src.data.preprocessor import TextPreprocessor
from src.models.classifier import SentimentClassifier
from src.visualization.dashboard import SentimentDashboard


def setup_dependencies():
    """Download required NLTK data."""
    print("Setting up dependencies...")
    # Download all necessary NLTK data packages
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("Setup complete.")


def run_reddit_collector(stop_event):
    """Run Reddit collection in a separate thread."""
    try:
        collector = RedditCollector()
        collector.start_collection()
    
        # Keep running until stop event is set
        while not stop_event.is_set():
            time.sleep(1)
    
        print("Reddit collection stopped")
    except Exception as e:
        print(f"Error in Reddit collector: {e}")


def run_news_collector(stop_event):
    """Run news collection in a separate thread."""
    try:
        collector = NewsCollector()
        
        while not stop_event.is_set():
            collector.collect_news()
            # Sleep for the configured interval before fetching news again
            time.sleep(60 * 60)  # Collect news every hour
    
    except Exception as e:
        print(f"Error in News collector: {e}")


def run_preprocessor(stop_event):
    """Run post preprocessing in a separate thread."""
    preprocessor = TextPreprocessor()

    while not stop_event.is_set():
        # Process a batch of posts
        processed = preprocessor.process_unprocessed_tweets(batch_size=50)

        # If no posts were processed, wait a bit
        if processed == 0:
            time.sleep(10)
        else:
            time.sleep(2)


def run_classifier(stop_event):
    """Run sentiment classification in a separate thread."""
    try:
        classifier = SentimentClassifier()

        while not stop_event.is_set():
            # Classify a batch of tweets
            classified = classifier.classify_processed_tweets(batch_size=50)

            # If no tweets were classified, wait a bit
            if classified == 0:
                time.sleep(10)
            else:
                time.sleep(2)
    except FileNotFoundError:
        print("Error: Sentiment model not found. Please train a model first.")
        return


def run_dashboard():
    """Run the visualization dashboard."""
    dashboard = SentimentDashboard()
    dashboard.run(debug=False)


def run_pipeline():
    """Run the complete sentiment analysis pipeline."""
    stop_event = threading.Event()

    try:
        # Start Reddit collector thread
        reddit_thread = threading.Thread(target=run_reddit_collector, args=(stop_event,))
        reddit_thread.daemon = True
        reddit_thread.start()
        print("Reddit collector started")

        # Start News collector thread
        news_thread = threading.Thread(target=run_news_collector, args=(stop_event,))
        news_thread.daemon = True
        news_thread.start()
        print("News collector started")

        # Start preprocessor thread
        preprocessor_thread = threading.Thread(
            target=run_preprocessor, args=(stop_event,)
        )
        preprocessor_thread.daemon = True
        preprocessor_thread.start()
        print("Content preprocessor started")

        # Check if model exists before starting classifier
        if os.path.exists("models/saved_models/sentiment_model.pkl"):
            classifier_thread = threading.Thread(
                target=run_classifier, args=(stop_event,)
            )
            classifier_thread.daemon = True
            classifier_thread.start()
            print("Sentiment classifier started")
        else:
            print("Warning: No sentiment model found. Classification will not run.")
            print("Train a model using the trainer module first.")

        # Start dashboard (this will block until dashboard is closed)
        print("Starting dashboard...")
        run_dashboard()

    except KeyboardInterrupt:
        print("\nStopping all processes...")
    finally:
        # Set the event to signal threads to exit
        stop_event.set()
        time.sleep(2)  # Give threads time to clean up
        print("Pipeline stopped")


def train_model():
    """Train a sentiment analysis model using sample data."""
    # First ensure all dependencies are set up
    setup_dependencies()
    
    from src.models.trainer import SentimentModelTrainer

    print("Model training requires a labeled dataset.")
    print("Please download a dataset like Sentiment140 from Kaggle:")
    print("https://www.kaggle.com/datasets/kazanova/sentiment140")
    print("and place it in the 'data/raw' directory.")

    # Check if both compressed or uncompressed versions might exist
    file_path = r"data\raw\training.1600000.processed.noemoticon.csv"
    zip_path = r"data\raw\training.1600000.processed.noemoticon.csv.zip"
    
    if os.path.exists(zip_path):
        file_path = zip_path
    elif not os.path.exists(file_path):
        print(f"Error: File not found at {file_path} or {zip_path}")
        return

    print(f"Using dataset: {file_path}")
    # For Sentiment140:
    # Column 0: sentiment (0=negative, 4=positive)
    # Column 5: tweet text
    text_column = 5
    label_column = 0

    trainer = SentimentModelTrainer()
    trainer.train_from_file(
        file_path=file_path, text_column=text_column, label_column=label_column
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reddit & News Sentiment Analysis Pipeline")
    parser.add_argument(
        "--train", action="store_true", help="Train the sentiment model"
    )
    parser.add_argument(
        "--dashboard-only", action="store_true", help="Run only the dashboard"
    )
    parser.add_argument(
        "--setup", action="store_true", help="Just download dependencies"
    )

    args = parser.parse_args()

    if args.setup:
        setup_dependencies()
    elif args.train:
        train_model()
    elif args.dashboard_only:
        run_dashboard()
    else:
        # Ensure dependencies are set up before running the pipeline
        setup_dependencies()
        run_pipeline()
