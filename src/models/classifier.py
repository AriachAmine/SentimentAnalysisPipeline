import os
import pickle
import yaml
import time
from src.data.database import Database

class SentimentClassifier:
    def __init__(self, config_path='config/config.yaml', model_path=None):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize database connection
        self.database = Database(config_path)
        
        # Load model
        if not model_path:
            model_path = 'models/saved_models/sentiment_model.pkl'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        print(f"Loaded sentiment analysis model from {model_path}")
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text."""
        # Ensure text is a string
        if not isinstance(text, str) or not text.strip():
            return 'neutral'
            
        # Use the pipeline to predict sentiment
        # The pipeline includes preprocessing, vectorization, and classification
        prediction = self.model.predict([text])[0]
        
        # Map numeric predictions to sentiment labels if needed
        if isinstance(prediction, (int, float)):
            if prediction == 1:
                return 'positive'
            elif prediction == 0:
                return 'negative'
            else:
                return 'neutral'
        
        return prediction
    
    def classify_processed_tweets(self, batch_size=100):
        """Classify processed tweets that don't have sentiment yet."""
        # Get unclassified tweets
        tweets = self.database.get_processed_unclassified_tweets(limit=batch_size)
        
        if not tweets:
            print("No unclassified tweets found.")
            return 0
            
        classified_count = 0
        
        for tweet in tweets:
            # Skip if no processed text
            if not tweet.processed_text:
                continue
                
            # Predict sentiment
            sentiment = self.predict_sentiment(tweet.processed_text)
            
            # Update tweet with sentiment
            update_data = {'sentiment': sentiment}
            success = self.database.update_tweet(tweet.tweet_id, update_data)
            
            if success:
                classified_count += 1
                
        print(f"Classified {classified_count} tweets.")
        return classified_count
    
    def run_continuous_classification(self, interval_seconds=30):
        """Run classification continuously with a specified interval."""
        try:
            print(f"Starting continuous classification (every {interval_seconds} seconds).")
            while True:
                classified = self.classify_processed_tweets()
                if classified == 0:
                    print("Waiting for new tweets to process...")
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("Classification process stopped by user.")
