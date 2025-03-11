import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from src.data.database import Database

# Download ALL necessary NLTK data!
print("Downloading required NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self, config_path='config/config.yaml'):
        self.database = Database(config_path)
        
        try:
            self.stopwords = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except LookupError:
            # If resources still not available, try explicit download
            print("NLTK resources not found. Downloading now...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            # Try again
            self.stopwords = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text):
        """Clean and normalize text."""
        if not isinstance(text, str) or not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove user mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags symbols but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        try:
            # Tokenize with error handling
            tokens = word_tokenize(text)
        except LookupError:
            # Fallback tokenization if NLTK resources still not available
            tokens = text.split()
        
        # Remove stopwords and lemmatize
        clean_tokens = []
        for token in tokens:
            if token not in self.stopwords and len(token) > 2:
                try:
                    lemma = self.lemmatizer.lemmatize(token)
                    clean_tokens.append(lemma)
                except:
                    clean_tokens.append(token)
        
        # Return cleaned text
        return ' '.join(clean_tokens)
        
    def process_unprocessed_tweets(self, batch_size=100):
        """Process tweets from the database that haven't been processed yet."""
        unprocessed_tweets = self.database.get_unprocessed_tweets(limit=batch_size)
        
        if not unprocessed_tweets:
            print("No unprocessed tweets found.")
            return 0
            
        processed_count = 0
        
        for tweet in unprocessed_tweets:
            processed_text = self.preprocess_text(tweet.raw_text)
            
            # Update tweet in database
            update_data = {
                'processed': 1,
                'processed_text': processed_text
            }
            
            success = self.database.update_tweet(tweet.tweet_id, update_data)
            
            if success:
                processed_count += 1
                
        print(f"Processed {processed_count} tweets.")
        return processed_count
        
    def get_processed_text_df(self):
        """Get processed tweets as a pandas DataFrame."""
        query = """
        SELECT tweet_id, processed_text, sentiment
        FROM tweets
        WHERE processed = 1
        """
        return pd.read_sql(query, self.database.engine)
