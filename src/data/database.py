import os
import sqlite3
import yaml
import pandas as pd
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Tweet(Base):
    """SQLAlchemy model for tweets table."""

    __tablename__ = "tweets"

    tweet_id = Column(String, primary_key=True)
    raw_text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)
    user_id = Column(String, nullable=False)
    username = Column(String, nullable=False)
    processed = Column(Integer, default=0)  # 0=not processed, 1=processed
    processed_text = Column(Text, nullable=True)
    sentiment = Column(String, nullable=True)  # 'positive', 'negative', 'neutral'


class Database:
    def __init__(self, config_path="config/config.yaml"):
        # Load configuration
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Create database directory if it's SQLite
        if self.config["database"]["type"] == "sqlite":
            os.makedirs(
                os.path.dirname(self.config["database"]["sqlite"]["path"]),
                exist_ok=True,
            )
            db_url = f"sqlite:///{self.config['database']['sqlite']['path']}"
        else:  # MySQL
            mysql_config = self.config["database"]["mysql"]
            db_url = f"mysql+pymysql://{mysql_config['username']}:{mysql_config['password']}@{mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}"

        # Create engine and tables (adding checkfirst to prevent errors)
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine, checkfirst=True)

        # Create session
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def insert_tweet(self, tweet_data):
        """Insert a new tweet into the database."""
        try:
            tweet = Tweet(**tweet_data)
            self.session.add(tweet)
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            print(f"Error inserting tweet: {e}")
            return False

    def insert_many_tweets(self, tweet_data_list):
        """Insert multiple tweets into the database."""
        try:
            tweets = [Tweet(**tweet_data) for tweet_data in tweet_data_list]
            self.session.add_all(tweets)
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            print(f"Error inserting tweets: {e}")
            return False

    def get_unprocessed_tweets(self, limit=100):
        """Get tweets that haven't been processed yet."""
        return self.session.query(Tweet).filter_by(processed=0).limit(limit).all()

    def update_tweet(self, tweet_id, update_data):
        """Update a tweet with new data."""
        try:
            self.session.query(Tweet).filter_by(tweet_id=tweet_id).update(update_data)
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            print(f"Error updating tweet: {e}")
            return False

    def get_processed_unclassified_tweets(self, limit=100):
        """Get processed tweets that don't have sentiment yet."""
        return (
            self.session.query(Tweet)
            .filter_by(processed=1)
            .filter(Tweet.sentiment.is_(None))
            .limit(limit)
            .all()
        )

    def get_tweets_by_date_range(self, start_date, end_date):
        """Get tweets within a specific date range."""
        return (
            self.session.query(Tweet)
            .filter(Tweet.created_at >= start_date, Tweet.created_at <= end_date)
            .all()
        )

    def get_sentiment_counts(self):
        """Get counts of tweets by sentiment."""
        query = """
        SELECT sentiment, COUNT(*) as count 
        FROM tweets 
        WHERE sentiment IS NOT NULL 
        GROUP BY sentiment
        """
        return pd.read_sql(query, self.engine)

    def get_sentiment_by_day(self):
        """Get daily sentiment counts."""
        query = """
        SELECT 
            DATE(created_at) as date,
            sentiment,
            COUNT(*) as count
        FROM tweets
        WHERE sentiment IS NOT NULL
        GROUP BY DATE(created_at), sentiment
        ORDER BY date
        """
        return pd.read_sql(query, self.engine)

    def close(self):
        """Close the database session."""
        self.session.close()
