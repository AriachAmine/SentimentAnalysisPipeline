import praw
import yaml
import datetime
from src.data.database import Database

class RedditCollector:
    def __init__(self, config_path='config/config.yaml'):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize database
        self.database = Database(config_path)
        
        # Initialize Reddit API
        self.reddit = praw.Reddit(
            client_id=self.config['reddit_api']['client_id'],
            client_secret=self.config['reddit_api']['client_secret'],
            user_agent=self.config['reddit_api']['user_agent']
        )
        
        # Set up subreddits to track
        self.subreddits = self.config['collection']['subreddits']
        
    def start_collection(self):
        """Start collecting posts from Reddit."""
        print("Starting Reddit collection...")
        
        subreddit = self.reddit.subreddit('+'.join(self.subreddits))
        
        for post in subreddit.stream.submissions():
            post_data = {
                'tweet_id': f"reddit_{post.id}",
                'raw_text': post.title + " " + (post.selftext if hasattr(post, 'selftext') else ""),
                'username': post.author.name if post.author else "[deleted]",
                'created_at': datetime.datetime.fromtimestamp(post.created_utc),
                'processed': 0
            }
            
            self.database.insert_tweet(post_data)
            print(f"Saved post from r/{post.subreddit.display_name}")
