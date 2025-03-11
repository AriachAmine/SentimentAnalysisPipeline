import requests
import yaml
import datetime
from src.data.database import Database

class NewsCollector:
    def __init__(self, config_path='config/config.yaml'):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize database
        self.database = Database(config_path)
        
        # Initialize News API key
        self.api_key = self.config['news_api']['api_key']
        self.keywords = self.config['collection']['keywords']
        
    def collect_news(self):
        """Collect news articles related to keywords."""
        print("Collecting news articles...")
        
        for keyword in self.keywords:
            url = f"https://newsapi.org/v2/everything?q={keyword}&apiKey={self.api_key}&language=en&pageSize=100"
            
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                
                for article in articles:
                    article_data = {
                        'tweet_id': f"news_{article['url'].split('/')[-1]}",
                        'raw_text': article['title'] + " " + (article['description'] or ""),
                        'username': article['source']['name'],
                        'created_at': datetime.datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ"),
                        'processed': 0
                    }
                    
                    self.database.insert_tweet(article_data)
                
                print(f"Saved {len(articles)} articles for keyword '{keyword}'")
            else:
                print(f"Error fetching news for '{keyword}': {response.status_code}")
