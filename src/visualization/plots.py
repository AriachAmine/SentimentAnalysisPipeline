import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from src.data.database import Database

class SentimentVisualizer:
    def __init__(self, config_path='config/config.yaml'):
        self.database = Database(config_path)
        
        # Create directory for saving plots
        os.makedirs('data/visualizations', exist_ok=True)
        
        # Set the style
        sns.set(style="whitegrid")
    
    def create_sentiment_distribution_pie(self, save_path=None):
        """Create a pie chart showing sentiment distribution."""
        print("Generating sentiment distribution pie chart...")
        
        # Get sentiment counts from database
        sentiment_counts = self.database.get_sentiment_counts()
        
        if sentiment_counts.empty:
            print("No sentiment data available.")
            return None
        
        # Create pie chart
        plt.figure(figsize=(8, 8))
        colors = ['#5cb85c', '#d9534f', '#f0ad4e']  # green, red, yellow for positive, negative, neutral
        
        plt.pie(
            sentiment_counts['count'], 
            labels=sentiment_counts['sentiment'], 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=[0.05] * len(sentiment_counts),
            shadow=True
        )
        
        plt.title('Sentiment Distribution', fontsize=16, pad=20)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            print(f"Pie chart saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        return save_path or True
    
    def create_sentiment_timeline(self, days=7, save_path=None):
        """Create a line chart showing sentiment trends over time."""
        print(f"Generating sentiment timeline for the last {days} days...")
        
        # Get sentiment by day
        sentiment_by_day = self.database.get_sentiment_by_day()
        
        if sentiment_by_day.empty:
            print("No sentiment timeline data available.")
            return None
        
        # Convert date string to datetime
        sentiment_by_day['date'] = pd.to_datetime(sentiment_by_day['date'])
        
        # Filter for the specified time range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        sentiment_by_day = sentiment_by_day[
            (sentiment_by_day['date'] >= start_date) & 
            (sentiment_by_day['date'] <= end_date)
        ]
        
        # Pivot the data for plotting
        pivot_data = sentiment_by_day.pivot(index='date', columns='sentiment', values='count')
        
        # Fill missing values with 0
        pivot_data = pivot_data.fillna(0)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot each sentiment
        for sentiment, color in zip(['positive', 'negative', 'neutral'], ['green', 'red', 'orange']):
            if sentiment in pivot_data.columns:
                plt.plot(
                    pivot_data.index, 
                    pivot_data[sentiment], 
                    marker='o',
                    linestyle='-',
                    linewidth=2,
                    markersize=6,
                    color=color,
                    label=sentiment
                )
        
        plt.title(f'Sentiment Trends Over Time (Last {days} Days)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Number of Tweets', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            print(f"Timeline chart saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        return save_path or True
    
    def create_daily_sentiment_bar(self, days=7, save_path=None):
        """Create a stacked bar chart showing daily sentiment counts."""
        print(f"Generating daily sentiment bar chart for the last {days} days...")
        
        # Get sentiment by day
        sentiment_by_day = self.database.get_sentiment_by_day()
        
        if sentiment_by_day.empty:
            print("No sentiment data available for bar chart.")
            return None
        
        # Convert date string to datetime
        sentiment_by_day['date'] = pd.to_datetime(sentiment_by_day['date'])
        
        # Filter for the specified time range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        sentiment_by_day = sentiment_by_day[
            (sentiment_by_day['date'] >= start_date) & 
            (sentiment_by_day['date'] <= end_date)
        ]
        
        # Pivot the data for plotting
        pivot_data = sentiment_by_day.pivot(index='date', columns='sentiment', values='count')
        
        # Fill missing values with 0
        pivot_data = pivot_data.fillna(0)
        
        # Convert date to string for better x-axis labels
        pivot_data = pivot_data.reset_index()
        pivot_data['date'] = pivot_data['date'].dt.strftime('%Y-%m-%d')
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Create bar colors
        colors = {'positive': 'green', 'negative': 'red', 'neutral': 'orange'}
        
        # Create stacked bar
        bottom = None
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in pivot_data.columns:
                plt.bar(
                    pivot_data['date'], 
                    pivot_data[sentiment],
                    bottom=bottom,
                    color=colors.get(sentiment),
                    label=sentiment
                )
                
                # Update bottom for stacking
                if bottom is None:
                    bottom = pivot_data[sentiment].values
                else:
                    bottom = bottom + pivot_data[sentiment].values
        
        plt.title(f'Daily Sentiment Distribution (Last {days} Days)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Number of Tweets', fontsize=12)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            print(f"Bar chart saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        return save_path or True
    
    def generate_all_visualizations(self):
        """Generate all visualizations and save them."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate and save each visualization
        self.create_sentiment_distribution_pie(
            save_path=f'data/visualizations/sentiment_distribution_{timestamp}.png'
        )
        
        self.create_sentiment_timeline(
            days=7,
            save_path=f'data/visualizations/sentiment_timeline_7days_{timestamp}.png'
        )
        
        self.create_daily_sentiment_bar(
            days=7,
            save_path=f'data/visualizations/daily_sentiment_bar_7days_{timestamp}.png'
        )
        
        print("All visualizations generated and saved.")
