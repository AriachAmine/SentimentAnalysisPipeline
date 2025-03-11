import os
import yaml
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
import pandas as pd
from src.data.database import Database
from src.visualization.plots import SentimentVisualizer


class SentimentDashboard:
    def __init__(self, config_path="config/config.yaml"):
        # Load configuration
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Initialize database and visualizer
        self.database = Database(config_path)
        self.visualizer = SentimentVisualizer(config_path)

        # Create Flask app
        self.app = Flask(__name__)

        self.setup_routes()

        # Visualization update settings
        self.update_interval = self.config["visualization"]["update_interval"]
        self.last_update = None

    def setup_routes(self):
        """Set up Flask routes."""

        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.app.route("/api/sentiment_counts")
        def sentiment_counts():
            df = self.database.get_sentiment_counts()
            if df.empty:
                return jsonify({"labels": [], "counts": []})

            return jsonify(
                {"labels": df["sentiment"].tolist(), "counts": df["count"].tolist()}
            )

        @self.app.route("/api/sentiment_trend")
        def sentiment_trend():
            days = int(request.args.get("days", 7))
            df = self.database.get_sentiment_by_day()

            if df.empty:
                return jsonify(
                    {"dates": [], "positive": [], "negative": [], "neutral": []}
                )

            # Convert date string to datetime
            df["date"] = pd.to_datetime(df["date"])

            # Filter for requested days
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

            # Pivot data
            pivot_df = df.pivot(
                index="date", columns="sentiment", values="count"
            ).fillna(0)

            # Reset index to convert date to column
            pivot_df = pivot_df.reset_index()

            # Format dates for JSON
            dates = pivot_df["date"].dt.strftime("%Y-%m-%d").tolist()

            # Get data for each sentiment
            result = {"dates": dates}
            for sentiment in ["positive", "negative", "neutral"]:
                if sentiment in pivot_df.columns:
                    result[sentiment] = pivot_df[sentiment].tolist()
                else:
                    result[sentiment] = [0] * len(dates)

            return jsonify(result)

        @self.app.route("/api/recent_tweets")
        def recent_tweets():
            # Get the 10 most recent tweets with sentiment
            query = """
            SELECT tweet_id, raw_text, username, created_at, sentiment
            FROM tweets
            WHERE sentiment IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 10
            """
            df = pd.read_sql(query, self.database.engine)

            if df.empty:
                return jsonify([])

            # Format for JSON response
            tweets = []
            for _, row in df.iterrows():
                tweets.append(
                    {
                        "id": row["tweet_id"],
                        "text": row["raw_text"],
                        "username": row["username"],
                        "created_at": row["created_at"],
                        "sentiment": row["sentiment"],
                    }
                )

            return jsonify(tweets)

    def run(self, debug=True):
        """Run the Flask dashboard."""
        host = self.config["visualization"]["dashboard"]["host"]
        port = self.config["visualization"]["dashboard"]["port"]

        print(f"Starting dashboard on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
