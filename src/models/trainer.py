import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import pickle
from src.data.preprocessor import TextPreprocessor


class SentimentModelTrainer:
    def __init__(self, config_path="config/config.yaml"):
        # Load configuration
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Model parameters
        self.random_state = self.config["model"]["random_state"]
        self.test_size = self.config["model"]["train_test_split"]

        # Create directories for models
        os.makedirs("models/saved_models", exist_ok=True)

    def load_dataset(self, file_path):
        """Load a labeled sentiment dataset."""
        print(f"Loading dataset from {file_path}")

        # Check file extension to determine format
        if file_path.endswith(".zip"):
            # Handle zip file directly
            df = pd.read_csv(file_path, compression='zip', encoding='latin1', header=None)
        elif file_path.endswith(".csv"):
            # Try different encodings if needed
            try:
                # First check if this is the Sentiment140 format (no header)
                with open(file_path, 'r', encoding='latin1') as f:
                    first_line = f.readline().strip()
                    # If first line starts with a number, it's likely not a header
                    if first_line.split(',')[0].strip('"').isdigit():
                        df = pd.read_csv(file_path, encoding='latin1', header=None)
                    else:
                        df = pd.read_csv(file_path, encoding='latin1')
            except Exception as e:
                print(f"Error with latin1 encoding: {e}")
                try:
                    df = pd.read_csv(file_path, encoding='ISO-8859-1', header=None)
                except Exception as e:
                    print(f"Error with ISO-8859-1 encoding: {e}")
                    # As a last resort, try with error handling
                    df = pd.read_csv(file_path, encoding='utf-8', errors='replace', header=None)
        elif file_path.endswith(".tsv"):
            df = pd.read_csv(file_path, sep="\t", encoding='latin1')
        else:
            raise ValueError("Unsupported file format. Use CSV or TSV.")

        print(f"Dataset loaded with {len(df)} rows")
        return df

    def preprocess_dataset(self, df, text_column, label_column):
        """Preprocess the dataset for training."""
        # Initialize preprocessor
        preprocessor = TextPreprocessor()

        print("Preprocessing dataset...")
        # Clean text - handle both string column names and integer indices
        try:
            # Try as column name first
            df["processed_text"] = df[text_column].apply(preprocessor.preprocess_text)
        except KeyError:
            # If that fails, try as integer index
            try:
                text_col_idx = int(text_column)
                df["processed_text"] = df.iloc[:, text_col_idx].apply(preprocessor.preprocess_text)
            except (ValueError, IndexError) as e:
                raise ValueError(f"Could not access text column: {text_column}. Error: {e}")

        # Handle label column similarly
        try:
            # Try as column name
            labels = df[label_column]
        except KeyError:
            # Try as integer index
            try:
                label_col_idx = int(label_column)
                labels = df.iloc[:, label_col_idx]
            except (ValueError, IndexError) as e:
                raise ValueError(f"Could not access label column: {label_column}. Error: {e}")

        # Map labels if needed (e.g., 4 -> 1 for positive, 0 -> 0 for negative)
        if labels.nunique() == 2 and set(labels.unique()) != {0, 1}:
            sentiment_mapping = {
                min(labels.unique()): 0,  # Lowest value is negative
                max(labels.unique()): 1,  # Highest value is positive
            }
            df["sentiment"] = labels.map(sentiment_mapping)
        else:
            df["sentiment"] = labels

        # Drop rows with empty processed text
        df = df[df["processed_text"].str.strip().astype(bool)]

        print(f"Preprocessing complete. {len(df)} valid samples.")
        return df

    def train_logistic_regression(self, X_train, y_train):
        """Train a logistic regression model with TF-IDF features."""
        print("Training logistic regression model...")

        # Create pipeline with TF-IDF vectorizer and logistic regression
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000)),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0, max_iter=1000, random_state=self.random_state, n_jobs=-1
                    ),
                ),
            ]
        )

        # Train the model
        pipeline.fit(X_train, y_train)
        print("Model training complete!")

        return pipeline

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the trained model."""
        print("\nEvaluating model...")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

        return accuracy, report

    def save_model(self, model, model_name="sentiment_model"):
        """Save the trained model to disk."""
        model_path = f"models/saved_models/{model_name}.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"Model saved to {model_path}")

    def train_from_file(
        self, file_path, text_column, label_column, model_name="sentiment_model"
    ):
        """Complete training workflow from file loading to model saving."""
        # Load dataset
        df = self.load_dataset(file_path)

        # Preprocess dataset
        processed_df = self.preprocess_dataset(df, text_column, label_column)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_df["processed_text"],
            processed_df["sentiment"],
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=processed_df["sentiment"],
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Testing set: {len(X_test)} samples")

        # Train model based on configuration
        model_type = self.config["model"]["type"]
        if model_type == "logistic_regression":
            model = self.train_logistic_regression(X_train, y_train)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Evaluate model
        self.evaluate_model(model, X_test, y_test)

        # Save model
        self.save_model(model, model_name)

        return model
