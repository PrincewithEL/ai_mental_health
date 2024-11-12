# mental_health_App/ai_module/emotion_analysis.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Dict
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
from django.conf import settings
from textblob import TextBlob
from django.contrib.staticfiles import finders

# Ensure NLTK data is downloaded
def ensure_nltk_downloads():
    """Ensure all required NLTK data is downloaded."""
    required_nltk_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for item in required_nltk_data:
        try:
            nltk.data.find(f'corpora/{item}')
        except LookupError:
            logging.info(f"Downloading NLTK resource: {item}")
            nltk.download(item, download_dir=os.path.join(settings.BASE_DIR, 'nltk_data'))
    logging.info("NLTK data check completed successfully")

# Call this function at module import to ensure required NLTK resources are downloaded
ensure_nltk_downloads()

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess(self, text: str) -> str:
        """Clean and lemmatize the input text for better semantic matching."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

def load_response_data() -> Tuple[pd.DataFrame, TfidfVectorizer, np.ndarray]:
    """Load and preprocess the response data, fitting a TF-IDF vectorizer."""
    global response_data, vectorizer, context_vectors
    
    # Define the path to the dataset
    path = os.path.join(settings.BASE_DIR, 'media', 'Dataset.csv')
    
    # Load data
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset.csv not found at {path}")
    
    data = pd.read_csv(path)
    data = data[['Context', 'Response']].dropna().drop_duplicates()

    # Preprocess data
    preprocessor = TextPreprocessor()
    data['processed_context'] = data['Context'].apply(preprocessor.preprocess)
    
    # Initialize and fit vectorizer with refined parameters
    vectorizer = TfidfVectorizer(
        max_features=2000,       # Increase features for richer context representation
        ngram_range=(1, 2),      # Include bigrams for better contextual capture
        min_df=1,                # Lower threshold for rare but potentially relevant terms
        max_df=0.85              # More general terms included
    )
    context_vectors = vectorizer.fit_transform(data['processed_context'])

    response_data = data
    logging.info(f"Loaded response data: {data.shape[0]} rows, {data.shape[1]} columns")
    return data, vectorizer, context_vectors

def get_best_response(user_input: str) -> str:
    """Match the user input to the best response based on cosine similarity."""
    try:
        preprocessor = TextPreprocessor()
        processed_input = preprocessor.preprocess(user_input)
        
        # Transform the input to match with context vectors
        input_vector = vectorizer.transform([processed_input])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(input_vector, context_vectors).flatten()
        best_match_idx = np.argmax(similarities)
        
        # Check if similarity meets a minimum threshold
        if similarities[best_match_idx] < 0.5:  # Adjust threshold for better relevance
            logging.warning("No good match found for the input.")
            return "I'm here to listen, but I couldn't find a specific response. Could you tell me more?"
        
        best_response = response_data.iloc[best_match_idx]['Response']
        return best_response
    except Exception as e:
        logging.error(f"Error finding best response: {str(e)}")
        return "I'm sorry, I encountered an error while processing your request."