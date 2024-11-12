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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the NLTK data path
NLTK_DATA_PATH = os.path.join(settings.BASE_DIR, 'nltk_data')
os.makedirs(NLTK_DATA_PATH, exist_ok=True)  # Ensure directory exists
nltk.data.path.append(NLTK_DATA_PATH)

def ensure_nltk_downloads():
    """Ensure all required NLTK data is downloaded."""
    required_nltk_data = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'corpora/wordnet': 'wordnet',
        'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger'
    }
    
    for resource_path, resource_name in required_nltk_data.items():
        try:
            # First try to find the resource
            nltk.data.find(resource_path)
            logger.info(f"Found NLTK resource: {resource_name}")
        except LookupError:
            try:
                # If not found, download it
                logger.info(f"Downloading NLTK resource: {resource_name}")
                nltk.download(resource_name, download_dir=NLTK_DATA_PATH, quiet=True)
                logger.info(f"Successfully downloaded {resource_name}")
            except Exception as e:
                logger.error(f"Error downloading {resource_name}: {str(e)}")
                raise RuntimeError(f"Failed to download required NLTK resource: {resource_name}")

# Initialize NLTK resources safely
try:
    ensure_nltk_downloads()
except Exception as e:
    logger.error(f"Failed to initialize NLTK resources: {str(e)}")
    raise

class TextPreprocessor:
    def __init__(self):
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.error(f"Error initializing TextPreprocessor: {str(e)}")
            raise
        
    def preprocess(self, text: str) -> str:
        """Clean and lemmatize the input text for better semantic matching."""
        try:
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            raise

def load_response_data() -> Tuple[pd.DataFrame, TfidfVectorizer, np.ndarray]:
    """Load and preprocess the response data, fitting a TF-IDF vectorizer."""
    try:
        global response_data, vectorizer, context_vectors
        
        path = os.path.join(settings.BASE_DIR, 'media', 'Dataset.csv')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset.csv not found at {path}")
        
        data = pd.read_csv(path)
        data = data[['Context', 'Response']].dropna().drop_duplicates()

        preprocessor = TextPreprocessor()
        data['processed_context'] = data['Context'].apply(preprocessor.preprocess)
        
        vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.85
        )
        context_vectors = vectorizer.fit_transform(data['processed_context'])

        response_data = data
        logger.info(f"Loaded response data: {data.shape[0]} rows, {data.shape[1]} columns")
        return data, vectorizer, context_vectors
    except Exception as e:
        logger.error(f"Error loading response data: {str(e)}")
        raise

def get_best_response(user_input: str) -> str:
    """Match the user input to the best response based on cosine similarity."""
    try:
        preprocessor = TextPreprocessor()
        processed_input = preprocessor.preprocess(user_input)
        
        input_vector = vectorizer.transform([processed_input])
        similarities = cosine_similarity(input_vector, context_vectors).flatten()
        best_match_idx = np.argmax(similarities)
        
        if similarities[best_match_idx] < 0.5:
            logger.warning("No good match found for the input.")
            return "I'm here to listen, but I couldn't find a specific response. Could you tell me more?"
        
        best_response = response_data.iloc[best_match_idx]['Response']
        return best_response
    except Exception as e:
        logger.error(f"Error finding best response: {str(e)}")
        return "I'm sorry, I encountered an error while processing your request."