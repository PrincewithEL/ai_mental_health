from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.conf import settings
from django.shortcuts import redirect
from django.http import JsonResponse
from .forms import *
import openai
from rapidfuzz import process, fuzz 
import os
from django.views import View
from django.http import JsonResponse
from django.views import View
from django.http import JsonResponse
import logging
import openai
from textblob import TextBlob
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from django.conf import settings
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv('OPENAI_API_KEY', 'your-fallback-api-key')

# Define the NLTK data path
NLTK_DATA_PATH = os.path.join(settings.BASE_DIR, 'nltk_data')
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

class NLTKDownloader:
    @staticmethod
    def ensure_nltk_downloads():
        """Ensure all required NLTK data is downloaded."""
        required_nltk_data = {
            'stopwords': 'corpora/stopwords',
            'wordnet': 'corpora/wordnet',
            'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
        }
        
        for resource_name, resource_path in required_nltk_data.items():
            try:
                nltk.download(resource_name, download_dir=NLTK_DATA_PATH, quiet=True)
                logger.info(f"Successfully downloaded {resource_name}")
            except Exception as e:
                logger.error(f"Error downloading {resource_name}: {str(e)}")
                raise RuntimeError(f"Failed to download required NLTK resource: {resource_name}")

class TextPreprocessor:
    def __init__(self):
        """Initialize the TextPreprocessor with required NLTK components."""
        try:
            NLTKDownloader.ensure_nltk_downloads()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.error(f"Error initializing TextPreprocessor: {str(e)}")
            raise

    def simple_tokenize(self, text: str) -> list:
        """Tokenize text into words."""
        return re.findall(r'\b\w+\b', text.lower())
        
    def preprocess(self, text: str) -> str:
        """Preprocess text by lowercasing, removing special characters, and lemmatizing."""
        try:
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = self.simple_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) 
                     for token in tokens 
                     if token not in self.stop_words]
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            return re.sub(r'[^a-zA-Z\s]', '', text.lower())

class DataLoader:
    @staticmethod
    def load_response_data() -> Tuple[pd.DataFrame, TfidfVectorizer, np.ndarray]:
        """Load and preprocess the response data, fitting a TF-IDF vectorizer."""
        try:
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
                max_df=0.85,
                token_pattern=r'\b\w+\b'
            )
            context_vectors = vectorizer.fit_transform(data['processed_context'])

            logger.info(f"Loaded response data: {data.shape[0]} rows, {data.shape[1]} columns")
            return data, vectorizer, context_vectors
        except Exception as e:
            logger.error(f"Error loading response data: {str(e)}")
            raise

class EmotionResponseView(View):
    def __init__(self, **kwargs):
        """Initialize the EmotionResponseView with required components."""
        super().__init__(**kwargs)
        self.preprocessor = None
        self.response_data = None
        self.vectorizer = None
        self.context_vectors = None
        try:
            self.initialize_components()
        except Exception as e:
            logger.error(f"Error initializing EmotionResponseView: {str(e)}")
            raise

    def initialize_components(self):
        """Initialize all required components."""
        try:
            self.preprocessor = TextPreprocessor()
            self.response_data, self.vectorizer, self.context_vectors = DataLoader.load_response_data()
            logger.info("Successfully initialized all components")
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise

    def get(self, request, *args, **kwargs):
        """Handle GET requests with fallback chain."""
        try:
            user_message = request.GET.get('message', '')
            if not user_message:
                return self.create_response("Could you please share what's on your mind?")

            # Fallback chain
            response = (
                self.get_tfidf_response(user_message) or
                self.get_openai_response(user_message) or
                self.get_local_response(user_message)
            )
            
            return self.create_response(response)
            
        except Exception as e:
            logger.error(f"Error in GET request handling: {str(e)}")
            return self.create_response(
                "I'm here to support you. Could you please share more about what you're feeling?"
            )
            
    def get_tfidf_response(self, user_message: str) -> Optional[str]:
        """Get response using TF-IDF matching."""
        try:
            if any(component is None for component in 
                  [self.preprocessor, self.vectorizer, self.context_vectors, self.response_data]):
                logger.error("TF-IDF components not properly initialized")
                return None

            processed_input = self.preprocessor.preprocess(user_message)
            input_vector = self.vectorizer.transform([processed_input])
            similarities = cosine_similarity(input_vector, self.context_vectors).flatten()
            best_match_idx = np.argmax(similarities)
            
            # Changed the comparison to use a scalar value
            if float(similarities[best_match_idx]) < 0.5:  # Convert to float for comparison
                logger.warning("No good TF-IDF match found")
                return None
            
            return self.response_data.iloc[best_match_idx]['Response']
        except Exception as e:
            logger.error(f"Error in TF-IDF matching: {str(e)}")
            return None

    def get_openai_response(self, user_message: str) -> Optional[str]:
        """Get response from OpenAI API with error handling."""
        try:
            # Check if OpenAI API key is set in environment variables first
            api_key = os.getenv('OPENAI_API_KEY')
            
            # If not in environment, try getting from Django settings
            if not api_key and hasattr(settings, 'OPENAI_API_KEY'):
                api_key = settings.OPENAI_API_KEY
            
            if not api_key:
                logger.warning("OpenAI API key not found in environment or settings")
                return None

            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user",
                    "content": f"The user said: '{user_message}'. Provide a helpful and empathetic response."
                }],
                max_tokens=150,
                temperature=0.7
            )
            return response['choices'][0]['message']['content'].strip()
        except openai.error.AuthenticationError:
            logger.error("OpenAI API key is invalid")
            return None
        except openai.error.RateLimitError:
            logger.error("OpenAI API rate limit exceeded")
            return None
        except Exception as e:
            logger.error(f"Error fetching OpenAI response: {str(e)}")
            return None

    def get_local_response(self, user_message: str) -> str:
        """Get response using local emotion mapping."""
        try:
            normalized_message = user_message.strip().lower()
            
            response_map = {
                'sad': "I understand you're feeling sad. Would you like to talk about what's troubling you?",
                'happy': "I'm glad you're feeling happy! What's bringing you joy?",
                'angry': "I can sense your frustration. Would you like to talk about what's bothering you?",
                'anxious': "It's normal to feel anxious sometimes. Would you like to share what's making you feel this way?",
                'worried': "I hear that you're worried. Would you like to talk about your concerns?",
                'lonely': "I'm here with you. Would you like to talk about what's making you feel lonely?",
                'stressed': "Dealing with stress can be challenging. Would you like to share what's causing your stress?",
                'overwhelmed': "It's okay to feel overwhelmed. Would you like to break down what's contributing to this feeling?",
                'tired': "I understand you're feeling tired. Would you like to talk about what's draining your energy?",
                'confused': "It's okay to feel confused. Would you like to talk through what's on your mind?",
                    # Immediate crisis responses
                "i feel like harming myself": "I'm so sorry you're feeling this way. Your pain is real, and you don't have to face it alone. Please reach out to someone who can help, like the 988 Suicide and Crisis Lifeline or a trusted person in your life. Your life matters, and there are people who care deeply about you. Would you consider reaching out for support right now?",
                
                "i want to disappear": "It sounds like you're going through an incredibly tough time. Please know that you don't have to face this alone. There are people who care about you and want to help. Would you consider speaking with someone you trust or calling a crisis line? You are valued and important, even if it doesn't feel like it right now.",

                # Depression-specific responses
                "i feel empty inside": "That feeling of emptiness can be so hard to bear. It may help to talk about it with someone you trust or a mental health professional. Sometimes doing small, nurturing activities like taking a walk, journaling, or listening to calming music can help start to fill that emptiness. You're not alone, and there is hope.",
                
                "i feel hopeless": "I'm so sorry you're feeling this way. It can be hard to see a way forward when things feel so bleak. Sometimes taking just one small step, like reaching out to a friend or doing something kind for yourself, can make a difference. Would you like to talk about what's on your mind?",

                # Anxiety-specific responses
                "i feel like i'm losing control": "It can be so scary to feel this way. Let's try to focus on the present moment. Take a deep breath with me: in for 4 counts, hold for 4, and out for 4. Grounding techniques like this can help. You're not alone, and it's okay to seek support to navigate these feelings.",
                
                "i can't stop feeling anxious": "Anxiety can feel overwhelming, but it doesn't define you. Let's try to find ways to manage it together. Would you consider writing down your worries and focusing on just one thing you can address? Breathing exercises or even talking to someone about how you're feeling can help a lot.",

                # Social anxiety responses
                "i can't talk to people": "Social anxiety can make interactions feel so intimidating. Let's start small—could you practice saying hello to one person or even just being present in a group without speaking? Over time, these little steps can build confidence. Remember, it's okay to seek support from a therapist who can guide you through this process.",
                
                "i feel judged by everyone": "It's tough to feel like you're under a microscope. The truth is, most people are focused on themselves, not judging you. Could you try challenging this thought by asking yourself, 'What evidence do I have that this is true?' Over time, small shifts in perspective can help reduce these feelings.",

                # Trauma responses
                "i can't escape my past": "I'm so sorry you're feeling trapped by your past. Healing is possible, even if it feels out of reach right now. Would you consider talking to a trauma-informed therapist? They can help you find ways to process these feelings and reclaim your present.",
                
                "i feel broken": "Feeling broken can be incredibly painful, but you're not alone. You're a whole person, even if you feel shattered right now. Healing takes time, and there are people who care about you and want to help you rebuild piece by piece. Would you like to explore resources for support?",

                # Grief responses
                "i can't stop missing them": "Grieving someone you love is such a deep and personal experience. It's okay to feel this way and to give yourself permission to feel the waves of loss. Sometimes sharing memories with others or writing a letter to your loved one can provide comfort. You're not alone in this.",
                
                "the pain of losing them won't go away": "Grief isn't something you 'get over,' but something you learn to carry. It's okay to feel the pain, and it doesn't mean you're weak. Would you like to talk about your loved one or explore ways to honor their memory? Support groups or grief counseling can also be incredibly helpful.",

                # Relationship challenges
                "i feel trapped in my relationship": "It sounds like you're facing a very challenging situation. Your feelings are valid, and your safety and well-being are important. Could you talk to someone you trust about this? There are also confidential resources like domestic violence hotlines that can provide guidance and support.",
                
                "i can't connect with my partner": "Relationships can go through tough patches, and feeling disconnected can be painful. Have you been able to talk openly with your partner about this? Sometimes exploring these feelings together or with a counselor can help rebuild understanding and intimacy.",

                # Self-esteem and identity
                "i feel like i don't belong": "It's really hard to feel like you don't fit in. You're not alone in this experience, even if it feels isolating. Could you explore groups or communities that share your interests or values? Connection is out there, even if it takes time to find.",
                
                "i don't like who i am": "Self-acceptance is a journey, and it's okay to feel this way as you work through it. Would you consider exploring what you do value about yourself or seeking guidance from a counselor? You're worthy of compassion and understanding, from both yourself and others.",

                # Work/study stress
                "i feel overwhelmed by my workload": "Overwhelm can feel paralyzing, but you're not alone in this. Could we try breaking tasks into smaller, more manageable steps? Prioritizing and setting realistic goals can help. It's also okay to ask for help or adjust deadlines if needed.",
                
                "i feel like i'm failing at everything": "It sounds like you're carrying a lot right now. Remember, failure is an experience, not your identity. Can you think of one thing that went well today, no matter how small? Taking time to acknowledge your efforts is important.",

                # Life transitions
                "i don't know where my life is going": "It's okay to feel lost—life is full of uncertainties. Could you try focusing on one thing that brings you joy or curiosity? Sometimes exploring new activities or reflecting on your values can help you find a sense of direction.",
                
                "i'm scared of change": "Change can be so intimidating, but it's also a chance for growth. Could you focus on one small step forward? You're not alone in this, and reaching out for support can make the transition feel more manageable.",

                # Sleep issues
                "i have nightmares every night": "Nightmares can leave you feeling exhausted and unsettled. Would you consider keeping a journal of your dreams and talking to a therapist about them? Creating a calming bedtime routine or trying grounding techniques can also help.",
                
                "i can't wake up in the morning": "Struggling to wake up can be tied to your sleep patterns or emotional state. Could we try adjusting your routine with things like setting a consistent bedtime, exposure to natural light in the morning, or a gentle alarm? Talking to a healthcare provider might also help uncover underlying causes."
                       
            }
            
            for keyword, response in response_map.items():
                if keyword in normalized_message:
                    return response
            
            return "I'm here to help you. Could you share more about what you're feeling?"
            
        except Exception as e:
            logger.error(f"Error in local response generation: {str(e)}")
            return "I'm here to listen. Would you like to tell me more?"

    def create_response(self, response_text: str) -> JsonResponse:
        """Create JSON response with emotion detection."""
        try:
            blob = TextBlob(response_text)
            sentiment = blob.sentiment.polarity
            emotion = 'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'
            
            return JsonResponse({
                'response': response_text,
                'emotion': emotion
            })
        except Exception as e:
            logger.error(f"Error in create_response: {str(e)}")
            return JsonResponse({
                'response': response_text,
                'emotion': 'neutral'
            })



openai.api_key = os.getenv('OPENAI_API_KEY', 'your-fallback-api-key')

def landing_page(request):
    return render(request, 'landing.html')

def landing(request):
    return render(request, 'landing.html')

def about(request):
    return render(request, 'about.html')

def privacy(request):
    return render(request, 'privacy.html')

def terms(request):
    return render(request, 'terms.html')

def contact(request):
    return render(request, 'contact.html')

def resources(request):
    return render(request, 'resources.html')  

from django.contrib.auth.views import LoginView
from django.urls import reverse_lazy

class StudentLoginView(LoginView):
    template_name = 'login.html'
    redirect_authenticated_user = True
    next_page = reverse_lazy('dashboard')  # Replace with the student's dashboard or home page

@login_required
def dashboard(request):
    print(f"Is user authenticated? {request.user.is_authenticated}")
    print(f"Profile picture path: {request.user.student_profile.profile_picture}")
    print(f"Media URL: {settings.MEDIA_URL}")
    print(f"Media Root: {settings.MEDIA_ROOT}")
    context = {
    'MEDIA_URL': settings.MEDIA_URL,
    }
    return render(request, 'dashboard.html', context)

def logout_view(request):
    logout(request)  # Use the `logout` function from django.contrib.auth
    return redirect('landing_page')

from .models import Feedback

class FeedbackView(View):
    def post(self, request, *args, **kwargs):
        form = FeedbackForm(request.POST)
        if form.is_valid():
            feedback = form.save(commit=False)  # Create a Feedback instance without saving to the database yet
            feedback.user = request.user  # Associate the feedback with the logged-in user
            feedback.save()  # Now save it to the database
            return JsonResponse({'success': True, 'message': "Thank you for your feedback!"})

        return JsonResponse({'success': False, 'errors': form.errors})
