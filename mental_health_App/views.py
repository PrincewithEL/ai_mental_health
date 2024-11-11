from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.conf import settings
from django.shortcuts import redirect
from django.http import JsonResponse
from .forms import *

def landing_page(request):
    return render(request, 'landing.html')

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

from django.http import JsonResponse
from django.views import View
from .ai_module.emotion_analysis import load_response_data, get_best_response
import logging

class EmotionResponseView(View):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            # Load data and initialize models
            load_response_data()
        except Exception as e:
            logging.error(f"Error initializing EmotionResponseView: {str(e)}")
    
    def get(self, request, *args, **kwargs):
        try:
            user_message = request.GET.get('message', '')
            
            # Get response
            response = get_best_response(user_message)
            
            return JsonResponse({
                'response': response,
                'emotion': 'neutral'  # Replace with actual emotion if emotion detection is implemented
            })
            
        except Exception as e:
            logging.error(f"Error in EmotionResponseView: {str(e)}")
            return JsonResponse({
                'response': "I'm here to support you. Could you please share more about what you're feeling?",
                'emotion': 'neutral'
            })

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
