from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import StudentProfile

class StudentRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    course = forms.CharField(max_length=100)
    year_of_study = forms.IntegerField()

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.is_staff = False  # Ensures student role

        if commit:
            user.save()
            # Create StudentProfile linked to user
            StudentProfile.objects.create(
                user=user,
                course=self.cleaned_data['course'],
                year_of_study=self.cleaned_data['year_of_study']
            )
        return user

from .models import Feedback

class FeedbackForm(forms.ModelForm):
    class Meta:
        model = Feedback
        fields = ['message']  # Adjust based on your Feedback model fields


