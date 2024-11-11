from django.db import models
from django.contrib.auth.models import User

class StudentProfile(models.Model):
    # Link to the Django User model for authentication
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='student_profile')
    
    # Student-specific fields
    course = models.CharField(max_length=100, null=True, blank=True)
    year_of_study = models.IntegerField(null=True, blank=True)
    enrolled_date = models.DateField(auto_now_add=True)
    profile_picture = models.ImageField(upload_to='profile_pictures/', null=True, blank=True)
    
    def __str__(self):
        return f"{self.user.username} - Student"

class Feedback(models.Model):
    message = models.TextField()
    user = models.OneToOneField(User, on_delete=models.CASCADE,blank=True, null=True)    
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback: {self.message[:20]}..."


