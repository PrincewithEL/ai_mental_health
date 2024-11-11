from django.contrib import admin
from .models import StudentProfile
from .models import Feedback

class FeedbackAdmin(admin.ModelAdmin):
    list_display = ('message', 'created_at')  # Columns to display in the admin list view
    ordering = ('-created_at',)  # Order by creation date, newest first

admin.site.register(Feedback, FeedbackAdmin)

admin.site.register(StudentProfile)