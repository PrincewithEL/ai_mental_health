from django.contrib import admin
from django.http import HttpResponse
import csv
from datetime import datetime
from .models import StudentProfile, Feedback

class FeedbackAdmin(admin.ModelAdmin):
    list_display = ('message', 'created_at')
    ordering = ('-created_at',)
    
    # Add actions to the admin
    actions = ['export_as_csv', 'generate_feedback_summary']
    
    def export_as_csv(self, request, queryset):
        # Create the HttpResponse object with CSV header
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename=feedback_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        # Create CSV writer
        writer = csv.writer(response)
        
        # Write header row
        writer.writerow(['Message', 'Created At'])
        
        # Write data rows
        for feedback in queryset:
            writer.writerow([feedback.message, feedback.created_at])
            
        return response
    
    export_as_csv.short_description = "Export selected feedback to CSV"
    
    def generate_feedback_summary(self, request, queryset):
        # Create the HttpResponse object
        response = HttpResponse(content_type='text/plain')
        response['Content-Disposition'] = f'attachment; filename=feedback_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        # Generate summary statistics
        total_feedback = queryset.count()
        latest_feedback = queryset.order_by('-created_at').first()
        oldest_feedback = queryset.order_by('created_at').first()
        
        # Write summary to response
        summary = [
            "Feedback Summary Report",
            "===================",
            f"Total Feedback Items: {total_feedback}",
            f"Date Range: {oldest_feedback.created_at.date() if oldest_feedback else 'N/A'} to {latest_feedback.created_at.date() if latest_feedback else 'N/A'}",
            "\nFeedback Messages:",
            "----------------"
        ]
        
        for feedback in queryset:
            summary.append(f"\n{feedback.created_at.strftime('%Y-%m-%d %H:%M')}: {feedback.message}")
            
        response.write("\n".join(summary))
        return response
    
    generate_feedback_summary.short_description = "Generate feedback summary report"

class StudentProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'get_email')
    actions = ['export_profiles_csv']
    
    def get_email(self, obj):
        return obj.user.email
    get_email.short_description = 'Email'
    
    def export_profiles_csv(self, request, queryset):
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename=student_profiles_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        writer = csv.writer(response)
        writer.writerow(['Username', 'Email'])
        
        for profile in queryset:
            writer.writerow([profile.user.username, profile.user.email])
            
        return response
    
    export_profiles_csv.short_description = "Export selected profiles to CSV"

admin.site.register(Feedback, FeedbackAdmin)
admin.site.register(StudentProfile, StudentProfileAdmin)