from django.db import migrations
from django.contrib.auth.hashers import make_password

def create_admin_user(apps, schema_editor):
    User = apps.get_model('auth', 'User')
    if not User.objects.filter(username='admin').exists():
        User.objects.create(
            username='admin',
            email='admin@gmail.com',
            password=make_password('12345Admin..'),
            is_staff=True,
            is_superuser=True
        )

def reverse_create_admin_user(apps, schema_editor):
    User = apps.get_model('auth', 'User')
    User.objects.filter(username='admin').delete()

class Migration(migrations.Migration):
    dependencies = [
        ('mental_health_App', '0001_initial'),  # Make sure this matches your app name
        ('auth', '__first__'),
    ]

    operations = [
        migrations.RunPython(create_admin_user, reverse_create_admin_user),
    ]