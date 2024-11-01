# api/urls.py
from django.urls import path
from .views import analyze_pig_data

urlpatterns = [
    path('analyze/', analyze_pig_data, name='analyze_pig_data'),
]
