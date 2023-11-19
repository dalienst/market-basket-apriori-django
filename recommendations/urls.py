from django.urls import path
from .views import recommend

app_name = "recommendations"

urlpatterns = [
    path('recommend/', recommend, name='recommend'),
]