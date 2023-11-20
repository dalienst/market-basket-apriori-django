from django.urls import path
from .views import recommend, render_visualizations

app_name = "recommendations"

urlpatterns = [
    path("recommend/", recommend, name="recommend"),
    path("visualise/", render_visualizations, name="visualise"),
]
