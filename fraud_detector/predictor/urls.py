from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("predict", views.predict, name="predict"),
    path("api/predict", views.predict_api, name="predict_api"),
    path("health", views.health_check, name="health_check"),
]