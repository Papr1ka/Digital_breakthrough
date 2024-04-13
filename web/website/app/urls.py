from django.urls import path
from .views import (
    HomeView,
)

urlpatterns = [
    #общие страницы
    path('', HomeView.as_view(), name="home"),
]
