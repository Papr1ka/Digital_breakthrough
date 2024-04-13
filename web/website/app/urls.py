from django.urls import path
from .views import (
    HomeView,
    HTMXImages
)

urlpatterns = [
    #общие страницы
    path('', HomeView.as_view(), name="home"),
    path('htmx_images', HTMXImages.as_view(), name="images"),
]
