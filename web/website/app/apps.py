from django.apps import AppConfig
from django.conf import settings


class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'
    PATH_TO_DATASET = settings.PATH_TO_DATASET
