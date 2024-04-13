from django.db import models
from dataclasses import dataclass


# Create your models here.
@dataclass
class MuseumObject:
    image: str # path
    group: str
    description: str
