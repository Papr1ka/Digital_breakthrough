from django.db import models
from dataclasses import dataclass
from PIL import Image


# Create your models here.
@dataclass
class MuseumObject:
    image: str # path
    name: str
    group: str
    description: str


@dataclass
class PredictedImage:
    image: Image
    group: str
    description: str
