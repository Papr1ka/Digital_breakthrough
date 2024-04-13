from django.db import models
from dataclasses import dataclass
from PIL.Image import Image


@dataclass
class MuseumObject:
    image: str  #  Относительный путь до изображения в формате {object_id}/img_name
    name: str
    group: str
    description: str


@dataclass
class PredictedImage:
    image: Image
    group: str
    description: str
