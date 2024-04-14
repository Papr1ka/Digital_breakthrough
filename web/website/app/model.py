import PIL.Image
import pandas as pd
import random
from os import path
from typing import List, Tuple
from .models import MuseumObject, PredictedImage
from django.conf import settings
import PIL
from .classifier import predict_image_class
from .predictAll import task


def predict(image: PIL.Image.Image, images: bool, group: bool, description: bool) -> dict:
    return task.predict(image, images, group, description)
