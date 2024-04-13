import pandas as pd
import random
from os import path
from typing import List
from .models import MuseumObject
from django.conf import settings
from PIL import Image

PATH_TO_DATASET = settings.PATH_TO_DATASET
PATH_TO_CSV = path.join(PATH_TO_DATASET, "train.csv")

df = pd.read_csv(PATH_TO_CSV, sep = ';', encoding = 'utf-8')

def predict(image: Image) -> List[MuseumObject]:
    indexes = [random.randint(0, len(df)) for i in range(10)]
    answer = df.iloc[indexes]
    answer = [MuseumObject(path.join(str(i[0]), str(i[4])), i[3], str(i[2]) if str(i[2]) != "nan" else "") for i in answer.values]
    return answer
