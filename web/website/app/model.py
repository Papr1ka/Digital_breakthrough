import pandas as pd
import numpy as np
import random
from os import path
from typing import List
from .models import MuseumObject
from django.conf import settings

PATH_TO_DATASET = settings.PATH_TO_DATASET
PATH_TO_CSV = path.join(PATH_TO_DATASET, "train.csv")
PATH_TO_IMAGES = path.join(PATH_TO_DATASET, "train")

df = pd.read_csv(PATH_TO_CSV, sep = ';', encoding = 'utf-8')

def predict() -> List[MuseumObject]:
    indexes = [random.randint(0, len(df)) for i in range(10)]
    answer = df.iloc[indexes]
    
    answer = [MuseumObject(path.join(str(i[0]), str(i[4])), i[3], str(i[2]) if str(i[2]) != "nan" else "") for i in answer.values]
    for i in answer:
        assert isinstance(i.image, str), (type(i.image), "image")
        assert isinstance(i.group, str), (type(i.group), "group")
        assert isinstance(i.description, str), (type(i.description), "description")
    print(answer)
    return answer
