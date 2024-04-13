import PIL.Image
import pandas as pd
import random
from os import path
from typing import List, Tuple
from .models import MuseumObject, PredictedImage
from django.conf import settings
import PIL

PATH_TO_DATASET = settings.PATH_TO_DATASET
PATH_TO_CSV = path.join(PATH_TO_DATASET, "train.csv")

df = pd.read_csv(PATH_TO_CSV, sep = ';', encoding = 'utf-8')

def predict(image: PIL.Image.Image) -> Tuple[PredictedImage, List[MuseumObject]]:
    # assert isinstance(image, Image), type(image)
    object = PredictedImage(image, "Предсказанный класс", 'Lorem Ipsum - это текст-"рыба", часто используемый в печати и вэб-дизайне. Lorem Ipsum является стандартной "рыбой" для текстов на латинице с начала XVI века. В то время некий безымянный печатник создал большую коллекцию размеров и форм шрифтов, используя Lorem Ipsum для распечатки образцов. Lorem Ipsum не только успешно пережил без заметных изменений пять веков, но и перешагнул в электронный дизайн. Его популяризации в новое время послужили публикация листов Letraset с образцами Lorem Ipsum в 60-х годах и, в более недавнее время, программы электронной вёрстки типа Aldus PageMaker, в шаблонах которых используется Lorem Ipsum.')
    
    indexes = [random.randint(0, len(df)) for i in range(10)]
    answer = df.iloc[indexes]
    answer = [MuseumObject(path.join(str(i[0]), str(i[4])), str(i[1]), i[3], str(i[2]) if str(i[2]) != "nan" else "") for i in answer.values]
    return object, answer
