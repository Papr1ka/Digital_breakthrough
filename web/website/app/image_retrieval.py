import pandas as pd
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch
import numpy as np
# Простой индекс без ivf на всех (типа) изображениях
import faiss
from tqdm import tqdm
from django.conf import settings
from os import path


PATH_TO_DATASET = settings['PATH_TO_DATASET']
PATH_TO_WEIGHTS = settings['PATH_TO_WEIGHTS']

PATH_TO_WEIGHTS = path.join(PATH_TO_WEIGHTS, "option_data")

df = pd.read_csv(path.join(PATH_TO_DATASET, "train.csv"), sep=';') 

# Загрузка ViT
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, output_hidden_states=True).to(device)

def get_embed(image):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    image = image.resize((224, 224))

    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)

    return output.hidden_states[0].mean(dim=1)[0].cpu().numpy()


image_vectors = [df.shape[0]]

dim = 768
num_components = 64
pca_matrix = faiss.PCAMatrix(dim, num_components)


def find_simmilar_pca(image):
    index = faiss.read_index("./working.index")
    PCA = faiss.read_VectorTransform(path.join(PATH_TO_WEIGHTS, "PCA.pca"))
    query_image_vector = PCA.apply_py(np.array([image])) 

    k = 10
    distances, indices = index.search(query_image_vector, k)
    res = [df.iloc[i] for i in indices][0]
    return res
