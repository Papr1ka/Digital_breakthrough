# !wget https://lodmedia.hb.bizmrg.com/case_files/1080340/train_dataset_mincult-train.zip
import pandas as pd

df = pd.read_csv('train.csv', sep=';')  # Весь датасет
# ## Загрузка вит
%pip install faiss-gpu
%pip install transformers
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch
import numpy as np

# Загрузка ViT
model_name = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, output_hidden_states=True).cuda()

# ## Простой индекс без ivf на всех (типа) изображениях
import faiss

model_name = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, output_hidden_states=True).cuda()


def get_embed(image):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    image = image.resize((224, 224))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)

    return output.hidden_states[0].mean(dim=1)[0].cpu().numpy()
from tqdm import tqdm


def get_embed_all():
    image_vectors = []
    for i in tqdm(range(df.shape[0])):
        img_dir = f"train/{df.iloc[i]['object_id']}/{df.iloc[i]['img_name']}"
        image_vectors.append(get_embed(Image.open(img_dir)))
    return image_vectors
image_vectors = get_embed_all()
# # РАБОТАЮЩИЙ ИНДЕКС БОЖЕ
dim = image_vectors[0].shape[0]
num_components = 64
# создаем объект PCAMatrix
pca_matrix = faiss.PCAMatrix(dim, num_components)

image_vectors = np.array(image_vectors)
# обучаем PCAMatrix на исходных векторах
pca_matrix.train(image_vectors)

# применяем PCAMatrix для понижения размерности векторов
output_vectors = pca_matrix.apply_py(image_vectors)

index = faiss.IndexFlatL2(output_vectors[0].shape[0])

index.add(output_vectors)
faiss.write_VectorTransform(pca_matrix, "PCA.pca")
faiss.write_index(index, "working.index")
def find_simmilar_pca(image):
    index = faiss.read_index("working.index")
    PCA = faiss.read_VectorTransform("PCA.pca")
    query_image_vector = pca_matrix.apply_py(np.array([get_embed(image)]))

    k = 10
    distances, indices = index.search(query_image_vector, k)
    res = [df.iloc[i] for i in indices][0]
    return res
import matplotlib.pyplot as plt


img_dir = f"train/{df.iloc[10]['object_id']}/{df.iloc[10]['img_name']}"
image = Image.open(img_dir)
plt.imshow(image)
res = find_simmilar_pca(image).iloc[2]
img_dir = f"train/{res['object_id']}/{res['img_name']}"
image = Image.open(img_dir)
plt.imshow(image)
