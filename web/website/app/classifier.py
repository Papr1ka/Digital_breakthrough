import os
import torch
import torch.nn as nn
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.utils.data import random_split

from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from transformers import Trainer, TrainingArguments
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from transformers import get_scheduler
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
class MuseumDataset(torch.utils.data.Dataset):
    """Датасет с фотками музея."""
    def __init__(self, csv_file, root_dir, transform=None):
        super(MuseumDataset, self).__init__()
        self.museum_items = pd.read_csv(csv_file, sep = ';', encoding = 'utf-8')
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = self.museum_items['group'].unique()
        
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file = os.path.join(self.root_dir, 
                                str(self.museum_items.iloc[idx, 0]), self.museum_items.iloc[idx, 4])

        image = Image.open(img_file)
        items = self.museum_items.iloc[idx, 3]
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'class': items}
        
        class_idx = np.where(self.class_names == sample['class'])[0][0]
        class_name = torch.tensor(class_idx)

        return sample['image'], class_name


    def __len__(self):
        return len(self.museum_items)

trans = transforms.Compose([
    transforms.Resize((224, 224)),  # изменение размера изображения на 224x224 пикселей
    transforms.ToTensor(),
    
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


df_train = MuseumDataset("train_dataset_mincult-train/train.csv", os.path.join(os.getcwd(), "train_dataset_mincult-train", "train"), trans)
train_dataloader = DataLoader(df_train, batch_size=8,
                        shuffle=True, num_workers=0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

image_batch, label_batch = next(iter(train_dataloader))

image, label = image_batch[0], label_batch[0]

print(image.shape, label)
print(df_train.class_names)
plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
plt.axis(False);
class MyViTWithMLP(nn.Module):
    def __init__(self, vit_model, num_classes=15):
        super(MyViTWithMLP, self).__init__()
        self.vit_model = vit_model
        self.hidden_size = vit_model.config.hidden_size
        self.mlp_head = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.vit_model(x)
        output = self.mlp_head(features.last_hidden_state[:, 0])
        return output

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for batch_idx, (images, labels) in enumerate(train_dataloader):
    images = images.to(device)

    inputs = processor(images=images, return_tensors="pt").to(device)

    
    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
print(len(train_dataloader))
print(last_hidden_states[:10])
torch.save(last_hidden_states, 'last_hidden_states3.pt')
model.save_pretrained('vit3')

classifier = MyViTWithMLP(model, num_classes=15).to(device)

print(classifier.hidden_size)
num_epochs = 3
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(classifier.parameters(), lr=5e-5)

for epoch in range(num_epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(train_dataloader)
    for batch in progress_bar:
        images, labels = batch
        labels = torch.tensor(labels).to(device)

        images = images.to(device)

        outputs = classifier(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_dataloader)}")

torch.save(classifier.state_dict(), 'classifier.pth')
trans = transforms.Compose([
    transforms.Resize((224, 224)),  # изменение размера изображения на 224x224 пикселей
    transforms.ToTensor(),
    
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


df_train = MuseumDataset("train_dataset_mincult-train/train.csv", os.path.join(os.getcwd(), "train_dataset_mincult-train", "train"), trans)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_size = int(0.8 * len(df_train))  
test_size = len(df_train) - train_size  

train_dataset, test_dataset = random_split(df_train, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)


image_batch, label_batch = next(iter(train_dataloader))


image, label = image_batch[0], label_batch[0]


print(image.shape, label)
print(df_train.class_names)
plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
plt.axis(False);


last_hidden_states = torch.load('last_hidden_states.pt')

classifier2 = MyViTWithMLP(model, num_classes=15)


num_epochs = 3
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(classifier2.parameters(), lr=5e-5)

for epoch in range(num_epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(train_dataloader)
    for batch in progress_bar:
        images, labels = batch
        labels = torch.tensor(labels).to(device)

        images = images.to(device)

        outputs = classifier2(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_dataloader)}")
torch.save(classifier.state_dict(), 'classifier2.pth')
