import os
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTModel
from django.conf import settings


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

WEIGHTS_PATH = os.path.join(settings.PATH_TO_WEIGHTS, "classifier_final2.pth")

# Vmodel = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)
# model = MyViTWithMLP(Vmodel).to(device)
# model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))

im_classes = ['Археология', 'Оружие', 'Прочие', 'Нумизматика', 'Фото, негативы',
  'Редкие книги', 'Документы', 'Печатная продукция', 'ДПИ', 'Скульптура',
  'Графика', 'Техника', 'Живопись', 'Естественнонауч.коллекция', 'Минералогия']


def predict_image_class(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval() 
        output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)[0]
    predicted_class_index = torch.argmax(probabilities).item()

    return im_classes[predicted_class_index]


def getModelClassifier(path: str, vit_name: str) -> MyViTWithMLP:
    model = ViTModel.from_pretrained(vit_name)
    model.to(device)
    return MyViTWithMLP(model, num_classes=15).to(device)
