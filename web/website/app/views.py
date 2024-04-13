from django.shortcuts import render
from django.views import View
from .model import predict
from .models import MuseumObject, PredictedImage
from typing import List, Tuple, Any
from os import path
from django.conf import settings
from .forms import ImageForm
from PIL import Image
from django.core.files.uploadedfile import InMemoryUploadedFile
import base64
from io import BytesIO


def get_image_path(relative_path: str):
    return path.join(settings.MEDIA_URL, "train", relative_path)


def handle_file(file: InMemoryUploadedFile) -> Image:
    r = file.read()
    file.size
    image = Image.frombytes()
    image.verify()
    image.save("test.png")
    return image

def split_rows(array: List, groups: int = 3, shift=1) -> List[List[Tuple[int, Any]]]:
    """
    Разбивает массив на groups подмассивов (чтобы отображать по столбцам, где столбец - i-я группа)
    1-й столбец = первая группа
    Элемент = кортеж из номер + shift, объект (картинка)
    """
    for i in enumerate(array):
        array[i[0]] = i[0] + shift, i[1]
    return [array[group::groups] for group in range(groups)]

def encode_image_src_base64(image: InMemoryUploadedFile):
    buffer = BytesIO()
    for i in image.chunks():
        buffer.write(i)
    return "data:image/png;base64," + str(base64.b64encode(buffer.getvalue()))[2:-1]


DEBUG = True

class HomeView(View):
    template_name = "app/home.html"

    def get(self, request):
        form = ImageForm()
        if DEBUG:
            predicted, images = predict(None)
            images: List[MuseumObject]
            for image in images:
                image.image = get_image_path(image.image)
            return render(request, self.template_name, context={"images": [], "form": form, "images": {
                "main": images[0],
                "relative": split_rows(images[1:], shift=2)
            }}) 
            
        return render(request, self.template_name, context={"images": [], "form": form})

    def post(self, request):
        form: ImageForm = ImageForm(request.POST, request.FILES)
        data = dict(form=form)
        
        if form.is_valid():
            print(form, dir(form), vars(form))
            image = form.fields['image'].to_python(form.files['image'])
            print(image)
            predicted_image, images = predict(image)
            print(predicted_image)
            images: List[MuseumObject]
            predicted_image: PredictedImage
            
            predicted_image.image = encode_image_src_base64(predicted_image.image)
            
            for image in images:
                image.image = get_image_path(image.image)
            data.update(images={
                "main": images[0],
                "relative": split_rows(images[1:], shift=2),
            }, predicted=predicted_image)
        else:
            data.update(message="Файл некорректен")
            
        return render(request, self.template_name, context=data)
