from django.shortcuts import render
from django.views import View
from .model import predict
from .models import MuseumObject
from typing import List, Tuple, Any
from os import path
from django.conf import settings
from .forms import ImageForm
from PIL import Image
from django.core.files.uploadedfile import InMemoryUploadedFile


def get_image_path(relative_path: str):
    return path.join(settings.MEDIA_URL, "train", relative_path)


def handle_file(file: InMemoryUploadedFile) -> Image:
    with Image.open(file) as image:
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

DEBUG = True

class HomeView(View):
    template_name = "app/home.html"

    def get(self, request):
        form = ImageForm()
        if DEBUG:
            images: List[MuseumObject] = predict(None)
            for image in images:
                image.image = get_image_path(image.image)
            return render(request, self.template_name, context={"images": [], "form": form, "images": images, "test": split_rows(images)}) 
            
        return render(request, self.template_name, context={"images": [], "form": form})

    def post(self, request):
        form = ImageForm(request.POST, request.FILES)
        data = dict(form=form)
        
        if form.is_valid():
            image = handle_file(request.FILES["image"])
            images: List[MuseumObject] = predict(image)
            for image in images:
                image.image = get_image_path(image.image)
            data.update(images=images)
        else:
            data.update(message="Файл некорректен")
            
        return render(request, self.template_name, context=data)
    