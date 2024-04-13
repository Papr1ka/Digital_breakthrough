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


def split_rows(array: List, groups: int = 3, shift=1) -> List[List[Tuple[int, Any]]]:
    """
    Разбивает массив на groups подмассивов (чтобы отображать по столбцам, где столбец - i-я группа)
    1-й столбец = первая группа
    Элемент = кортеж из номер + shift, объект (картинка)
    """
    for i in enumerate(array):
        array[i[0]] = i[0] + shift, i[1]
    return [array[group::groups] for group in range(groups)]


def get_image_bytes(image: InMemoryUploadedFile) -> BytesIO:
    buffer = BytesIO()
    for i in image.chunks():
        buffer.write(i)
    return buffer


def encode_image_src_base64(buffer: BytesIO):
    return "data:image/png;base64," + str(base64.b64encode(buffer.getvalue()))[2:-1]
        

DEBUG = False


class HomeView(View):
    template_name = "app/home.html"

    def get(self, request):
        form = ImageForm()
        return render(request, self.template_name, context={"images": [], "form": form})
    
    def post(self, request):
        form: ImageForm = ImageForm(request.POST, request.FILES)
        data = dict(form=form)
        
        if form.is_valid():
            # В модель
            images = form.cleaned_data["images"]
            group = form.cleaned_data["group"]
            description = form.cleaned_data["description"]
            print(images, group, description)
            
            image = form.fields['image'].to_python(form.files['image'])
            
            buffer = get_image_bytes(image)
            image_pil = Image.open(buffer)
            
            data = predict(image_pil, images, group, description)
            images = data.get("images")
            if images is not None:
            
                images = [
                    MuseumObject(
                        get_image_path(
                            path.join(str(i[0]), str(i[4]))
                        ),
                        str(i[1]),
                        i[3],
                        str(i[2]) if str(i[2]) != "nan" else ""
                    ) for i in images
                ]
                data.update(
                    images={
                        "main": images[0],
                        "relative": split_rows(images[1:], shift=2)
                    }
                )
            group = data.get("group", "")
            description = data.get("description", "")
            if group or description:

                predicted_image = PredictedImage(
                    encode_image_src_base64(buffer),
                    data.get("group", ""),
                    data.get("description", "")
                )
                data.update(predicted=predicted_image)
            
            data.update(form=ImageForm())
        return render(request, self.template_name, context=data)
