from django.shortcuts import render
from django.views import View
from .model import predict
from .models import MuseumObject
from typing import List
from os import path
from django.conf import settings

sample_data = [
    
]

def get_image_path(relative_path: str):
    return path.join(settings.MEDIA_URL, "train", relative_path)

# Create your views here.
class HomeView(View):
    template_name = "app/home.html"

    def get(self, request):
        
        return render(request, self.template_name, context={"images": []})

    def post(self, request):
        images: List[MuseumObject] = predict()
        for image in images:
            image.image = get_image_path(image.image)
        return render(request, self.template_name, context={"images": images})


class HTMXImages(View):
    template_name = "app/partials/images.html"
    
    def post(self, request):
        images: List[MuseumObject] = predict()
        for image in images:
            image.image = get_image_path(image.image)
        return render(request, self.template_name, context={"images": images})
    