import PIL
import torch
from image_retrieval import get_embed, feature_extractor, find_simmilar_pca
from transformers import AutoTokenizer
from transformers import VisionEncoderDecoderModel, AutoTokenizer
import faiss
import gc
from classifier import getModelClassifier, predict_image_class
from django.conf import settings
from os import path

PATH_TO_WEIGHTS = settings['PATH_TO_WEIGHTS']

path_to_classifier = path.join(PATH_TO_WEIGHTS, "classifier_final2.pth")
vit_name = 'google/vit-base-patch16-224-in21k'
encdec_name = 'tuman/vit-rugpt2-image-captioning'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class automateTask:
    def __init__(self):
        self.classifier = getModelClassifier(path_to_classifier, vit_name)
        checkpoint = torch.load(path_to_classifier, map_location=lambda storage, loc: storage)
        self.classifier.load_state_dict(checkpoint)
        self.generator = VisionEncoderDecoderModel.from_pretrained(PATH_TO_WEIGHTS, local_files_only=True).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(encdec_name)

    def predict(self, 
                image: PIL.Image.Image, 
                images: bool, 
                group: bool, 
                description: bool) -> dict:

        response = {}
        vectors = get_embed(image)
        if images:
            response['sim_images'] = find_simmilar_pca(vectors)
        if group:
            response['class'] = predict_image_class(image, self.classifier)
        if description:
            response['description'] = self.predict_caption([image])
        return response
    
    def predict_caption(self, image_paths: list, **args):
        images = []
        for img in image_paths:
            if img.mode != "RGB":
                img = img.convert(mode="RGB")

            images.append(img)

        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = self.generator.generate(pixel_values, **args)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds

task = automateTask()
