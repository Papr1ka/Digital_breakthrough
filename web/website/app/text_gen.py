# Load model directly
from transformers import AutoTokenizer, AutoModel
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
import nltk
import evaluate
from PIL import Image
tokenizer = AutoTokenizer.from_pretrained("tuman/vit-rugpt2-image-captioning")
model = VisionEncoderDecoderModel.from_pretrained("tuman/vit-rugpt2-image-captioning", cache_dir='E:\\ggames')
feature_extractor = ViTFeatureExtractor.from_pretrained("tuman/vit-rugpt2-image-captioning", cache_dir='E:\\ggames')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
dt = pd.read_csv('train.csv', sep=';', encoding='utf-8')
path_train = '/home/jupyter/datasphere/project/train'
dt_non_nan = dt[dt['description'].notna()].reset_index()
dt_non_nan['path'] = path_train + '/' + (dt_non_nan['object_id']).astype(str) + '/' + dt_non_nan['img_name']
dt_non_nan = dt_non_nan.drop(columns=['index', 'name', 'group', 'img_name', 'object_id'])
train_x, test_x, train_y, test_y = train_test_split(dt_non_nan['path'].values, 
                                                    dt_non_nan['description'].values, 
                                                    test_size=0.1)
test_x, valid_x, test_y, valid_y = train_test_split(test_x, 
                                                    test_y, 
                                                    test_size=0.05)
def get_pixels(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  return pixel_values

def get_label(texts, max_target_length):
    return tokenizer(texts,
                     return_tensors='pt',
                     padding='max_length',
                     max_length=max_target_length,
                     truncation=True).input_ids
from torch.utils.data import Dataset
class Custom(Dataset):
    def __init__(self, X, y, max_target_length=128):
        self.max_target_length = max_target_length
        self.all_files = X
        self.all_texts = y

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        pixel_values = get_pixels([self.all_files[idx]])
        # add labels (input_ids) by encoding the text
        labels = get_label(self.all_texts[idx], self.max_target_length)[0]
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
train_dataset = Custom(train_x, train_y)
test_dataset = Custom(test_x, test_y)
valid_dataset = Custom(valid_x, valid_y)
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    preds = ["\n".join(nltk.sent_tokenize(pred, language='russian')) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label, language='russian')) for label in labels]

    return preds, labels
import evaluate
metric = evaluate.load("rouge")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels,
                            use_stemmer=True)
    return result
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    eval_steps=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    output_dir="./image-captioning-output",
    save_steps=3000,
    report_to='clearml'
)

from transformers import default_data_collator

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=default_data_collator,
)
trainer.train(resume_from_checkpoint=True)

