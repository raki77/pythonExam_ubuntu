from transformers import ViTFeatureExtractor, TFViTForImageClassification
from PIL import Image

img=[Image.open('BSDS_242078.jpg'),Image.open('BSDS_361010.jpg'),Image.open('BSDS_376001.jpg')]

feature_extractor=ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model=TFViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs=feature_extractor(img,return_tensors='tf')
res=model(**inputs)

import tensorflow as tf
import matplotlib.pyplot as plt

for i in range(res.logits.shape[0]):
    plt.imshow(img[i]); plt.xticks([]); plt.yticks([]); plt.show()
    predicted_label=int(tf.math.argmax(res.logits[i],axis=-1))
    prob=float(tf.nn.softmax(res.logits[i])[predicted_label]*100.0)
    print(i,'번째 영상의 1순위 부류: ',model.config.id2label[predicted_label],prob)