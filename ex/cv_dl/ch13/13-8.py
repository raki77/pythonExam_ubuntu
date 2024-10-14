import keras_cv
import matplotlib.pyplot as plt

model=keras_cv.models.StableDiffusion(img_width=512,img_height=512)
img=model.text_to_image('A cute rabbit in an avocado armchair',batch_size=3)

for i in range(len(img)):
    plt.subplot(1,len(img),i+1)
    plt.imshow(img[i])
    plt.axis('off')