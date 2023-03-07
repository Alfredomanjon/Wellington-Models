# import the necessary packages
from siameseConfig import config
from siameseConfig import utils
from tensorflow.keras.models import load_model
from imutils.paths import list_images
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="/Users/alfredo/Desktop/Wellington/WellingtonV2/example_complet")
args = vars(ap.parse_args())


print("[INFO] cargando el dataset")
testImagePaths = list(list_images(args["input"]))
np.random.seed(42)
pairs = np.random.choice(testImagePaths, size=(10, 2))

print("[INFO] cargando el modelo siames")
model = load_model(config.MODEL_PATH)

for (i, (pathA, pathB)) in enumerate(pairs):
    imageA = Image.open("/Users/alfredo/Desktop/Wellington/WellingtonV2/example_wht_bg/tomate_natural14-removebg-preview.jpeg")
    imageB = Image.open(pathB)

    imageA.load()
    imageB.load()

    background1 = Image.new("RGB", imageA.size, (255, 255, 255))
    background1.paste(imageA, mask=imageA.split()[3]) 

    background2 = Image.new("RGB", imageB.size, (255, 255, 255))
    background2.paste(imageB, mask=imageB.split()[3]) 

    origA = background1.copy()
    origB = background2.copy()

    imageA = np.expand_dims(background1, axis=0) #axis=-1
    imageB = np.expand_dims(background2, axis=0) #axis=-1

    #imageA = np.expand_dims(imageA, axis=0)
    #imageB = np.expand_dims(imageB, axis=0)

    imageA = imageA / 255.0
    imageB = imageB / 255.0

    preds = model.predict([imageA, imageB])
    proba = preds[0][0]

    # initialize the figure
    fig = plt.figure("Pair #{}".format(i + 1), figsize=(4, 4))
    plt.suptitle("Similarity: {:.2f}".format(proba))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(origA, cmap=plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(origB, cmap=plt.cm.gray)
    plt.axis("off")
    
# show the plot
plt.show()
    
