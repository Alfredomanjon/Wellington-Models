#Generador de pares 
from imutils import build_montages
import numpy as np
import cv2
import tensorflow as tf

def make_pairs(images, labels):
    pairImages = []
    pairLabels = []
    numClasses = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        idxB = np.random.choice(idx[label])
        posImage = images[idxB]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]
        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
    # return a 2-tuple of our image pairs and labels
    return (np.array(pairImages), np.array(pairLabels))

# load MNIST dataset and scale the pixel values to the range of [0, 1]
print("[INFO] loading MNIST dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='/Users/alfredo/Desktop/Wellington/WellingtonV1/dataset/training_data',
    labels='inferred',
    image_size=(180, 180),
    color_mode= 'rgb')

(trainX, trainY), (testX, testY) = train_ds

print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = make_pairs(trainX, trainY)

images = []

# loop over a sample of our training pairs
for i in np.random.choice(np.arange(0, len(pairTrain)), size=(20,)):
    imageA = pairTrain[i][0]
    imageB = pairTrain[i][1]
    label = labelTrain[i]

    output = np.zeros((188, 368, 3), dtype="uint8")
    pair = np.hstack([imageA, imageB])
    output[8:188, 8:368] = pair

    text = "neg" if label[0] == 0 else "pos"
    color = (255, 0, 0) if label[0] == 0 else (0, 255, 0)

    vis = cv2.merge([output])
    cv2.putText(vis, text, (8, 25), cv2.FONT_HERSHEY_COMPLEX, 0.95,
        color, 2)   
    # add the pair visualization to our list of output images
    images.append(vis)


# construct the montage for the images
montages = build_montages(images, (360, 180), (6, 5))
 
for montage in montages:
    tf.keras.utils.array_to_img(montage).show()

