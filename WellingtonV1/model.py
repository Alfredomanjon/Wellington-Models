import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

batch_size = 32
img_height = 180
img_width = 180

def showDataSet():
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        break
    plt.show()

def augShow():
    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
    plt.show()

def loadData():
    global train_ds
    train_ds = tf.keras.utils.image_dataset_from_directory('/Users/alfredo/Desktop/Wellington/WellingtonV1/dataset/training_data', image_size=(img_height, img_width), batch_size=batch_size)

    global val_ds                                                
    val_ds = tf.keras.utils.image_dataset_from_directory('/Users/alfredo/Desktop/Wellington/WellingtonV1/dataset/training_data', image_size=(img_height, img_width), batch_size=batch_size) 

    global class_names  
    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))

    global num_classes
    num_classes = len(class_names)

    global data_augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])   

def createModel():

    global model
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.summary()


def fit():
    epochs = 15
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    model.save('/Users/alfredo/Desktop/Wellington/WellingtonV1/models')


def predict(url):

    new_model = tf.keras.models.load_model('/Users/alfredo/Desktop/Wellington/WellingtonV1/models')

    img = tf.keras.utils.load_img(
        url , target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


if __name__ == "__main__":
    loadData()
    createModel()
    #fit()
    #predict("/Users/alfredo/Desktop/TFG-RC/Predicts/cake1.jpeg")
    #predict("/Users/alfredo/Desktop/TFG-RC/Predicts/predict3.jpeg")
    #predict("/Users/alfredo/Desktop/TFG-RC/Predicts/predict4.jpeg")
    #predict("/Users/alfredo/Desktop/Wellington/WellingtonV1/Predicts/predict6.jpeg")