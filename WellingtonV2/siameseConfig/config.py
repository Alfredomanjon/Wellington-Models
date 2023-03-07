# import the necessary packages
import os

IMG_SHAPE = (180, 180, 3)

BATCH_SIZE = 1
EPOCHS = 30

BASE_OUTPUT = "output"

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model1"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot1.png"])
