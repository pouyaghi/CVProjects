import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
#Dataset path = /home/pouya/.cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1

data_path = "/home/pouya/.cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1"

data_dir = os.path.join(data_path, "asl_alphabet_train")

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)