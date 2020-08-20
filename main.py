import os
import PIL
import shutil
import PIL.Image as img
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
plt.style.use('ggplot')


batch_size = 32
img_height = 350
img_width = 350
data_dir = "data/images/"


def preprocess(csv_path):
    csv1 = pd.read_csv(csv_path)
    csv1 = csv1.drop("user.id", axis=1)
    map_dict = {"HAPPINESS": "happiness", "DISGUST": "disgust", "NEUTRAL": "neutral",
                "SADNESS": "sadness", "FEAR": "fear", "SURPRISE": "surprise", "ANGER": "anger"}

    csv1["emotion"] = csv1["emotion"].replace(map_dict)
    print(csv1["emotion"].value_counts(normalize=True))

    pie_plot(csv1["emotion"])

    # Add dir into image name
    csv1["image"] = "data/images/" + csv1["image"]

    # Shuffle dataset
    csv1.sample(n=1, random_state=1)
    return csv1


def exploration(csv1):
    # Check if all images in csv match the ones in file
    def check_exist(x):
        return os.path.isfile(x)

    csv1["exists"] = csv1["image"].apply(check_exist)
    csv_len = len(csv1)
    csv_filter = len(csv1[csv1["exists"] == True])

    all_dir_file = ["data/images/" + f for f in os.listdir("data/images") if
                    os.path.isfile(os.path.join("data/images", f))]
    csv_list = csv1["image"].to_list()
    remaining_images = list(set(all_dir_file) - set(csv_list))
    csv1 = csv1.drop("exists", axis=1)
    return csv1, remaining_images


def pie_plot(data):
    labels = 'neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt'

    fig1, ax1 = plt.subplots()
    ax1.pie(data.value_counts(normalize=True), labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, pctdistance=1.2, labeldistance=1.3)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 15)
    plt.savefig("target_pie_chart.png", dpi=100)
    plt.show()


def target_plot(data):
    target_dict = {"anger": 0,
                   "surprise": 1,
                   "disgust": 2,
                   "fear": 3,
                   "neutral": 4,
                   "happiness": 5,
                   "sadness": 157,
                   "contempt": 872
                   }
    plt.figure(figsize=(10, 10))
    i = 0
    for label, num in target_dict.items():
        plt.subplot(3, 3, i + 1)
        img = plt.imread(data.loc[num].image)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")
        i += 1
    plt.savefig("target.jpg")
    plt.show()


def create_dir(csv1):
    classes = ['data/images/neutral/', 'data/images/happiness/', 'data/images/surprise/',
               'data/images/sadness/', 'data/images/anger/', 'data/images/disgust/',
               'data/images/fear/', 'data/images/contempt/']

    other_dir = "data/test/"
    if not os.path.isdir(other_dir):
        os.mkdir(other_dir)
    for class_folder in classes:
        if not os.path.isdir(class_folder):
            os.mkdir(class_folder)

    def move_file(x):
        if os.path.isfile(x[0]):
            new_location = classes[[i for i, s in enumerate(classes) if x[1] in s][0]]
            if not os.path.isfile(new_location):
                shutil.move(x[0], new_location)

    csv1.apply(move_file, axis=1)

    for file in os.listdir("data/images/"):
        if file.endswith(".jpg"):
            shutil.move("data/images/" + file, other_dir)


def parse_function(filename, label):
    # image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(filename, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    resized_image = tf.image.resize_images(image, [350, 350])

    return resized_image, label


def data_pipeline(filenames, labels):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(parse_function, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


def create_model_1(num_classes):
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def create_model_2(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def main():
    data = preprocess("data/legend.csv")

    # Fixed data folder for tensorflow preprocessing
    # create_dir(data)

    # Analysis on dataset
    # data, remaining = exploration(data)

    # Plot image from each class
    # target_plot(data)

    # Split dataset to train, val and test set
    X_train, X_val, y_train, y_val = train_test_split(data.image, data.emotion, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    # Build data image pipeline for train, val and test set
    train_dataset = data_pipeline(X_train, y_train)
    val_dataset = data_pipeline(X_val, y_val)
    test_dataset = data_pipeline(X_test, y_test)


if __name__ == "__main__":
    main()
