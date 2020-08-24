import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image

import keras
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Sequential
from sklearn.utils import class_weight
from keras.constraints import maxnorm
from sklearn.metrics import roc_curve
from keras.utils.vis_utils import plot_model
from ann_visualizer.visualize import ann_viz
from keras.layers.convolutional import Conv2D
from sklearn.preprocessing import LabelEncoder
from keras.layers.convolutional import MaxPooling2D
from sklearn.model_selection import train_test_split

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# plt.style.use('ggplot')


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

    # Needed to replace jpg to jpeg file typpe
    csv1["image"] = csv1["image"].str.replace("jpg","jpeg")

    # pie_plot(csv1["emotion"])

    # Add dir into image name
    csv1["image"] = "data/images/" + csv1["image"]

    # Remove no needed row
    csv1 = csv1[csv1.image != 'data/images/facial-expressions_2868588k.jpeg']
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
    return csv1
    # all_dir_file = ["data/images/" + f for f in os.listdir("data/images") if
    #                 os.path.isfile(os.path.join("data/images", f))]
    # csv_list = csv1["image"].to_list()
    # remaining_images = list(set(all_dir_file) - set(csv_list))
    # csv1 = csv1.drop("exists", axis=1)
    # return csv1, remaining_images


def pie_plot(data):
    labels = 'neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt'

    fig1, ax1 = plt.subplots()
    ax1.pie(data.value_counts(normalize=True), labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, pctdistance=1.2, labeldistance=1.3)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(15, 15)
    plt.savefig("target_pie_chart.png", dpi=100)
    # plt.show()


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


    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=1)

    # Convert to greyscale
    # image = tf.image.rgb_to_grayscale(image)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    resized_image = tf.image.resize(image, [64, 64])

    return resized_image, label


def data_pipeline(filenames, labels):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(parse_function, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


def create_model_1(num_classes, metrics):
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), input_shape=(64, 64, 1), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam(lr=1e-3), metrics=metrics)
    return model


def create_model_2(num_classes, metrics):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu', padding='same'))
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
    model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=keras.optimizers.Adam(lr=1e-3),
                  metrics=metrics)
    return model


def plot_loss(history, label, n):
    # Use a log scale to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
                 color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
                 color=colors[n], label='Val ' + label,
                 linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.savefig("plots/loss_plot.png")
    plt.clf()


def plot_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
          plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
          plt.ylim([0.8, 1])
        else:
          plt.ylim([0, 1])

        plt.legend()
    plt.savefig("plots/metrics_plot.png")
    plt.clf()


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 20])
    plt.ylim([80, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


# class CategoricalTruePositives(tf.keras.metrics.Metric):
#
#     def __init__(self, num_classes, batch_size,
#                  name="categorical_true_positives", **kwargs):
#         super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
#
#         self.batch_size = batch_size
#         self.num_classes = num_classes
#
#         self.cat_true_positives = self.add_weight(name="ctp", initializer="zeros")
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = K.argmax(y_true, axis=-1)
#         y_pred = K.argmax(y_pred, axis=-1)
#         y_true = K.flatten(y_true)
#
#         true_poss = K.sum(K.cast((K.equal(y_true, y_pred)), dtype=tf.float32))
#
#         self.cat_true_positives.assign_add(true_poss)
#
#     def result(self):
#         return self.cat_true_positives


def main():
    data = preprocess("data/legend.csv")

    # Fixed data folder for tensorflow preprocessing
    # create_dir(data)

    # Analysis on dataset
    # data, remaining = exploration(data)
    # test = exploration(data)
    # Plot image from each class
    # target_plot(data)

    # Split dataset to train, val and test set
    X_train, X_val, y_train, y_val = train_test_split(data.image, data.emotion, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)
    y_val = encoder.transform(y_val)
    y_test = encoder.transform(y_test)

    image = parse_function('data/images/facial-expressions_2868582k.jpeg', "check")
    myarr = np.asarray(image[0])
    final = myarr.reshape(64, 64)
    plt.imshow(final)
    plt.savefig("processed_image.jpg")
    plt.show()

    # Build data image pipeline for train, val and test set
    train_dataset = data_pipeline(X_train, y_train)
    val_dataset = data_pipeline(X_val, y_val)
    test_dataset = data_pipeline(X_test, y_test)

    metrics = keras.metrics.CategoricalAccuracy(name='accuracy')

    # metrics = [
    #     keras.metrics.TruePositives(name='tp'),
    #     keras.metrics.FalsePositives(name='fp'),
    #     keras.metrics.TrueNegatives(name='tn'),
    #     keras.metrics.FalseNegatives(name='fn'),
    #     keras.metrics.CategoricalAccuracy(name='accuracy'),
    #     keras.metrics.Precision(name='precision'),
    #     keras.metrics.Recall(name='recall'),
    #     keras.metrics.AUC(name='auc'),
    #     ]

    # Get early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_acc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

    # Define checkpoints
    checkpoint_path = "model/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)

    model = create_model_1(np.unique(y_train).size, metrics)
    model.summary()
    plot_model(model, to_file='results/model_plot.png', show_shapes=True, show_layer_names=True)
    # ann_viz(model, filename="results/neural_network.png", title="NN Architecture")

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = dict(enumerate(class_weights))
    # Train model
    training_history = model.fit(train_dataset, epochs=500, batch_size=batch_size,
                                 callbacks=[early_stopping, cp_callback],
                                 validation_data=val_dataset, class_weight=class_weights)

    # Plot loss
    plot_loss(training_history, "Training Loss", 0)

    # Plot metrics
    # plot_metrics(training_history)

    # Perform evaluations on test set
    train_predictions_baseline = model.predict(train_dataset)
    test_predictions_baseline = model.predict(test_dataset)

    # Evaluate test set
    baseline_results = model.evaluate(test_dataset, verbose=0)

    # Get metrics
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)

    # Plot confusion matrix
    # plot_cm(y_test, test_predictions_baseline)

    # Plot ROC curve
    plot_roc("Train Baseline", y_train, train_predictions_baseline, color=colors[0])
    plot_roc("Test Baseline", y_test, test_predictions_baseline, color=colors[0], linestyle='--')
    plt.legend(loc='lower right')
    plt.savefig("plots/roc_plot.png")

if __name__ == "__main__":
    main()
