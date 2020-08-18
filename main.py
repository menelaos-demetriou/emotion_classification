import os
import PIL
import PIL.Image as img
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def main():
    csv1 = pd.read_csv("data/legend.csv")
    csv1 = csv1.drop("user.id", axis=1)
    map_dict = {"HAPPINESS": "happiness", "DISGUST": "disgust", "NEUTRAL": "neutral",
                "SADNESS": "sadness", "FEAR": "fear", "SURPRISE": "surprise", "ANGER": "anger"}

    csv1["emotion"] = csv1["emotion"].replace(map_dict)
    print(csv1["emotion"].value_counts(normalize=True))

    # pie_plot(csv1["emotion"])

    # Add dir into image name
    csv1["image"] = "data/images/" + csv1["image"]

    # Shuffle dataset
    csv1.sample(n=1, random_state=1)

    image = img.open(csv1["image"][0])
    plt.imshow(image)
    plt.show()

    # Convert all images in the same size and greyscale

    # Plot each target example image

    # Tensorflow csv dataset with batches


if __name__ == "__main__":
    main()
