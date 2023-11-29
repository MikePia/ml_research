# %%
import tensorflow as tf
import tensorflow.keras as K
from sklearn.preprocessing import LabelEncoder

# from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


# %%
# Load and preprocess ISIC data
def load_isic_data(image_directory, metadata_path):
    metadata = pd.read_csv(metadata_path)
    images = []
    labels = []

    for i, row in metadata.iterrows():
        img_path = os.path.join(image_directory, row["isic_id"] + ".JPG")
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)
        labels.append(row["diagnosis"])  # Assuming 'label' column exists

    X = np.vstack(images)
    Y = np.array(labels)
    return X, Y


# %%
def preprocess_data(X, Y):
    X = K.applications.densenet.preprocess_input(X)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(Y)
    Y = K.utils.to_categorical(encoded_labels, num_classes=8)

    # Y = K.utils.to_categorical(Y, num_classes=8)
    return X, Y, label_encoder


# %%
# Load your data
image_directory = "../data/images"
metadata_path = "../data/ham10000_metadata_2023-11-27.csv"
os.getcwd()

# %%
X, Y = load_isic_data(image_directory, metadata_path)
X, Y, label_encoder = preprocess_data(X, Y)


# %%

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


# %%
input_tensor = K.Input(
    shape=(224, 224, 3)
)  # Adjusted to match the shape of your preprocessed images

model = K.applications.DenseNet201(
    include_top=False,
    weights="imagenet",
    input_tensor=input_tensor,  # Directly use the input_tensor
    pooling="max",
    classes=1000,
)

# %%

# %%
for layer in model.layers:
    layer.trainable = False
output = model.layers[-1].output
flatten = K.layers.Flatten(name="feats")
output = flatten(output)
model = K.models.Model(inputs=input_tensor, outputs=output)
model.summary()


# %%

# %%
y_test.shape

# %%
preds_train = model.predict(x_train)
preds_train.shape


# %%

# %%
preds_test = model.predict(x_test)
preds_test.shape


# %%
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=8, random_state=42, max_iter=1000, algorithm="elkan", tol=0.000001
).fit(preds_test)


# %%


# %%
from collections import Counter


def view_data(y1, preds, mapping):
    # Initialize dictionaries to store the results
    cluster_stats = {}
    totals = {key: 0 for key in range(8)}
    max_values = {key: {"label": None, "value": 0} for key in range(8)}

    for val in set(y1):
        inds = [i for i in range(len(y1)) if y1[i] == val]
        p = preds[inds]
        y2 = y1[inds]
        counts = dict(Counter(p))

        # Store the counts in the cluster_stats dictionary
        cluster_stats[val] = counts

        # Calculate totals and update max_values for each key (cluster)
        for key, value in counts.items():
            totals[key] += value
            if value > max_values[key]["value"]:
                max_values[key] = {"label": key, "value": value}

        print("Cluster:", val)
        print("Counts:", counts)
        # Max value and label
        print("Max value:", max(counts.values()))
        max_label = max(counts, key=counts.get)
        print("Max label:", max_label, mapping[max_label])
        # Total count
        print("Total count:", sum(counts.values()))
        print("----------------")

    # After processing all clusters, print the aggregated statistics
    print("Total Counts for Each Cluster:", totals)
    print("Maximum Value and Label for Each Cluster:", max_values)

    return cluster_stats, totals, max_values


# Usage of the function (assuming preds and y1 are defined)
# cluster_stats, totals, max_values = view_data(y1, preds)


# %%
y1 = np.argmax(y_train, axis=1)
preds = kmeans.predict(preds_train)

decoded_labels = label_encoder.inverse_transform(preds)


mapping = {}
reverse_mapping = {}

for d, p in zip(decoded_labels, preds):
    mapping[d] = p
    reverse_mapping[p] = d
    if len(mapping) == 8:
        break

cluster_stats, totals, max_values = view_data(y1, preds, reverse_mapping)
mapping

# %%
cluster_stats
xxx = {}
for i in range(8):
    # get max value and label for for cluster_stats[i]
    max_value = max(cluster_stats[i].values())
    max_label = [k for k, v in cluster_stats[i].items() if v == max_value][0]
    xxx[i] = (
        reverse_mapping[max_label],
        max_value,
    )

xxx


# %%
# get the max value and label for each cluster
import numpy as np


def get_max_value_label(cluster_labels, cluster_values):
    max_value = np.max(cluster_values)
    max_label = cluster_labels[np.argmax(cluster_values)]
    return max_value, max_label, cluster_values, cluster_labels


max_value, max_label, cluster_values, cluster_labels = get_max_value_label(
    cluster_stats, totals
)

# %%
max_label


# %%


# %%


# %%

# %%
# dense net
mapping = {0: 4, 1: 7, 2: 0, 3: 2, 4: 3, 5: 5, 6: 9, 7: 8, 8: 6, 9: 7}
correct = 0
for val in set(y1):
    inds = [i for i in range(len(y1)) if y1[i] == val]  # all indices
    p = preds[inds]
    y2 = y1[inds]
    counts = dict(Counter(p))
    correct += counts[mapping[y2[0]]]
print("accuracy: ", correct * 100 / len(y1))

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
