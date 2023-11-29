from collections import Counter
import tensorflow as tf
import tensorflow.keras as K
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


# %%
# Load and preprocess ISIC data
def load_data(image_directory, metadata_path):
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


def get_percentage_of_classes(Y, hot_encoded=False):
    if hot_encoded:
        Y = np.argmax(Y, axis=1)
    counts = Counter(Y)
    total = sum(counts.values())
    percents = {key: value / total * 100 for key, value in counts.items()}
    return percents


def preprocess_data(X, Y, num_classes=8):
    X = K.applications.densenet.preprocess_input(X)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(Y)
    Y = K.utils.to_categorical(encoded_labels, num_classes=num_classes)

    return X, Y, label_encoder


def get_DenseNet(num_classes=8):
    base_model = tf.keras.applications.DenseNet201(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    predictions = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    return model

    base_model = K.applications.DenseNet201(
        include_top=False, weights="imagenet", input_tensor=input_tensor, pooling="max"
    )

    for layer in base_model.layers:
        layer.trainable = False

    # Adding custom layers
    x = base_model.output
    x = K.layers.Flatten(name="feats")(x)
    # Add your custom layers here, e.g., Dense, Dropout, etc.
    # For example:
    # x = K.layers.Dense(1024, activation='relu')(x)
    # predictions = K.layers.Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = K.models.Model(inputs=base_model.input, outputs=x)

    print(model.summary())


def get_fine_tuned_DenseNet(
    input_tensor,
    num_classes=8,
):
    print("Getting finetuned DenseNet")
    UNFREEZE_LAYER = -30

    # Step 1 load model
    base_model = tf.keras.applications.DenseNet201(
        include_top=False, weights="imagenet", input_tensor=input_tensor
    )

    # Step 2 add Custom layers
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)

    predictions = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    # Step 3 Compile Model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Step 4 Optional Initail Training
    # Step 5 Freeze Layers and set trainable
    unfreeze_layer = UNFREEZE_LAYER
    for layer in model.layers[:unfreeze_layer]:
        layer.trainable = False
    for layer in model.layers[unfreeze_layer:]:
        layer.trainable = True

    # Step 6 Compile Model again
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Step 7, Fine tuning done at the return
    return model


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


def view_results(y_data, pred_data, kmeans, label_encoder):
    y1 = np.argmax(y_data, axis=1)
    preds = kmeans.predict(pred_data)

    decoded_labels = label_encoder.inverse_transform(preds)

    mapping = {}
    reverse_mapping = {}

    for d, p in zip(decoded_labels, preds):
        mapping[d] = p
        reverse_mapping[p] = d
        if len(mapping) == 8:
            break

    cluster_stats, totals, max_values = view_data(y1, preds, reverse_mapping)


def Main(metadata_path):
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    image_directory = os.path.join(data_dir, "images")

    assert os.path.exists(data_dir), "Data directory not found"
    assert os.path.exists(image_directory), "Image directory not found"
    assert os.path.exists(metadata_path), "Metadata file not found"

    X, Y = load_data(image_directory, metadata_path)
    original_percentages = get_percentage_of_classes(Y)
    X, Y, label_encoder = preprocess_data(X, Y)
    encoded_percentages = get_percentage_of_classes(Y, hot_encoded=True)

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    y_train_percentages = get_percentage_of_classes(y_train, hot_encoded=True)
    y_test_percentages = get_percentage_of_classes(y_test, hot_encoded=True)

    input_tensor = K.Input(shape=(224, 224, 3))
    model = get_fine_tuned_DenseNet(input_tensor, num_classes=8)

    print("Training pretrained model")
    model.fit(
        x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test)
    )

    print(model.summary())

    preds_train = model.predict(x_train)
    preds_test = model.predict(x_test)

    kmeans = KMeans(
        n_clusters=8, random_state=42, max_iter=1000, algorithm="elkan", tol=0.000001
    ).fit(preds_test)

    view_results(y_test, preds_test, kmeans, label_encoder)

    kmeans = KMeans(
        n_clusters=8, random_state=42, max_iter=1000, algorithm="elkan", tol=0.000001
    ).fit(preds_train)
    view_results(y_train, preds_train, kmeans, label_encoder)


if __name__ == "__main__":
    metadata_fn = "/uw/ml_unsuper/ISIC_proc/data/ham10000_metadata_2023-11-27.csv"
    Main(metadata_fn)
