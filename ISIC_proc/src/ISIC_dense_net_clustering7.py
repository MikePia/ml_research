# %%
from collections import Counter
import numpy as np
import os
import pandas as pd

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %%
# Can save Densenet models fine tuned with different labels
# melanocytic (2 classes)
THE_MODEL = "fine_tuned_model_melanocytic.keras"

# diagnosis (8 classes)
# THE_MODEL = "fine_tuned_model.keras"

# benign_malignant (2 classes)
# THE_MODEL = "fine_tuned_model_bm.keras"

label = "melanocytic"
metadata_path = "/uw/ml_unsuper/ISIC_proc/data/ham10000_metadata_2023-11-27.csv"
# The minimum number of images with a label required to include a label in the dataset
minimum_count = 10
augment_training_data = False
augment_test_data = False
force_load_model = False
force_load_preds = True
EPOCHS = 10


# %%
data_gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.9, 1.1],
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)


# %%
def image_generator(
    image_directory,
    filtered_metadata,
    label_encoder,
    additional_augmentations,
    batch_size=32,
    num_classes=8,
    augment_data=False,
    label="diagnosis",
    filter=True,
):
    num_samples = len(filtered_metadata)
    augmented_images_counter = {label: 0 for label in additional_augmentations.keys()}

    while True:  # Generator loops indefinitely
        for offset in range(0, num_samples, batch_size):
            batch_samples = filtered_metadata.iloc[offset : offset + batch_size]

            images = []
            labels = []
            for ix, row in batch_samples.iterrows():
                if filter is True and pd.isna(row[label]):
                    continue

                img_path = os.path.join(image_directory, row["isic_id"] + ".JPG")
                try:
                    img = image.load_img(img_path, target_size=(224, 224))
                    img = image.img_to_array(img)
                    img = K.applications.densenet.preprocess_input(img)

                    # Apply LabelEncoder to get the encoded label
                    encoded_label = label_encoder.transform([row[label]])[0]

                    # Check if augmentation is needed for this label
                    if augment_data and augmented_images_counter[row[label]] < additional_augmentations[row[label]]:
                        augmented_img = img.reshape((1,) + img.shape)  # Reshape for data_gen
                        augmentations_to_do = additional_augmentations[row[label]] - augmented_images_counter[row[label]]
                        for _ in range(augmentations_to_do):
                            augmented_image = data_gen.flow(augmented_img, batch_size=1).next()[0]
                            images.append(augmented_image)
                            labels.append(encoded_label)
                            augmented_images_counter[row[label]] += 1
                            if augmented_images_counter[row[label]] >= additional_augmentations[row[label]]:
                                break

                    # Append the original image and its label
                    images.append(img)
                    labels.append(encoded_label)
                except Exception as e:
                    print(f"Error processing file {img_path}: {e}")
                    continue

            if images:
                X_batch = np.array(images)
                Y_batch = K.utils.to_categorical(labels, num_classes=num_classes)
                yield X_batch, Y_batch



# %%
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


# %%
def get_densenet_num_classes():
    assert model, "Model not found"
    output_layer = model.layers[-1]
    output_shape = output_layer.output_shape
    num_classes = output_shape[-1]  # Assuming the output layer is a Dense layer
    print(f"Number of classes in the output layer: {num_classes}")
    return num_classes

# %%
def get_metadata(
    image_directory,
    metadata_path,
    label_column,
    minimum_count=10,
    filter_images=True,
    min_count_threshold=0.25,
):
    # Load and preprocess metadata
    metadata = pd.read_csv(metadata_path)
    # if not filter_images:
    # diagnosis_counts = metadata[label_column].value_counts(dropna=False)
    # else:
    diagnosis_counts = metadata[label_column].value_counts()

    # Filter metadata
    if filter_images:
        filtered_metadata = metadata[
            metadata[label_column].isin(
                diagnosis_counts[diagnosis_counts >= minimum_count].index
            )
        ]
    else:
        filtered_metadata = metadata

    # Fit the LabelEncoder on the filtered labels
    label_encoder = LabelEncoder()
    if not filter_images:
        label_encoder.fit(filtered_metadata[label_column])
    else:
        label_encoder.fit(filtered_metadata[label_column].dropna())
    num_classes = len(label_encoder.classes_)

    return filtered_metadata, label_encoder, num_classes


# %%


# %%
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

        # sort counts by key
        counts = dict(sorted(counts.items()))

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
        percentage = (max(counts.values()) / sum(counts.values())) * 100
        percentage = round(percentage, 2)
        print(f"{percentage}% exclusive")
        # Total count
        print("Total count:", sum(counts.values()))
        print("----------------")

    # After processing all clusters, print the aggregated statistics
    print("Total Counts for Each Cluster:", totals)
    print("Maximum Value and Label for Each Cluster:", max_values)

    return cluster_stats, totals, max_values


# %%
def is_standard_python() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return False   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        # Probably a standard Python interpreter
        return True

# %%
def view_results(y_data, pred_data, kmeans, label_encoder):
    labels_true = np.argmax(y_data, axis=1)
    preds = kmeans.predict(pred_data)
    num_classes = len(y_data[0])

    decoded_labels = label_encoder.inverse_transform(preds)
    print(set(decoded_labels))

    mapping = {}
    reverse_mapping = {}

    for d, p in zip(decoded_labels, preds):
        mapping[d] = p
        reverse_mapping[p] = d



    cluster_stats, totals, max_values = view_data(labels_true, preds, reverse_mapping)

    return cluster_stats, totals, max_values


# %%
if __name__ == "__main__" and is_standard_python():
    d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(d, "data")
else:
    data_dir = "../data"

image_directory = os.path.join(data_dir, "images")

num_classes_path = os.path.join(data_dir, "num_classes.npy")
save_preds_train = os.path.join(data_dir, "preds_train.npy")
save_preds_test = os.path.join(data_dir, "preds_test.npy")

save_true_Ytrain = os.path.join(data_dir, "true_Ytrain.npy")
save_true_Ytest = os.path.join(data_dir, "true_Ytest.npy")

MODEL_NAME = os.path.join(data_dir, THE_MODEL)

local_save_paths = [num_classes_path, save_preds_train, save_preds_test,
                    save_true_Ytrain, save_true_Ytest, MODEL_NAME]


assert os.path.exists(data_dir), "Data directory not found"
assert os.path.exists(image_directory), "Image directory not found"
assert os.path.exists(metadata_path), "Metadata file not found"


# %%
def calculate_augmentations(filtered_metadata, label_column, min_count_threshold=0.25, empty=False):
    """
    Calculate the number of additional images needed for each label in a dataset.

    Parameters:
        filtered_metadata (pandas.DataFrame): The filtered metadata containing the image labels. The applied
            filters may include a minimum number of labels and exclusion of NaN labels.
        label_column (str): The name of the column in the metadata that contains the image labels.
        min_count_threshold (float, optional): The minimum count threshold as a fraction of the maximum count.
            Default is 0.25.

    Returns:
        dict: A dictionary where the keys are the labels and the values are the number of additional images
            needed for each label.
    """
    if empty:
        return {label: 0 for label in filtered_metadata[label_column].unique()}
    label_counts = filtered_metadata[label_column].value_counts()
    max_count = max(label_counts)
    minimum_count = round(max_count * min_count_threshold)

    additional_images_needed = {
        label: max(0, minimum_count - count) for label, count in label_counts.items()
    }
    return additional_images_needed


# %%

filtered_metadata, label_encoder, num_classes = get_metadata(
    image_directory, metadata_path, label, minimum_count=minimum_count
)


# %%
def predict_generator(model, metadata, x_generator, batch_size=32):
    num_samples = len(metadata)
    predict_steps = np.ceil(num_samples / batch_size)

    predictions = []
    true_labels = []
    for _ in range(int(predict_steps)):
        print(f"Predicting batch {_:+3}/{predict_steps:+3}...", end="\r")
        X_batch, Y_batch = next(x_generator)  # Get both features and labels from the same generator

        batch_predictions = model.predict(X_batch)
        predictions.extend(batch_predictions)
        true_labels.extend(Y_batch)  # Assuming Y_batch is not one-hot encoded; if it is, convert it back

    return np.array(predictions), np.array(true_labels)


# %%
# Assuming filtered_metadata is already prepared
train_metadata, test_metadata = train_test_split(
    filtered_metadata, test_size=0.4, random_state=42
)
# Calculate additional augmentations needed
# label_column = "melanocytic"
train_augments = calculate_augmentations(train_metadata, label)
test_augments = calculate_augmentations(test_metadata, label)

train_generator = image_generator(
    image_directory,
    train_metadata,
    label_encoder,
    train_augments,
    batch_size=32,
    num_classes=num_classes,
    label=label,
    augment_data=False
)
test_generator = image_generator(
    image_directory,
    test_metadata,
    label_encoder,
    test_augments,
    batch_size=32,
    num_classes=num_classes,
    label=label,
    augment_data=False
)


# %%
if not os.path.exists(MODEL_NAME) or force_load_model is True:
    print("Fine tuning the model. This will take a while. Please wait.")
    print("Loading training data for DenseNet")

    print("Fine-tuning DenseNet ...")
    input_tensor = K.Input(shape=(224, 224, 3))
    model = get_fine_tuned_DenseNet(input_tensor, num_classes=num_classes)

    print("Training pre-trained model")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=len(train_metadata) // 32,
    )
    model.save(MODEL_NAME)  # Use model.save instead of np.save for models

else:
    print("Loading pre-trained model from local storage")
    model = load_model(MODEL_NAME)

model.summary()

# %%



# %%

if not os.path.isfile(save_preds_train) or force_load_preds == True:


    # preds_train, true_Ytrain = predict_generator(model, train_metadata, train_generator)
    preds_test, true_Ytest = predict_generator(model, test_metadata, test_generator)


    # np.save(save_preds_train, preds_train)
    np.save(save_preds_test, preds_test)

    # np.save(save_true_Ytrain, true_Ytrain)
    np.save(save_true_Ytest, true_Ytest)
else:
    print("Loading precomputed predictions")

    preds_train = np.load(save_preds_train)
    preds_test = np.load(save_preds_test)

    true_Ytrain = np.load(save_true_Ytrain)
    true_Ytest = np.load(save_true_Ytest)


# %%
# kmeans_train = KMeans(
#     n_clusters=num_classes,
#     random_state=42,
#     max_iter=1000,
#     algorithm="elkan",
#     tol=0.000001,
# ).fit(preds_train)

kmeans_test = KMeans(
    n_clusters=num_classes,
    random_state=22,
    max_iter=2000,
    algorithm="elkan",
    tol=0.000001,
).fit(preds_test)



# %%
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np


def evaluate_clusters(labels_true, labels_pred):
    """
    Evaluates the clustering performance based on true labels.

    Args:
    labels_true: array-like, true class labels
    labels_pred: array-like, predicted cluster labels

    Returns:
    A dictionary containing ARI, NMI, and purity scores.
    """

    # Adjusted Rand Index (ARI)
    ari = adjusted_rand_score(labels_true, labels_pred)

    # Normalized Mutual Information (NMI)
    nmi = normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic')

    # Purity
    # Create a contingency matrix
    contingency_matrix = pd.crosstab(labels_true, labels_pred)

    # Normalize the contingency matrix
    # contingency_matrix = contingency_matrix / np.sum(contingency_matrix, axis=1)

    # Calculate purity
    purity = np.max(contingency_matrix, axis=1) / np.sum(contingency_matrix, axis=1)


    return {"ARI": ari, "NMI": nmi, "Purity": purity}

# %%
# Assuming kmeans is already fitted to preds_X
labels_pred = kmeans_test.predict(preds_test)

# Convert one-hot encoded labels to single labels if necessary
if true_Ytest.shape[1] > 1:  # Checking if true_Y is one-hot encoded
    labels_true = np.argmax(true_Ytest, axis=1)
else:
    labels_true = true_Ytest

print(label)
# Evaluate clusters
evaluation_results = evaluate_clusters(labels_true, labels_pred)
print("Evaluation Results:", evaluation_results)

# %%
view_results(true_Ytest, preds_test, kmeans_test, label_encoder)
[7, 3, 5, 4, 0, 1, 5, 6]



# %%
def clear_local_saved_data():
    for r in local_save_paths:
        if os.path.isfile(r):
            try:
                os.remove(r)
                print(f"Removed {r}")
            except OSError:
                print(f"Failed to remove {r}")
        else:
            print(f"{r} does not exist.")
    return
    


# ask user if they want to remove the data
remove_data = input("Do you want to remove the data? (y/n): ")
if remove_data.lower() == "y":
    clear_local_saved_data()


# %%
test_metadata.diagnosis.value_counts()

# %%



