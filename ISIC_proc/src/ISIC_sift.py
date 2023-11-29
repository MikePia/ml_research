import cv2
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pickle
import zipfile


def load_and_process_image(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Initialize SIFT
    sift = cv2.SIFT_create()

    # Detect and compute SIFT features
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        descriptors = np.array([])

    return descriptors


def load_data(image_folder, metadata_file):
    # Read metadata CSV file
    metadata = pd.read_csv(metadata_file)

    # Initialize lists for storing image features and metadata
    image_features = []
    image_metadata = []

    for i, row in metadata.iterrows():
        if not i % 100:
            print(f"Processing image {i} / {len(metadata)}")
        file_id = row["isic_id"]
        image_path = os.path.join(image_folder, file_id + ".JPG")

        if os.path.exists(image_path):
            # Process each image and store the descriptors
            descriptors = load_and_process_image(image_path)
            image_features.append(descriptors)
            image_metadata.append(row)  # Store the entire row of metadata

    # Split the dataset into training and testing sets
    x_train, x_test, metadata_train, metadata_test = train_test_split(
        image_features, image_metadata, test_size=0.2, random_state=42
    )

    return (x_train, metadata_train), (x_test, metadata_test)


def save_datasets(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def prepare_dataset(archive_file, metadata_path, force=False):
    # Path to your images and metadata
    path = os.path.split(os.path.realpath(__file__))[0]
    IMG_DIR = os.path.join(path, "../data/images")
    DATA_DIR = os.path.join(path, "../data/sift_datasets")
    image_folder = IMG_DIR
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

    metadata_file = metadata_path
    sift_datasets_folder = DATA_DIR

    zip_file = zipfile.ZipFile(archive_file)

    zip_file.extractall(image_folder)
    zip_file.close()
    print("Dataset extracted successfully.")
    if os.path.exists(os.path.join(image_folder, "attribution.txt")):
        with open(os.path.join(image_folder, "attribution.txt"), "r") as f:
            print("Attributes:")
            print(f.read().strip())

    # copy the metadata file to DATA_DIR
    os.system(f"cp {metadata_file} {sift_datasets_folder}")

    print("Metadata file copied successfully.")

    return image_folder, metadata_file, sift_datasets_folder


def load_data_and_metadata(image_folder, metadata_file):
    # Load data and metadata
    (x_train, metadata_train), (x_test, metadata_test) = load_data(
        image_folder, metadata_file
    )
    return x_train, x_test, metadata_train, metadata_test


def save(x_train, x_test, metadata_train, metadata_test, sift_datasets_folder):
    # Save the datasets with metadata
    save_datasets(
        (x_train, metadata_train), os.path.join(sift_datasets_folder, "train_data.pkl")
    )
    save_datasets(
        (x_test, metadata_test), os.path.join(sift_datasets_folder, "test_data.pkl")
    )
    print("Datasets saved successfully.")


if __name__ == "__main__":
    # Path to your images and metadata, You must download the data and
    # manually set the path here
    meta = "/uw/ml_unsuper/ISIC_proc/data/ham10000_metadata_2023-11-27.csv"
    archive_file = "/uw/ml_unsuper/ISIC_proc/data/ISIC-images.zip"

    image_folder, metadata_file, sift_datasets_folder = prepare_dataset(
        archive_file, meta
    )

    x_train, x_test, metadata_train, metadata_test = load_data_and_metadata(
        image_folder, metadata_file
    )
    save(x_train, x_test, metadata_train, metadata_test, sift_datasets_folder)
    print("Datasets saved successfully.")
    print("Done.")
