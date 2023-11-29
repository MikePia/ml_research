import pickle
import os
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


# Load the processed datasets
raw_train = "/uw/ml_unsuper/ISIC_proc/data/raw_datasets/train_data.pkl"
raw_test = "/uw/ml_unsuper/ISIC_proc/data/raw_datasets/test_data.pkl"

with open(raw_train, "rb") as f:
    x_train, metadata_train = pickle.load(f)

with open(raw_test, "rb") as f:
    x_test, metadata_test = pickle.load(f)


# Step 1: Apply KMeans Clustering directly on flattened image data
# Assuming x_train and x_test are lists of flattened image arrays
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(x_train)


# %%
# Predict clusters for the test set
clusters = kmeans.predict(x_test)


# %%
# Extracting diagnosis labels from metadata
diagnosis_labels = [metadata["diagnosis"] for metadata in metadata_test]


# %%
# Creating a DataFrame for analysis
cluster_vs_diagnosis = pd.DataFrame(
    {"Cluster": clusters, "Diagnosis": diagnosis_labels}
)


# %%
# Analyzing the correspondence
correspondence_analysis = (
    cluster_vs_diagnosis.groupby(["Cluster", "Diagnosis"]).size().unstack(fill_value=0)
)
print(correspondence_analysis)

# %%


# %%
# Find max values from each row
max_values = correspondence_analysis.max(axis=1)

# Check if max values are unique
unique_max_values = len(max_values) == len(set(max_values))

# Output results
print("Max values from each row:\n", max_values)
print("Each row has a unique max value:", unique_max_values)


# %%
# Find max values and corresponding diagnosis for each row
max_values_with_labels = correspondence_analysis.idxmax(axis=1)

# Display the results
print("Max diagnosis in each cluster:")
print(max_values_with_labels)


# %%
# Find max values and their counts
max_counts = correspondence_analysis.max(axis=1)
max_diagnoses = correspondence_analysis.idxmax(axis=1)

# Combine the information
max_info = pd.DataFrame({"Max Diagnosis": max_diagnoses, "Count": max_counts})

print(max_info)


# %%


# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(max_info.index, max_info["Count"])

# Add labels for each bar
for idx, label in enumerate(max_info["Max Diagnosis"]):
    plt.text(idx, max_info["Count"][idx], str(label), ha="center", va="bottom")

plt.xlabel("Cluster")
plt.ylabel("Max Diagnosis Count")
plt.title("Maximum Diagnosis Count in Each Cluster with Corresponding Diagnosis")
plt.show()


# %%
# Display the count of each diagnosis in each cluster
for cluster_id in correspondence_analysis.index:
    print(f"Cluster {cluster_id}:")
    print(correspondence_analysis.loc[cluster_id].sort_values(ascending=False))
    print()


p = "../data/images/"
image_paths = [p + x.isic_id + ".JPG" for x in metadata_test]

# Assuming 'image_paths' is a list of file paths corresponding to your images
# and 'clusters' and 'diagnosis_labels' are as previously defined
image_data = pd.DataFrame(
    {"Path": image_paths, "Cluster": clusters, "Diagnosis": diagnosis_labels}
)


# %%
cluster_index = 2
max_diagnosis = correspondence_analysis.loc[cluster_index].idxmax()
max_diagnosis


# %%


def display_images_from_cluster(cluster_index, num_images=5):
    # Get the diagnosis with the max value in the specified cluster
    max_diagnosis = correspondence_analysis.loc[cluster_index].idxmax()

    # Filter images from the specified cluster with the max diagnosis
    filtered_images = [
        img_path
        for img_path, metadata in zip(image_paths, metadata_test)
        if metadata["Cluster"] == cluster_index
        and metadata["Diagnosis"] == max_diagnosis
    ]

    # Randomly select a few images to display
    sample_images = np.random.choice(
        filtered_images, size=min(num_images, len(filtered_images)), replace=False
    )

    # Display the images
    fig, axs = plt.subplots(1, len(sample_images), figsize=(20, 5))
    for i, img_path in enumerate(sample_images):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert color from BGR to RGB

        axs[i].imshow(img)
        axs[i].set_title(f"Cluster {cluster_index} - {max_diagnosis}")
        axs[i].axis("off")

    plt.show()


# Example usage
display_images_from_cluster(
    2
)  # Replace '2' with the cluster index you want to visualize


max_values = correspondence_analysis.max(axis=1)

# Create a bar chart

plt.bar(correspondence_analysis.index, max_values)
plt.xlabel("Cluster")
plt.ylabel("Projected Likelihood of Diagnosis")
plt.title("Correspondence Analysis")
plt.show()
