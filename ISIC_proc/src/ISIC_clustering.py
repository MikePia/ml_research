import pickle
import os
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
import numpy as np
import pandas as pd


# Load the processed datasets
sift_train = "/uw/ml_unsuper/ISIC_proc/data/sift_datasets/train_data.pkl"
sift_test = "/uw/ml_unsuper/ISIC_proc/data/sift_datasets/test_data.pkl"

with open(sift_train, "rb") as f:
    x_train, metadata_train = pickle.load(f)

with open(sift_test, "rb") as f:
    x_test, metadata_test = pickle.load(f)



# %%
def create_histograms(feature_list, kmeans_model):
    histograms = []
    for features in feature_list:
        # Check if the features array is empty
        if features.size == 0:
            # Assign a zero histogram if there are no features
            hist = np.zeros(kmeans_model.n_clusters)
        else:
            words, _ = vq(features, kmeans_model.cluster_centers_)
            hist, _ = np.histogram(
                words, bins=range(kmeans_model.n_clusters + 1), density=True
            )
        histograms.append(hist)
    return histograms


# %% [markdown]
# 

# %%
number_of_visual_words = 50  # You can adjust this number
all_descriptors = []
total_features = len(x_train)
print("Aggregating descriptors...")

for i, feature in enumerate(x_train):
    all_descriptors.extend(feature)
    
    # Print progress every 5% (or choose a different interval if needed)
    if not i % 500:
        print(f"Progress: {i / total_features * 100:.2f}% complete")

print("Aggregation complete.")


# %%
kmeans_vw = KMeans(n_clusters=number_of_visual_words, random_state=42, verbose=1)
kmeans_vw.fit(all_descriptors)


# %%
# Step 2: Create histograms for training and testing sets
x_train_histograms = create_histograms(x_train, kmeans_vw)
x_test_histograms = create_histograms(x_test, kmeans_vw)


# %%
# Step 3: Apply KMeans Clustering on histograms
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(x_train_histograms)


# %%
# Predict clusters for the test set
clusters = kmeans.predict(x_test_histograms)


# %%
x=3
x

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
max_info = pd.DataFrame({
    "Max Diagnosis": max_diagnoses,
    "Count": max_counts
})

print(max_info)


# %%
import matplotlib.pyplot as plt

# Create a bar chart
plt.figure(figsize=(10,6))
plt.bar(max_info.index, max_info['Count'])

# Add labels for each bar
for idx, label in enumerate(max_info['Max Diagnosis']):
    plt.text(idx, max_info['Count'][idx], str(label), ha='center', va='bottom')

plt.xlabel('Cluster')
plt.ylabel('Max Diagnosis Count')
plt.title('Maximum Diagnosis Count in Each Cluster with Corresponding Diagnosis')
plt.show()


# %%
# Display the count of each diagnosis in each cluster
for cluster_id in correspondence_analysis.index:
    print(f"Cluster {cluster_id}:")
    print(correspondence_analysis.loc[cluster_id].sort_values(ascending=False))
    print()


# %%
import cv2
import matplotlib.pyplot as plt

p = "../data/images/"
image_paths = [p + x.isic_id + ".JPG" for x in metadata_test]

# Assuming 'image_paths' is a list of file paths corresponding to your images
# and 'clusters' and 'diagnosis_labels' are as previously defined
image_data = pd.DataFrame({
    'Path': image_paths,
    'Cluster': clusters,
    'Diagnosis': diagnosis_labels
})


# %%
cluster_index = 2
max_diagnosis = correspondence_analysis.loc[cluster_index].idxmax()
max_diagnosis

# %%
correspondence_analysis


# %%

    # Filter images from the specified cluster with the max diagnosis
    filtered_images = [img_path for img_path, metadata in zip(image_paths, metadata_test)
                       if metadata['Cluster'] == cluster_index and metadata['Diagnosis'] == max_diagnosis]

# %%
import cv2
import matplotlib.pyplot as plt

def display_images_from_cluster(cluster_index, num_images=5):
    # Get the diagnosis with the max value in the specified cluster
    max_diagnosis = correspondence_analysis.loc[cluster_index].idxmax()

    # Filter images from the specified cluster with the max diagnosis
    filtered_images = [img_path for img_path, metadata in zip(image_paths, metadata_test)
                       if metadata['Cluster'] == cluster_index and metadata['Diagnosis'] == max_diagnosis]

    # Randomly select a few images to display
    sample_images = np.random.choice(filtered_images, size=min(num_images, len(filtered_images)), replace=False)

    # Display the images
    fig, axs = plt.subplots(1, len(sample_images), figsize=(20, 5))
    for i, img_path in enumerate(sample_images):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert color from BGR to RGB

        axs[i].imshow(img)
        axs[i].set_title(f"Cluster {cluster_index} - {max_diagnosis}")
        axs[i].axis('off')

    plt.show()

# Example usage
display_images_from_cluster(2)  # Replace '2' with the cluster index you want to visualize


# %%


# %%
import matplotlib.pyplot as plt
# Use correspondence_analysis to get the max value in each row of the DataFrame The Y labels include the name of the diagnosis 

max_values = correspondence_analysis.max(axis=1)

# Create a bar chart

plt.bar(correspondence_analysis.index, max_values)
plt.xlabel("Cluster")
plt.ylabel("Projected Likelihood of Diagnosis")
plt.title("Correspondence Analysis")
plt.show()


# %%


# %%


# %%
os.path.exists(image_paths[0])

# %%



