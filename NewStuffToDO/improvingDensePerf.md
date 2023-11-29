The code you've provided appears well-structured for clustering image data using a KMeans approach after feature extraction with a pre-trained DenseNet201 model. However, if the performance in clustering 8 unique diagnoses into separate groups is not satisfactory, there could be several factors to consider:

1. **Quality and Preprocessing of Data**: 
   - Ensure that the images are preprocessed correctly. For medical images like dermatological images, specific preprocessing steps (like contrast enhancement, noise reduction) might be necessary.
   - Check if the images are properly aligned and cropped to focus on the relevant areas.

2. **Feature Extraction Model Suitability**:
   - DenseNet201 is a powerful model for feature extraction, but it's pre-trained on ImageNet, which is significantly different from medical images. Consider fine-tuning the model on a dataset similar to yours or using a model pre-trained on medical images.

3. **KMeans Limitations**:
   - KMeans clustering assumes that clusters are spherical and equally sized, which might not be the case for your data. This can lead to poor performance if the actual clusters have different shapes or densities.
   - The number of clusters (`n_clusters=8`) is set based on the number of diagnoses. However, the actual data distribution might not align well with this assumption. It's worth experimenting with different numbers of clusters.

4. **Feature Space Dimensionality**:
   - The feature space obtained from DenseNet201 might be too high-dimensional for effective KMeans clustering. Consider using dimensionality reduction techniques like PCA (Principal Component Analysis) before clustering.

5. **Hyperparameter Tuning**:
   - Parameters of KMeans (like `max_iter`, `tol`, etc.) and the neural network can be fine-tuned for better performance.
   - The random_state parameter in KMeans can also affect results; trying different seeds might give a better clustering.

6. **Alternative Clustering Methods**:
   - If KMeans is not effective, consider other clustering algorithms like DBSCAN, Agglomerative Hierarchical Clustering, or GMM (Gaussian Mixture Models) which do not assume spherical clusters.
   - For high-dimensional data, algorithms like t-SNE or UMAP for dimensionality reduction followed by clustering can sometimes yield better results.

7. **Evaluation Metrics**:
   - Use appropriate evaluation metrics to assess the clustering performance. If you have labeled data, metrics like Silhouette Score, Calinski-Harabasz Index, or Davies-Bouldin Index can be useful.

8. **Training Data Balance**:
   - Ensure that the training data is balanced among different classes. Imbalanced data can lead to biased feature extraction and clustering.

9. **Incorporating Domain Knowledge**:
   - Sometimes, incorporating domain-specific features or using domain knowledge in preprocessing can improve performance significantly in tasks like medical image analysis.

By addressing these aspects, you might improve the clustering performance of your model. It often requires a bit of experimentation to find the right combination of techniques and parameters for a specific dataset and task.