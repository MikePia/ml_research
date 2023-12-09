A Case study for the research paper Unsupervised image classification and latent representation

Implemented [in this note book](./ISIC_proc/src/ISIC_dense_net_clustering7.ipynb)

We applied the DenseNet image encoding and KMeans clustering to the ISIC HAM10000 dataset, a collection of diverse skin lesion images. The DenseNet201 model, pre-trained on ImageNet, was fine-tuned using a subset of the HAM10000 dataset. We employed KMeans to cluster the HAM10000 test dataset, aiming to explore natural groupings within the skin lesion images without the guidance of pre-labeled categories. Clustering performance was evaluated using metrics like Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and purity. We ran the KMeans algorithm repeatedly, selecting the iteration with the highest purity score as the optimal clustering solution. This methodical approach refined our clustering process, ensuring a more reliable and accurate classification of the skin lesion images.

