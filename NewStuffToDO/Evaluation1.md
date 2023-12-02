### Data source images from ISIC challenge 2020 tested against the diagnosis provided in the metadata. The test was how they would cluster using the KMeans algorithm.
#### Notes about the data:
* Removed images that lacked a diagnosis.
* Removed diagnoses that had fewer than 10 examples (out of about 6000 images)
* Augmented the numbers using standard augmentation techniques so that each remaing category had at least 1/4 the number of the max catagory
* Use a pretrained model, DenseNet, finetuned with ISIC HAM100000 images. 
* The categories in in the HAM10000 were not the same as the categories in Challenge 2020 dataset, they overlapped only 'nevus' and 'melanoma'
HAM 10000 diagnoses [  
    actinic keratosis,
    basal cell carcinoma,
    dermatofibroma,
    nevus,
    melanoma,
    pigmented benign keratosis,
    squamous cell carcinoma,
    vascular lesion


Challenge 2020 diagnoses [ 
    lentigo NOS
    lichenoid keratosis' 
    melanoma' 
    nevus
    other' 
    seborrheic keratosis]
The fact that they were different is significant in that the model differentiated them into catagories inspite of not have any apecific examples of many of the diagnoses. 

The maximum count  for each cluster  is 93-100%  exclusive. Five of the 6 categories exceede 98% exclusive.

(Personal note, this is going out on a limb a bit as I don't know the about diagnoses and it's very possible that categories are closely related.)




Cluster: 0
Counts: {1: 269, 0: 2}
Max value: 269
Max label: 1 lichenoid keratosis
99% exclusive

----------------
Cluster: 1
Counts: {2: 251, 0: 1}
Max value: 251
Max label: 2 melanoma
99.6% exclusive
Total count: 252

----------------
Cluster: 2
Counts: {5: 202, 3: 4, 1: 2, 0: 42}
Max value: 202
Max label: 5 seborrheic keratosis
89% exclusive
Total count: 250

----------------
Cluster: 3
Counts: {0: 1033, 5: 20, 3: 3, 2: 1}
Max value: 1033
Max label: 0 lentigo NOS
98% exclusive
Total count: 1057

----------------
Cluster: 4
Counts: {4: 253}
Max value: 253
Max label: 4 other
100% exclusive
Total count: 253

----------------
Cluster: 5
Counts: {3: 240, 2: 3, 5: 5, 0: 7}
Max value: 240
Max label: 3 nevus
94% exclusive
Total count: 255

----------------
Total Counts for Each Cluster: {0: 1085, 1: 271, 2: 255, 3: 247, 4: 253, 5: 227, 6: 0, 7: 0}
Maximum Value and Label for Each Cluster: {0: {'label': 0, 'value': 1033}, 1: {'label': 1, 'value': 269}, 2: {'label': 2, 'value': 251}, 3: {'label': 3, 'value': 240}, 4: {'label': 4, 'value': 253}, 5: {'label': 5, 'value': 202}, 6: {'label': None, 'value': 0}, 7: {'label': None, 'value': 0}}
/uw/.venvs/ml_research/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:1416: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  super()._check_params_vs_input(X, default_n_init=10)







Cluster: 0
Counts: {4: 1297, 2: 2}
Max value: 1297
Max label: 4 other
99.85% exclusive
Total count: 1299

----------------
Cluster: 1
Counts: {3: 1298, 2: 1}
Max value: 1298
Max label: 3 nevus
99.92% exclusive
Total count: 1299

----------------
Cluster: 2
Counts: {0: 1208, 2: 85, 5: 4, 4: 2}
Max value: 1208
Max label: 0 lentigo NOS
92.99% exclusive
Total count: 1299

----------------
Cluster: 3
Counts: {2: 5171, 0: 20, 5: 2, 3: 2}
Max value: 5171
Max label: 2 melanoma
99.54% exclusive
Total count: 5195

----------------
Cluster: 4
Counts: {1: 1299}
Max value: 1299
Max label: 1 lichenoid keratosis
100.0% exclusive
Total count: 1299


----------------
Cluster: 5
Counts: {5: 1283, 2: 8, 0: 5, 3: 3}
Max value: 1283
Max label: 5 seborrheic keratosis
98.77% exclusive
Total count: 1299

----------------
Total Counts for Each Cluster: {0: 1233, 1: 1299, 2: 5267, 3: 1303, 4: 1299, 5: 1289, 6: 0, 7: 0}
Maximum Value and Label for Each Cluster: {0: {'label': 0, 'value': 1208}, 1: {'label': 1, 'value': 1299}, 2: {'label': 2, 'value': 5171}, 3: {'label': 3, 'value': 1298}, 4: {'label': 4, 'value': 1297}, 5: {'label': 5, 'value': 1283}, 6: {'label': None, 'value': 0}, 7: {'label': None, 'value': 0}}
/uw/.venvs/ml_research/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:1416: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  super()._check_params_vs_input(X, default_n_init=10)



________
# Tried traing densenet using all 2020 challenge images/ 27K of 33K are unlabeled
### results are not great
Cluster: 0
Counts: {3: 9, 6: 10, 1: 109, 7: 10, 0: 4, 4: 2, 5: 4, 2: 1}
Max value: 109
Max label: 1 basal cell carcinoma
73.15% exclusive
Total count: 149

----------------
Cluster: 1
Counts: {6: 60, 1: 341, 7: 52, 0: 55, 2: 22, 5: 38, 3: 32, 4: 22}
Max value: 341
Max label: 1 basal cell carcinoma
54.82% exclusive
Total count: 622

----------------
Cluster: 2
Counts: {0: 45, 6: 7, 5: 10, 1: 77, 2: 8, 7: 4, 4: 7, 3: 2}
Max value: 77
Max label: 1 basal cell carcinoma
48.12% exclusive
Total count: 160

----------------
Cluster: 3
Counts: {2: 86, 0: 253, 1: 480, 6: 127, 5: 98, 7: 80, 3: 127, 4: 54}
Max value: 480
Max label: 1 basal cell carcinoma
36.78% exclusive
Total count: 1305

----------------
Cluster: 4
Counts: {0: 5399, 5: 345, 6: 339, 1: 1231, 4: 293, 2: 56, 7: 40, 3: 34}
Max value: 5399
Max label: 0 actinic keratosis
69.78% exclusive
Total count: 7737

----------------
Cluster: 5
Counts: {0: 217, 1: 753, 4: 62, 6: 111, 5: 74, 7: 62, 2: 30, 3: 29}
Max value: 753
Max label: 1 basal cell carcinoma
56.28% exclusive
Total count: 1338

----------------
Cluster: 6
Counts: {1: 164, 4: 5, 6: 21, 0: 8, 7: 13, 2: 5, 5: 4, 3: 9}
Max value: 164
Max label: 1 basal cell carcinoma
71.62% exclusive
Total count: 229

----------------
Cluster: 7
Counts: {0: 84, 1: 51, 5: 10, 4: 12, 6: 13, 7: 6, 2: 2, 3: 2}
Max value: 84
Max label: 0 actinic keratosis
46.67% exclusive
Total count: 180

----------------
Total Counts for Each Cluster: {0: 6065, 1: 3206, 2: 210, 3: 244, 4: 457, 5: 583, 6: 688, 7: 267}