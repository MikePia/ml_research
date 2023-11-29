# %%
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %%
def preprocess_data(X, Y):
    X = K.applications.densenet.preprocess_input(X)
    Y = K.utils.to_categorical(Y)
    return X, Y

# %%
(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

# %%
initializer = K.initializers.he_normal()
input_tensor = K.Input(shape=(32, 32, 3))

# resized_images = K.layers.Lambda(lambda image: tf.image.resize(image, (224, 224)))(input_tensor)
model = K.applications.DenseNet201(include_top=False,
                                   weights='imagenet',
                                   input_tensor=resized_images,
                                   input_shape=(224, 224, 3),
                                   pooling='max',
                                   classes=1000)

# %%
for layer in model.layers:
    layer.trainable = False
output = model.layers[-1].output
flatten = K.layers.Flatten(name='feats')
output = flatten(output)
model = K.models.Model(inputs=input_tensor, outputs=output)
model.summary()

# %%
y_test.shape

# %%
preds=model.predict(x_train)
preds.shape

# %%
preds_test=model.predict(x_test)
preds_test.shape

# %%


# %%
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=42, max_iter=1000, algorithm='elkan', tol=0.000001).fit(preds_test)

preds=kmeans.predict(preds_test) # use features extracted from denseNet
preds.shape

# %%
y1=np.argmax(y_test,axis=1)
y1.shape

# %%
# assign predicted clusters to actual classes to get the accuracy
from collections import Counter

correct=0
mapping={}

for val in set(y1):
  inds=[i for i in range(len(y1)) if y1[i]==val] # all indices for a particular class
  p=preds[inds] # predictions made for that class
  y2=y1[inds]
  counts=dict(Counter(p))

  print("y2: ", y2[:2])
  print("counts: ", counts)
  print("----------------")

# %%
#dense net
mapping={
    0:4,
    1:7,
    2:0,
    3:2,
    4:3,
    5:5,
    6:9,
    7:8,
    8:6,
    9:7
}
correct=0
for val in set(y1):
  inds=[i for i in range(len(y1)) if y1[i]==val] # all indices
  p=preds[inds]
  y2=y1[inds]
  counts=dict(Counter(p))
  correct+=counts[mapping[y2[0]]]
print("accuracy: ", correct*100/len(y1))

# %%



