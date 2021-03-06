# -*- coding: utf-8 -*-
"""model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Raddcwq5jkObKqUoquN0ZtSjXdbkxqT0
"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# %matplotlib inline

def get_datagen(dataset):
    return ImageDataGenerator().flow_from_directory(
              dataset,
              target_size=(71,71),
              color_mode='rgb',
              shuffle = True,
              class_mode='categorical',
              batch_size=32)

from skimage import io, transform

X_test_gen= get_datagen('/content/drive/MyDrive/New/data')

X_test = np.zeros((len(X_test_gen.filepaths), 71, 71, 3))
Y_test = np.zeros((len(X_test_gen.filepaths), 4))
for i in range(0,len(X_test_gen.filepaths)):
  x = io.imread(X_test_gen.filepaths[i], as_gray=True)
  X_test[i,:] = transform.resize(x, (71,71,3))
  Y_test[i,X_test_gen.classes[i]] = 1

model = load_model('/content/drive/MyDrive/Project/model_r.h5') 

y_pred_probs = model.predict(X_test)
y_pred = y_pred_probs.argmax(axis=1)
y_true = Y_test.argmax(axis=1)

incorrect = np.count_nonzero(y_pred-y_true)
print("Accuracy on test images: {:.2%}".format(1.0 - incorrect/len(y_true)))

from sklearn.metrics import confusion_matrix
from seaborn import heatmap

emotions = {0:'goddess',1:'tree', 2:'warrior2', 3: 'false'}

cmat_df_test=pd.DataFrame(
  confusion_matrix(y_true, y_pred, normalize='true').round(2),
  index=emotions.values(), 
  columns=emotions.values()
  )

plt.figure(figsize=(5,5))
heatmap(cmat_df_test,annot=True,cmap=plt.cm.Reds)
plt.tight_layout()
plt.title('Confusion Matrix on Test Set')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()