#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np

from sklearn.metrics import accuracy_score


# In[2]:


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


# In[3]:


for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))


# In[4]:


def one_hot(a, n_classes):
    return np.squeeze(np.eye(n_classes)[a.reshape(-1)])

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
#y_train = one_hot(y_train, 10)
#y_test = one_hot(y_test, 10)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255.0
X_test /= 255.0


# In[11]:


l = tf.keras.layers
inputs = l.Input(shape=(28, 28, 1))

splits = tf.split(inputs, 2, axis=0) #x1, x2 = tf.keras.layers.Lambda(lambda tensor: tf.split(tensor, 2, axis=1))(inputs)
x1, x2 = splits[0], splits[1]

x1 = l.Conv2D(16, 5, padding='same', activation=tf.nn.relu)(inputs)
x2 = l.Conv2D(16, 5, padding='same', activation=tf.nn.relu)(inputs)

x = tf.concat([x1, x2], axis=1)
x = l.MaxPooling2D((2, 2), (2, 2), padding='same')(x)
x = l.Conv2D(64, 5, padding='same', activation=tf.nn.relu)(x)
x = l.MaxPooling2D((2, 2), (2, 2), padding='same')(x)
x = l.Flatten()(x)
x = l.Dense(1024, activation=tf.nn.relu)(x)
x = l.Dropout(0.4)(x)
x = l.Dense(10, activation=tf.nn.softmax)(x)

outputs = x

model = tf.keras.models.Model(inputs=inputs, outputs=x)


# In[ ]:


model.compile(optimizer=tf.keras.optimizers.Adam(),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(),
             metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

#print(model(X_train[0]))

history = model.fit(X_train, y_train, batch_size=64, epochs=5, verbose=1)


# In[10]:


preds_test = model.predict(X_test)
acc_test = accuracy_score(one_hot(y_test, 10), preds_test)
print(f'Test accuracy: {acc_test}')

