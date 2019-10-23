#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np

from sklearn.metrics import accuracy_score

from tqdm import tqdm_notebook as tqdm


# In[2]:


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


# In[3]:


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


# In[69]:


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))


# In[37]:


class Model:
    def __init__(self):
        l = tf.keras.layers
        self.pool = l.MaxPooling2D((2, 2), (2, 2), padding='same')
        self.conv1 = l.Conv2D(32, 5, padding='SAME', activation=tf.nn.relu)
        self.conv2 = l.Conv2D(64, 5, padding='SAME', activation=tf.nn.relu)
        self.flatten = l.Flatten()
        self.dropout = l.Dropout(0.4)
        self.dense1 = l.Dense(1024, activation=tf.nn.relu)
        self.dense2 = l.Dense(10, activation=tf.nn.softmax)
        
        layers = [self.pool, self.conv1, self.conv2, self.flatten, self.dropout, self.dense1, self.dense2]
        self.trainable_variables = [v for v in (layer.trainable_variables for layer in layers)]
        
        self.optimizer = tf.optimizers.Adam()
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.batch_size = 128

    def build(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        
        return x

    def train(self, dataset, epochs):
        dataset = dataset.batch(self.batch_size)
        for epoch in range(epochs):
            print(f'[Epoch {epoch+1}/{epochs}]')
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            
            progress = tqdm(total=X_train.shape[0], unit='samples')
            for x, y in dataset:
                progress.update(self.batch_size)
                preds = self.forward(x)
                loss = lambda: self.loss_function(y_true=y, y_pred=preds)
                #grads = tape.gradient(loss, self.trainable_variables)
                self.optimizer.minimize(loss, self.trainable_variables)
                epoch_loss_avg(loss())
                epoch_accuracy(y, preds)
            print(f'Loss: {epoch_loss_avg.result():.4f}, Accuracy: {epoch_accuracy.result():.2%}')
            print()
    


# In[71]:


def build_model(split_convolutions=False):
    l = tf.keras.layers
    
    inputs = l.Input(shape=(28, 28, 1))
    weights = {
        'wc1': tf.Variable(tf.random.normal([5, 5, 1, 32])),
    }
    
    biases = {
        'bc1': tf.Variable(tf.random.normal([32]))
    }
    
    
    x = inputs
    if split_convolutions:
        x1, x2 = tf.split(x, 2, axis=1)
        x1 = tf.nn.conv2d(x1, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
        x1 = tf.nn.bias_add(x1, biases['bc1'])
        x2 = tf.nn.conv2d(x2, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
        x2 = tf.nn.bias_add(x2, biases['bc1'])
        x = tf.concat([x1, x2], axis=1)
    else:
        #x = l.Conv2D(32, 5, padding='SAME', activation=tf.nn.relu)(x)
        x = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, biases['bc1'])
    
    x = tf.nn.max_pool(x, (2, 2), (2, 2), 'SAME')
    x = l.Conv2D(64, 5, padding='SAME', activation=tf.nn.relu)(x)
    x = tf.nn.max_pool(x, (2, 2), (2, 2), 'SAME')
    x = l.Flatten()(x)
    x = l.Dense(1024, activation=tf.nn.relu)(x)
    x = tf.nn.dropout(x, 0.4)
    x = l.Dense(10, activation=tf.nn.softmax)(x)
    
    outputs = x
    
    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


# In[ ]:


model = build_model(True)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(train_dataset.shuffle(10000).repeat(5).batch(128), epochs=5, steps_per_epoch=X_train.shape[0] // 128)

