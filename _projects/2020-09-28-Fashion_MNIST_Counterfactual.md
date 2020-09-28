---
layout: project
title: Counterfactuals in Python on Fashion MNIST dataset 
description: Counterfactuals in Python on Fashion MNIST dataset 
category: Interpretability
---

Implementation of counterfactuals using the Fashion MNIST dataset and using [Alibi](https://github.com/SeldonIO/alibi) 

Start with loading in all the necessary libraries. 


```python
import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from time import time
from alibi.explainers import CounterFactual
print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly()) # False
```

    TF version:  2.0.0
    Eager execution enabled:  False


Load in my dataset 


```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)
plt.gray()
plt.imshow(x_test[1]);
```

    x_train shape: (60000, 28, 28) y_train shape: (60000,)



![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/Fashion_MNIST_Counterfactual_files/Fashion_MNIST_Counterfactual_3_1.png)


Scale and reshape my dataset 


```python
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, x_train.shape + (1,))
x_test = np.reshape(x_test, x_test.shape + (1,))
print('x_train shape:', x_train.shape, 'x_test shape:', x_test.shape)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print('y_train shape:', y_train.shape, 'y_test shape:', y_test.shape)
```

    x_train shape: (60000, 28, 28, 1) x_test shape: (10000, 28, 28, 1)
    y_train shape: (60000, 10) y_test shape: (10000, 10)



```python
xmin, xmax = -.5, .5
x_train = ((x_train - x_train.min()) / (x_train.max() - x_train.min())) * (xmax - xmin) + xmin
x_test = ((x_test - x_test.min()) / (x_test.max() - x_test.min())) * (xmax - xmin) + xmin
```

Build my CNN 


```python
def cnn_model():
    x_in = Input(shape=(28, 28, 1))
    x = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')(x_in)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x_out = Dense(10, activation='softmax')(x)

    cnn = Model(inputs=x_in, outputs=x_out)
    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return cnn
```


```python
cnn = cnn_model()
cnn.summary()
cnn.fit(x_train, y_train, batch_size=64, epochs=3)
```

    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         [(None, 28, 28, 1)]       0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 28, 28, 64)        320       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 14, 14, 64)        0         
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 14, 14, 64)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 14, 14, 32)        8224      
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 7, 7, 32)          0         
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 7, 7, 32)          0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 1568)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 256)               401664    
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                2570      
    =================================================================
    Total params: 412,778
    Trainable params: 412,778
    Non-trainable params: 0
    _________________________________________________________________
    Train on 60000 samples
    Epoch 1/3
    60000/60000 [==============================] - 46s 768us/sample - loss: 0.5812 - acc: 0.7880
    Epoch 2/3
    60000/60000 [==============================] - 46s 772us/sample - loss: 0.4006 - acc: 0.8547
    Epoch 3/3
    60000/60000 [==============================] - 48s 793us/sample - loss: 0.3606 - acc: 0.8686





    <tensorflow.python.keras.callbacks.History at 0x145f5a7d0>




```python
score = cnn.evaluate(x_test, y_test, verbose=0)
print('Test accuracy: ', score[1])
```

    Test accuracy:  0.8853


Grab my example to check the counterfactual 


```python
X = x_test[0].reshape((1,) + x_test[0].shape)
plt.imshow(X.reshape(28, 28));
```


![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/Fashion_MNIST_Counterfactual_files/Fashion_MNIST_Counterfactual_12_0.png)


To start I want to what changes to the above image would have to occur to change it to any class.  I specified `other` for this reason. 


```python
shape = (1,) + x_train.shape[1:]
target_proba = 1.0
tol = 0.01 # want counterfactuals with p(class)>0.99
target_class = 'other' # any class other than the current class
max_iter = 1000
lam_init = 1e-1
max_lam_steps = 10
learning_rate_init = 0.1
feature_range = (x_train.min(),x_train.max())
```


```python
# initialize explainer
cf = CounterFactual(cnn, shape=shape, target_proba=target_proba, tol=tol,
                    target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                    max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                    feature_range=feature_range)

start_time = time()
explanation = cf.explain(X)
print('Explanation took {:.3f} sec'.format(time() - start_time))
```

    Explanation took 8.955 sec



```python
pred_class = explanation.cf['class']
proba = explanation.cf['proba'][0][pred_class]

print(f'Counterfactual prediction: {pred_class} with probability {proba}')

fig, ax = plt.subplots(ncols = 2, figsize = (12, 8))
ax[0].imshow(explanation.cf['X'].reshape(28, 28))
ax[0].set_title('Counterfactual - Sneaker')

ax[1].imshow(X.reshape(28, 28))
ax[1].set_title('Original - Ankle Boot');
```

    Counterfactual prediction: 7 with probability 0.9917559623718262



![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/Fashion_MNIST_Counterfactual_files/Fashion_MNIST_Counterfactual_16_1.png)


The image on the left shows what would have to occur for the model to predict the image to be a sneaker instead of a ankle boot with greater than 99% probability.  


```python
target_class = 1

cf = CounterFactual(cnn, shape=shape, target_proba=target_proba, tol=tol,
                    target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                    max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                    feature_range=feature_range)

explanation = start_time = time()
explanation = cf.explain(X)
print('Explanation took {:.3f} sec'.format(time() - start_time))
```

    Explanation took 6.988 sec



```python
pred_class = explanation.cf['class']
proba = explanation.cf['proba'][0][pred_class]

print(f'Counterfactual prediction: {pred_class} with probability {proba}')

fig, ax = plt.subplots(ncols = 2, figsize = (12, 8))
ax[0].imshow(explanation.cf['X'].reshape(28, 28))
ax[0].set_title('Counterfactual - Trouser')

ax[1].imshow(X.reshape(28, 28))
ax[1].set_title('Original - Ankle Boot');
```

    Counterfactual prediction: 1 with probability 0.9915237426757812



![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/Fashion_MNIST_Counterfactual_files/Fashion_MNIST_Counterfactual_19_1.png)


The image on the left is what would have to occur for the model to predict the image to be a trouser with greater than 99% probability. 


```python
target_class = 3

cf = CounterFactual(cnn, shape=shape, target_proba=target_proba, tol=tol,
                    target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                    max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                    feature_range=feature_range)

explanation = start_time = time()
explanation = cf.explain(X)
print('Explanation took {:.3f} sec'.format(time() - start_time))
```

    Explanation took 6.274 sec



```python
pred_class = explanation.cf['class']
proba = explanation.cf['proba'][0][pred_class]

print(f'Counterfactual prediction: {pred_class} with probability {proba}')

fig, ax = plt.subplots(ncols = 2, figsize = (12, 8))
ax[0].imshow(explanation.cf['X'].reshape(28, 28))
ax[0].set_title('Counterfactual - Dress')

ax[1].imshow(X.reshape(28, 28))
ax[1].set_title('Original - Ankle Boot');
```

    Counterfactual prediction: 3 with probability 0.990838348865509



![png](https://raw.githubusercontent.com/sik-flow/sik-flow.github.io/master/_projects/images/Fashion_MNIST_Counterfactual_files/Fashion_MNIST_Counterfactual_22_1.png)


Finally, we the image on the left is what the model would need to predict the image to be a dress with greater than 99% probability. 
