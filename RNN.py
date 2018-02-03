import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

data = pd.read_csv("C:/Users/Asus1\OneDrive/workspace/manufacturing_Scheduling/timeSeries-master/timeSeries-master/dataPower.csv")
windPower = data['windPower'].tolist()

X = []
Y = []

for i in range(len(windPower)-31):
    data = windPower[i:i+31]
    a = []
    for j in range(28):
        a.append([data[j]])
    X.append(a)
    Y.append([data[28],data[29],data[30]])
X = np.array(X)
Y = np.array(Y)

train_indices = np.random.choice(len(X), int(len(X)*0.8), replace=False)
test_indices  = np.array(list(set(range(len(X))) - set(train_indices)))

trainX = X[train_indices]
testX  = X[test_indices]
trainY = Y[train_indices]
testY  = Y[test_indices]

# setting the random seed for reproducibility
seed = 0
tf.set_random_seed(seed)
np.random.seed(seed)

tf.reset_default_graph()   # We didn't have any previous graph objects running, but this would reset the graphs
num_periods = 28           # number of periods per vector we are using to predict one period ahead
inputs = 1                 # number of vectors submitted
hidden = 32                # number of neurons we will recursively work through, can be changed to improve accuracy
output = 3

x = tf.placeholder(tf.float32, [None, num_periods, inputs])   #create variable objects
y = tf.placeholder(tf.float32, [None, output])

def dynamicRNN(x):
    x = tf.unstack(x, num_periods, 1)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden, activation=tf.tanh)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.contrib.layers.fully_connected(outputs[-1], output, activation_fn=None)

pred = dynamicRNN(x)

# RMSE
targets = tf.placeholder(tf.float32, [None, 3])
predictions = tf.placeholder(tf.float32, [None, 3])
rmse = tf.sqrt(tf.reduce_sum(tf.square(targets - predictions)))

cost = tf.reduce_sum(tf.square(pred - y))  # sum of the squares
optimizer = tf.train.AdamOptimizer(0.00025)
train = optimizer.minimize(cost)

rate = 0.005
iterations = 1000
saver = tf.train.Saver()

with tf.Session() as sess:

    # init = tf.global_variables_initializer()
    # sess.run(init)
    saver.restore(sess,"C:/Users/Asus1/OneDrive/workspace/manufacturing_Scheduling/timeSeries-master/model.ckpt")
    step = 1
    epoch = 1

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, cost], feed_dict={x: trainX, y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

        if (i + 1)== iterations:
            save_path = saver.save(sess, "C:/Users/Asus1/OneDrive/workspace/manufacturing_Scheduling/timeSeries-master/model.ckpt")

    # Test step
    test_predict = sess.run(pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
        targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))
