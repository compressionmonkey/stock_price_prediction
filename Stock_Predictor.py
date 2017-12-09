import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


# Import data
data = pd.read_csv('data_stocks.csv')
# Drop date column
data = data.drop(['DATE'], 1)
# Dimensions of dataset
n = data.shape[0] # number of rows
p = data.shape[1] # number of columns

# Make data a numpy array
data = data.values
# just have a numerical array

# plt.plot('NASDAQ.AAL')
# plt.show()

# Training and test data
train_start = 0
train_end = int(np.floor(0.8*n)) # round to the number
#below it. Eg if your number is 87.67 you say 87
# Ceiling does the opposite
#Out of eg 10 datasets. We have made 8 our training.
test_start = train_end + 1
# This is done to make 9th our testing dataset.
test_end = n
data_train = data[np.arange(train_start, train_end), :]
# Arranges data into:
# [[ 2363.6101    42.33     143.68   ...,    63.86     122.        53.35  ]
#  [ 2364.1001    42.36     143.7    ...,    63.74     121.77      53.35  ]
#  [ 2362.6799    42.31     143.6901 ...,    63.75     121.7       53.365 ]
#  ...,
#  [ 2475.05      50.54     158.0143 ...,    76.37     117.8688    61.535 ]
#  [ 2474.8601    50.52     157.8701 ...,    76.35     117.91      61.52  ]
#  [ 2474.6201    50.52     157.8    ...,    76.335    117.83      61.54  ]]

data_test = data[np.arange(test_start, test_end), :]


scaler = MinMaxScaler()
# A feature removing/ scaling algorithm that trys to create a range between 0 and 1 or -1.
# It starts with: [x1 - min(x)]/[max(x) - min(x)]
scaler.fit(data_train)
data_train = scaler.transform(data_train)
# I guess it transform it into scalar
data_test = scaler.transform(data_test)
# Build X and y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]


# Define a and b as placeholders
a = tf.placeholder(dtype=tf.int8) # Tensor("Placeholder:0", dtype=int8)

b = tf.placeholder(dtype=tf.int8) # Tensor("Placeholder_1:0", dtype=int8)


# Define the addition
c = tf.add(a, b)

# Initialize the graph
graph = tf.Session()

# Run the graph
d = graph.run(c, feed_dict={a: 5, b: 4}) # ok so adds two dicts to output c

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks]) # shape=[None, n_stocks] is a 2-dimensional matrix
#output is a 1-dimensional vector
# We use the None argument because we do not yet know the number of observations that will go through
# the neural net graph in each batch, so we keep if flexible. We will later define the variable batch_size
# that controls the number of observations per training batch
Y = tf.placeholder(dtype=tf.float32, shape=[None])
