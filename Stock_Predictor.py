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

# SP500 = data['SP500']
#
# plt.plot(SP500)
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

data_train = data[np.arange(train_start, train_end),:]
# Arranges data columns into an index:
# [[ 2363.6101    42.33     143.68   ...,    63.86     122.        53.35  ]
#  [ 2364.1001    42.36     143.7    ...,    63.74     121.77      53.35  ]
#  [ 2362.6799    42.31     143.6901 ...,    63.75     121.7       53.365 ]
#  ...,
#  [ 2475.05      50.54     158.0143 ...,    76.37     117.8688    61.535 ]
#  [ 2474.8601    50.52     157.8701 ...,    76.35     117.91      61.52  ]
#  [ 2474.6201    50.52     157.8    ...,    76.335    117.83      61.54  ]]

data_test = data[np.arange(test_start, test_end),:]

scaler = MinMaxScaler()
# A feature removing/ scaling algorithm that trys to create a range between 0 and 1 or -1.
# It starts with: [x1 - min(x)]/[max(x) - min(x)]
scaler.fit(data_train)
# now all data_train has been normalized to 0 and 1 range.
data_train = scaler.transform(data_train)
# Seems it applys the normalization to all.
data_test = scaler.transform(data_test)

# Build X and y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# # Initialize the graph
# graph = tf.Session()
#
# # Run the graph
# d = graph.run(c, feed_dict={a: 5, b: 4}) # ok so adds two dicts to output c

n_stocks = 500
# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks]) # shape=[None, n_stocks] is a 2-dimensional matrix
#output is a 1-dimensional vector
# We use the None argument because we do not yet know the number of observations that will go through
# the neural net graph in each batch, so we keep if flexible. We will later define the variable batch_size
# that controls the number of observations per training batch
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Model architecture parameters
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128 # we have 4 neural layers
n_target = 1

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
print(tf.matmul(X, W_hidden_1))
# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
print(out)

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Make Session
net = tf.Session()
# Run initializer
net.run(tf.global_variables_initializer())

# Setup interactive plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test*0.5)
plt.show()

# Number of epochs and batch size
epochs = 10
batch_size = 256

for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        if np.mod(i, 5) == 0:
            # Prediction
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            file_name = 'epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
            plt.savefig(file_name)
            plt.pause(0.01)
# Print final MSE after Training
mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
print(mse_final)
