import tensorflow as tf


def get_weight(shape, lambda_1):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda_1)(var))
    return var

x = tf.placeholder(tf.float32, shape = (None, 2))
y_ = tf.placeholder(tf.float32, shape = (None, 1))
layer_dimension = [2, 10, 10, 10, 1]
n_layers = len(layer_dimension)
current_layer = x
in_dimension = layer_dimension[0]

for i in range(1, n_layers):
    out_dimension = layer_dimension[i]
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    current_layer = tf.nn.relu(tf.matmul(current_layer, weight) + bias)
    in_dimension = layer_dimension[i]

mse_loss = tf.reduce_mean(tf.square(y_ - current_layer))
tf.add_to_collection('losses', mse_loss)
loss = tf.add_n(tf.get_collection('losses'))