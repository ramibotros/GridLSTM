import tensorflow as tf
from tensorflow.contrib.grid_rnn import Grid2BasicLSTMCell, Grid1BasicLSTMCell, GridRNNCell
from data import data_specs
import model_specs
import numpy as np
from operator import mul


def Grid2LSTMLayers(input_op):
    first_dim = tf.slice(tf.shape(input_op) , [0] , [1])

    lstm_cell = Grid2BasicLSTMCell(model_specs.HIDDEN_LAYER_SIZE)
    multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * model_specs.NUM_HIDDEN_LAYERS)


    lstm_outputs, _ = tf.nn.dynamic_rnn(multi_lstm_cell, tf.zeros_like(input_op), dtype=tf.float32)
    # lstm_outputs shape = [batch_size, WINDOW_SIZE, 1]

    last_output = tf.slice(lstm_outputs, [0, data_specs.WINDOW_SIZE - 1, 0], [-1, -1, -1])
    return tf.squeeze(last_output, [1])


def FullyConnectedLayers(input_op):
    all_but_first_dims = input_op.get_shape().as_list()[1:]
    cur_input = tf.reshape(input_op, [-1, reduce(mul, all_but_first_dims, 1)])
    for _ in range(model_specs.NUM_HIDDEN_LAYERS):
        cur_input = tf.contrib.layers.fully_connected(cur_input, model_specs.HIDDEN_LAYER_SIZE)
    return cur_input


class Network:
    def __init__(self, hidden_layers_op):
        self.data_input = tf.placeholder(tf.float32,
                                         [None, data_specs.WINDOW_SIZE, data_specs.AMINO_PARAM_SIZE])
        self.data_targets = tf.placeholder(tf.int32, [None])

        self.is_training_mode = tf.placeholder(tf.bool, [])


        self.batch_size = tf.placeholder(tf.int32, [1])
        data_input.set_shape()

        BN_inputs = tf.contrib.layers.batch_norm(self.data_input, updates_collections=None,
                                                 is_training=self.is_training_mode)

        hidden_layers_output = hidden_layers_op(BN_inputs)

        # Softmax:
        prediction = tf.contrib.layers.fully_connected(hidden_layers_output, 2)
        logits_flat = tf.reshape(prediction, [-1, 2])

        targets_one_hot_flat = tf.reshape(tf.one_hot(self.data_targets, 2), [-1, 2])
        self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits_flat, targets_one_hot_flat, pos_weight=10))

        optimizer = tf.train.AdamOptimizer(learning_rate=model_specs.LEARNING_RATE)
        self.train_once = optimizer.minimize(self.cost)

        self.choices = tf.squeeze(tf.arg_max(prediction, 1)) # for testing
