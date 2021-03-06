from operator import mul

import tensorflow as tf
from tensorflow.contrib.grid_rnn import Grid2BasicLSTMCell, Grid1BasicLSTMCell

from args_parser import options




def Grid2LSTMLayers(input_op, arg_is_training=True):
    lstm_cell = Grid2BasicLSTMCell(options.hidden_layer_size, tied=True)
    multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * options.hidden_layer_count)

    lstm_outputs, _ = tf.nn.dynamic_rnn(multi_lstm_cell, input_op, dtype=tf.float32)
    # lstm_outputs shape = [batch_size, WINDOW_SIZE, 1]

    last_output = tf.slice(lstm_outputs, [0, options.window_size - 1, 0], [-1, -1, -1])
    return tf.squeeze(last_output, [1])


def Grid1LSTMLayers(input_op, arg_is_training=True):
    all_but_first_dims = input_op.get_shape().as_list()[1:]

    product = 1
    for item in all_but_first_dims:
        product *= item

    lstm_cell = Grid1BasicLSTMCell(options.hidden_layer_size)
    multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * options.hidden_layer_count)

    lstm_outputs, _ = tf.nn.dynamic_rnn(multi_lstm_cell, input_op, dtype=tf.float32)
    # lstm_outputs shape = [batch_size, WINDOW_SIZE, 1]

    last_output = tf.slice(lstm_outputs, [0, options.window_size - 1, 0], [-1, -1, -1])
    return tf.squeeze(last_output, [1])


def FullyConnectedLayers(input_op, arg_is_training=True):
    all_but_first_dims = input_op.get_shape().as_list()[1:]

    product = 1
    for item in all_but_first_dims:
        product *= item

    cur_input = tf.reshape(input_op, [-1, product])
    for _ in range(options.hidden_layer_count):
        cur_input = tf.contrib.layers.fully_connected(cur_input, options.hidden_layer_size,
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={"updates_collections": None,
                                                                         "is_training": arg_is_training})
    return cur_input

def parse_hidden_layer_op(arg):
    if arg == "FC":
        return FullyConnectedLayers
    elif arg == "1GridLSTM":
        return Grid1LSTMLayers
    elif arg == "2GridLSTM":
        return Grid2LSTMLayers

class Network:
    def __init__(self, hidden_layers_op):
        self.data_input = tf.placeholder(tf.float32,
                                         [options.batch_size, options.window_size, options.acid_parameter_size])
        self.data_targets = tf.placeholder(tf.int32, [options.batch_size])

        self.is_training_mode = tf.placeholder(tf.bool, [])

        BN_inputs = tf.contrib.layers.batch_norm(self.data_input, updates_collections=None,
                                                 is_training=self.is_training_mode)

        hidden_layers_output = hidden_layers_op(BN_inputs, arg_is_training=self.is_training_mode)

        # Softmax:
        prediction = tf.contrib.layers.fully_connected(hidden_layers_output, 2)
        logits_flat = tf.reshape(prediction, [-1, 2])

        targets_one_hot_flat = tf.reshape(tf.one_hot(self.data_targets, 2), [-1, 2])
        self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits_flat, targets_one_hot_flat,
                                                                            pos_weight=options.positive_error_weight))

        optimizer = tf.train.AdamOptimizer(learning_rate=options.learning_rate)
        self.train_once = optimizer.minimize(self.cost)

        self.choices = tf.squeeze(tf.arg_max(prediction, 1))  # for testing
