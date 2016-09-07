from operator import mul

import tensorflow as tf
from tensorflow.contrib.grid_rnn import Grid2BasicLSTMCell

from args import args


def Grid2LSTMLayers(input_op, arg_is_training=True):
    first_dim = tf.slice(tf.shape(input_op), [0], [1])

    lstm_cell = Grid2BasicLSTMCell(args.hidden_layer_size)
    multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * args.hidden_layer_count)

    lstm_outputs, _ = tf.nn.dynamic_rnn(multi_lstm_cell, input_op, dtype=tf.float32)
    # lstm_outputs shape = [batch_size, WINDOW_SIZE, 1]

    last_output = tf.slice(lstm_outputs, [0, args.window_size - 1, 0], [-1, -1, -1])
    return tf.squeeze(last_output, [1])


def FullyConnectedLayers(input_op, arg_is_training=True):
    all_but_first_dims = input_op.get_shape().as_list()[1:]

    product = 1
    for item in all_but_first_dims:
        product *= item

    cur_input = tf.reshape(input_op, [-1, product])
    for _ in range(args.hidden_layer_count):
        cur_input = tf.contrib.layers.fully_connected(cur_input, args.hidden_layer_size)
    return cur_input


class Network:
    def __init__(self, hidden_layers_op):
        self.data_input = tf.placeholder(tf.float32,
                                         [args.batch_size, args.window_size, args.acid_parameter_size])
        self.data_targets = tf.placeholder(tf.int32, [args.batch_size])

        self.is_training_mode = tf.placeholder(tf.bool, [])

        BN_inputs = tf.contrib.layers.batch_norm(self.data_input, updates_collections=None,
                                                 is_training=self.is_training_mode)

        hidden_layers_output = hidden_layers_op(BN_inputs, arg_is_training=self.is_training_mode)

        # Softmax:
        prediction = tf.contrib.layers.fully_connected(hidden_layers_output, 2)
        logits_flat = tf.reshape(prediction, [-1, 2])

        targets_one_hot_flat = tf.reshape(tf.one_hot(self.data_targets, 2), [-1, 2])
        self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits_flat, targets_one_hot_flat,
                                                                            pos_weight=args.positive_error_weight))

        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        self.train_once = optimizer.minimize(self.cost)

        self.choices = tf.squeeze(tf.arg_max(prediction, 1))  # for testing
