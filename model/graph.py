import tensorflow as tf
from tensorflow.contrib.grid_rnn import Grid2BasicLSTMCell, Grid1BasicLSTMCell, GridRNNCell
from data import data_specs
import model_specs
import numpy as np

class Network:
    def __init__(self):
        self.data_input = tf.placeholder(tf.float32, [model_specs.BATCH_SIZE, data_specs.WINDOW_SIZE , data_specs.AMINO_PARAM_SIZE])
        self.data_targets = tf.placeholder(tf.int32, [model_specs.BATCH_SIZE, data_specs.WINDOW_SIZE])


        lstm_cell = Grid2BasicLSTMCell(model_specs.HIDDEN_LAYER_SIZE)
        multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * model_specs.NUM_HIDDEN_LAYERS)

        lstm_outputs , _ = tf.nn.dynamic_rnn(lstm_cell, self.data_input, dtype=tf.float32)
        #lstm_outputs shape = [batch_size, WINDOW_SIZE, 1]

        self.prediction = tf.contrib.layers.fully_connected(lstm_outputs, 2)
        targets_one_hot = tf.one_hot(self.data_targets, 2)

        logits_flat = tf.reshape(self.prediction, [-1, 2])
        targets_one_hot_flat = tf.reshape(targets_one_hot, [-1, 2])
        self.cost = tf.nn.softmax_cross_entropy_with_logits(logits_flat, targets_one_hot_flat)

        optimizer = tf.train.AdamOptimizer()
        self.train_once = optimizer.minimize(self.cost)
