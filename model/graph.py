import tensorflow as tf
from tensorflow.contrib.grid_rnn import Grid2BasicLSTMCell, Grid1BasicLSTMCell, GridRNNCell
from data import data_specs
import model_specs
import numpy as np

data_input = tf.placeholder(tf.float32, [model_specs.BATCH_SIZE, data_specs.WINDOW_SIZE , data_specs.AMINO_PARAM_SIZE])
data_targets = tf.placeholder(tf.float32, [model_specs.BATCH_SIZE, data_specs.WINDOW_SIZE])


lstm_cell = Grid2BasicLSTMCell(model_specs.HIDDEN_LAYER_SIZE)
multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * model_specs.NUM_HIDDEN_LAYERS)

outputs , _ = tf.nn.dynamic_rnn(lstm_cell, data_input, dtype=tf.float32)
#outputs shape = [batch_size, WINDOW_SIZE, 1]


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    _input = np.random.rand(model_specs.BATCH_SIZE, data_specs.WINDOW_SIZE , data_specs.AMINO_PARAM_SIZE)
    _targets = np.random.rand(model_specs.BATCH_SIZE, data_specs.WINDOW_SIZE)

    _output_shape = sess.run(tf.shape(outputs), {data_input: _input, data_targets:_targets})
    print _output_shape