import tensorflow as tf
from model import graph, model_specs
from data import json_reader
import random

with open("sample.txt") as f:
    all_samples = json_reader.parse(f)

random.shuffle(all_samples)
train_samples = all_samples[:int(len(all_samples)*0.9)]
test_samples = all_samples[int(len(all_samples)*0.9):]


network = graph.Network()
cost_op = network.cost
train_op = network.train_once
test_op = network.prediction

#Example Training:
ITERATIONS = 100
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for _ in range(ITERATIONS):
        sess.run(train_op, {network.data_input: _input, network.data_targets:_targets})