import tensorflow as tf
from model import graph
from data import json_reader, batcher, pos_multiplier
import random
from sklearn.metrics import matthews_corrcoef
from args import args

all_samples = list(json_reader.parsed_iterator(args.data_file))

random.shuffle(all_samples)
train_samples = list(pos_multiplier.repeat_positives(all_samples[:int(len(all_samples) * 0.9)], args.positive_multiplier))
test_samples = all_samples[int(len(all_samples) * 0.9):]

print "Train has %d unordered cases" % sum([sample.state for sample in train_samples])
print "Test has %d unordered cases" % sum([sample.state for sample in test_samples])

train_batch_generator = batcher.Batcher(train_samples, args.batch_size)
test_batch = batcher.Batcher(test_samples, args.batch_size)

hidden_layer_op = graph.FullyConnectedLayers if args.hidden_layer_type == "FC" else graph.Grid2LSTMLayers

network = graph.Network(hidden_layers_op=hidden_layer_op)
cost_op = network.cost
train_op = network.train_once
test_op = network.choices

# Example Training:

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for _ in range(args.iterations):
        batch_inputs, batch_targets = next(train_batch_generator)
        training_cross_entropy, train_choices, _ = sess.run([cost_op, test_op, train_op],
                                                            {network.data_input: batch_inputs,
                                                             network.data_targets: batch_targets,
                                                             network.is_training_mode: True})
        train_mathews = matthews_corrcoef(batch_targets, train_choices)

        batch_inputs, batch_targets = next(test_batch)
        testing_cross_entropy, test_choices = sess.run([cost_op, test_op], {network.data_input: batch_inputs,
                                                                            network.data_targets: batch_targets,
                                                                            network.is_training_mode: False})
        test_mathew = matthews_corrcoef(batch_targets, test_choices)
        print "Training Cross Entropy = %f\tTesting Cross Entropy = %f\tTrain Mathew = %f\tTest Mathew = %f" % (
            training_cross_entropy, testing_cross_entropy, train_mathews, test_mathew)
