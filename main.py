from __future__ import print_function
import tensorflow as tf
from model import graph
from data import json_reader, batcher, pos_multiplier
import random
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from args import args

all_samples = list(json_reader.parsed_iterator(args.data_file))

random.shuffle(all_samples)
train_samples = list(pos_multiplier.repeat_positives(all_samples[:int(len(all_samples) * 0.9)], args.positive_multiplier))
test_samples = all_samples[int(len(all_samples) * 0.9):]

print ("Train has %d unordered cases" % sum([sample.state for sample in train_samples]))
print ("Test has %d unordered cases" % sum([sample.state for sample in test_samples]))

train_batch_generator = batcher.Batcher(train_samples, args.batch_size)

hidden_layer_op = None
if args.hidden_layer_type == "FC":
    hidden_layer_op = graph.FullyConnectedLayers
elif args.hidden_layer_type == "1GridLSTM":
    hidden_layer_op = graph.Grid1LSTMLayers
elif args.hidden_layer_type == "2GridLSTM":
    hidden_layer_op = graph.Grid2LSTMLayers

network = graph.Network(hidden_layers_op=hidden_layer_op)
cost_op = network.cost
train_op = network.train_once
test_op = network.choices

# Example Training:
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    if args.load_path:
        saver.restore(sess, args.load_path)
        print("Model restored from path %s." % args.load_path)

    for current_iteration in range(args.iterations):
        batch_inputs, batch_targets = next(train_batch_generator)
        training_cross_entropy, train_choices, _ = sess.run([cost_op, test_op, train_op],
                                                            {network.data_input: batch_inputs,
                                                             network.data_targets: batch_targets,
                                                             network.is_training_mode: True})
        train_mathews = matthews_corrcoef(batch_targets, train_choices)

        print("Iteration %d:\tTraining Cross Entropy = %f\tTrain Matthew's = %f" % (
            current_iteration, training_cross_entropy, train_mathews))

        if current_iteration % args.test_every == 0 :
            test_batch_generator = batcher.TestBatcher(test_samples, args.batch_size)
            all_test_targets = []
            all_test_choices = []
            for batch_inputs, batch_targets in test_batch_generator:
                test_choices = sess.run(test_op, {network.data_input: batch_inputs,
                                                                                    network.data_targets: batch_targets,
                                                                                    network.is_training_mode: False})
                all_test_targets.extend(batch_targets)
                all_test_choices.extend(test_choices)
            test_mathew = matthews_corrcoef(all_test_targets, all_test_choices)
            test_precision = precision_score(all_test_targets, all_test_choices)
            test_recall = recall_score(all_test_targets, all_test_choices)
            print ("Test Matthew's = %f\tPrecision = %f\tRecall = %f" % (test_mathew, test_precision, test_recall))

            save_path = saver.save(sess, args.save_path_basename, global_step=current_iteration)
            print("Model saved in file: %s" % save_path)
