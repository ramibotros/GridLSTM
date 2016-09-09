from __future__ import print_function
import tensorflow as tf
from args_parser import options
from data import json_reader, batcher, splitter
from model import graph, runners

train_samples = list(json_reader.parsed_iterator(options.train_file))
valid_samples = list(json_reader.parsed_iterator(options.valid_file))
test_samples = list(json_reader.parsed_iterator(options.test_file))
#train_samples, valid_samples, test_samples = splitter.split_data(all_samples, 0.8, 0.1, 0.1, options.positive_multiplier)

print("Train has %d unordered cases out of total %d" % (
    sum([sample.state for sample in train_samples]), len(train_samples)))
print("Valid has %d unordered cases out of total %d" % (
    sum([sample.state for sample in test_samples]), len(valid_samples)))
print("Test has %d unordered cases out of total %d" % (
        sum([sample.state for sample in test_samples]), len(test_samples)))

train_batch_generator = batcher.Batcher(train_samples, options.batch_size)

hidden_layer_op = graph.parse_hidden_layer_op(options.hidden_layer_type)
network = graph.Network(hidden_layers_op=hidden_layer_op)

validation_performance_tracker = runners.ValidationPerformanceTracker(options.early_stop_after)

saver = tf.train.Saver()
save_paths = dict()
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    if options.load_path:
        saver.restore(sess, options.load_path)
        print("Model restored from path %s." % options.load_path)

    for current_iteration in range(options.iterations):
        runners.train_runner(sess, network, train_batch_generator, current_iteration)

        if (current_iteration+1) % options.test_every == 0:
            valid_matthews = runners.test_runner(sess, network, valid_samples, options.batch_size, "ValidSet")
            runners.test_runner(sess, network, test_samples, options.batch_size, "TestSet")

            save_path = saver.save(sess, options.save_path_basename, global_step=current_iteration)
            print("Model saved in file: %s" % save_path)
            save_paths[current_iteration] = save_path

            #Early stopping:
            winning_iter = validation_performance_tracker.get_winning_ID(valid_matthews, current_iteration)
            if winning_iter is not None:
                print ("\nStopped because performance has not improved for the last %d ValidSet test runs." % options.early_stop_after)
                print ("Best iteration is %d :" % winning_iter)
                print ("Best model path : %s" % save_paths[winning_iter])
                break

