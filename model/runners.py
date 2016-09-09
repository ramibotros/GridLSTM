from data import batcher
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score


def train_runner(sess, network, train_batch_generator, current_iteration):
    cost_op = network.cost
    train_op = network.train_once
    test_op = network.choices

    batch_inputs, batch_targets = next(train_batch_generator)
    training_cross_entropy, train_choices, _ = sess.run([cost_op, test_op, train_op],
                                                        {network.data_input: batch_inputs,
                                                         network.data_targets: batch_targets,
                                                         network.is_training_mode: True})
    train_mathews = matthews_corrcoef(batch_targets, train_choices)

    print("Iteration %d:\tTraining Cross Entropy = %f\tTrain Matthew's = %f" % (
        current_iteration, training_cross_entropy, train_mathews))

    return train_mathews

def test_runner(sess, network, iterable, batch_size, set_name):
    test_op = network.choices
    test_batch_generator = batcher.TestBatcher(iterable, batch_size)
    all_test_targets = []
    all_test_choices = []
    for batch_inputs, batch_targets in test_batch_generator:
        test_choices = sess.run(test_op, {network.data_input: batch_inputs,
                                          network.data_targets: batch_targets,
                                          network.is_training_mode: False})
        all_test_targets.extend(batch_targets)
        all_test_choices.extend(test_choices)
    test_matthews = matthews_corrcoef(all_test_targets, all_test_choices)
    test_precision = precision_score(all_test_targets, all_test_choices)
    test_recall = recall_score(all_test_targets, all_test_choices)
    print("%s Matthew's = %f\tPrecision = %f\tRecall = %f" % (set_name, test_matthews, test_precision, test_recall))

    return test_matthews

class ValidationPerformanceTracker:
    #task: if performance did not improve in last N runs, notify
    def __init__(self, track_n):
        self.track_n = track_n
        self.records = []
        self.IDs = []

    def get_winning_ID(self, last_result, id):
        if len(self.records) < self.track_n:
            self.records.append(last_result)
            self.IDs.append(id)
            return None
        else:
            self.records.pop()
            self.records.append(last_result)
            self.IDs.pop()
            self.IDs.append(id)

            best_index, _ = max(enumerate(self.records), key=lambda item: item[1])
            should_stop = best_index != len(self.records) - 1

            if should_stop:
                return self.IDs[best_index]
            else:
                return None