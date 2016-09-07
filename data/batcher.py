import random


class Batcher:
    def __init__(self, iterable, batch_size, test_mode=False):
        self.batch_size = batch_size
        self.data = list(iterable)

        while len(self.data) < self.batch_size:
            self.data = self.data + self.data

        self.test_mode = test_mode

    def __iter__(self):
        return self


    def next(self):
        _inputs = []
        _targets = []

        chosen = random.sample(self.data, self.batch_size) if not self.test_mode else self.data
        for sample in chosen:
            features, target = sample.get_acids_2D_array(), sample.state
            _inputs.append(features)
            _targets.append(target)
        return _inputs, _targets


    def __next__(self):
        return self.next()