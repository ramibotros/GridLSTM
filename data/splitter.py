import random
from data import pos_multiplier


def split_data(iterable, train_p, valid_p, test_p, arg_pos_multiplier=1):
    # returns training, validation and testing data.
    # pos_multiplier duplicates "positive" samples in training data only
    assert train_p + valid_p + test_p == 1

    all = list(iterable)
    random.shuffle(all)

    train_stop_idx = int(len(all) * train_p)
    valid_stop_idx = train_stop_idx + int(len(all) * valid_p)

    train_set = iterable[:train_stop_idx]
    valid_set = iterable[train_stop_idx:valid_stop_idx]
    test_set = iterable[valid_stop_idx:]

    train_set = list(pos_multiplier.repeat_positives(train_set, arg_pos_multiplier))

    return train_set, valid_set, test_set
