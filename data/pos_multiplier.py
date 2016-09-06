def repeat_positives(in_data, n_times):
    for sample in in_data:
        if sample.state == 1:
            for _ in range(n_times):
                yield sample
        else:
            yield sample