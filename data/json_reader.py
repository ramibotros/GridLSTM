import json
from data import datum


def parsed_iterator(file_obj):
    _data = json.load(file_obj)
    for element in _data:
        _state = element["state"]  # output
        pattern = element["pattern"]  # input = 15 acids
        acids = [datum.Acid(_acid["statistics"], _acid["type"]) for _acid in pattern]

        sample = datum.Sample(acids, _state)
        yield sample

