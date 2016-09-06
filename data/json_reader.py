import json
from pprint import pprint
import datum


def parser(file_obj):
    _data = json.load(file_obj)
    for element in _data:
        _state = element["state"]  # output
        pattern = element["pattern"]  # input = 15 acids
        acids = [datum.acid(_acid["statistics"], _acid["type"]) for _acid in pattern]

        sample = datum.sample(acids, _state)
        yield sample
