import json
from pprint import pprint
import data

def parse(file_obj):
    _data = json.load(file_obj)
    samples = []
    for element in _data:
        _state = element["state"] #output
        pattern = element["pattern"] #input = 15 acids
        acids = [data.acid(_acid["statistics"], _acid["type"]) for _acid in pattern]

        sample = data.sample(acids,_state)
        samples.append(sample)

    return samples

