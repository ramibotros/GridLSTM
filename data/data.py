import numpy as np
#define training data elements
import data_specs

class acid:
    def __init__(self, _statistics, _type):
        #statistics: 20 numbers that describe some value related to each of the 20 possible letters of amino-acids
        #type: 1 = dummy , 0 = real
        assert len(_statistics) == data_specs.AMINO_PARAM_SIZE
        self.statistics = np.array(_statistics, dtype=np.int32)

        self.type = int(_type)


class sample:
    def __init__(self, _acids, _state):
        assert len(_acids) == data_specs.WINDOW_SIZE
        self.acids = _acids

        self.state = _state
