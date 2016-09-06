import numpy as np
#define training data elements
import data_specs

class Acid:
    def __init__(self, _statistics, _type):
        #statistics: 20 numbers that describe some value related to each of the 20 possible letters of amino-acids
        #type: 1 = dummy , 0 = real
        assert len(_statistics) + 1 == data_specs.AMINO_PARAM_SIZE
        self.statistics = _statistics
        self.type = int(_type)

    def as_1D_array(self):
        return [self.type] + self.statistics

class Sample:
    def __init__(self, _acids, _state):
        assert len(_acids) == data_specs.WINDOW_SIZE
        self.acids = _acids

        self.state = _state

    def get_acids_2D_array(self):
        return [acid.as_1D_array() for acid in self.acids]
