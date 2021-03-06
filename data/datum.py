from args_parser import options


#define training data elements
class Acid:
    def __init__(self, _statistics, _type):
        #statistics: 20 numbers that describe some value related to each of the 20 possible letters of amino-acids
        #type: 1 = dummy , 0 = real
        assert len(_statistics) + 1 == options.acid_parameter_size
        self.statistics = _statistics
        self.type = int(_type)

    def as_1D_array(self):
        return [self.type] + self.statistics

class Sample:
    def __init__(self, _acids, _state):
        assert len(_acids) == options.window_size
        self.acids = _acids

        self.state = _state

    def get_acids_2D_array(self):
        return [acid.as_1D_array() for acid in self.acids]
