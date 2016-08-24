
class attributes(object):
    def __init__(self, walker):
        self.N = walker._N
        self.minima = walker._minima
        self.width = walker._width
        self.skew = walker._skew
        self.depth = walker._depth
        self.T = walker._T
        self.kbT = walker._kbT
