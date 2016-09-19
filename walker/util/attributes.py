
class attributes(object):
    ''' Attributes of the potential energy surface'''
    def __init__(self, walker):
        self.N = walker._N
        self.minima = walker._minima
        self.width = walker._width
        self.skew = walker._skew
        self.intensity = walker._intensity
        self.extent = walker._extent
        self.kbT = walker._surface_kbT
