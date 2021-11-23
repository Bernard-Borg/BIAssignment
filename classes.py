import numpy
import pandas


class NeuralNet:
    def __init__(self):
        self.layers = []

    def layer(self, layer):
        self.layers.append(layer)

    def train(self, data):
        pass


class InputLayer:
    def __init__(self, size):
        self.size = size
        self.matrix = numpy.empty(size)

        for i in range(self.matrix.shape[0]):
            self.matrix[i] = (1 - 0) * numpy.random.random(1) + 0


class OutputLayer:
    def __init__(self, previous_layer, size):
        self.size = size
        self.previous_layer = previous_layer
        self.matrix = numpy.empty(size)

        for i in range(self.matrix.shape[0]):
            self.matrix[i] = (1 - 0) * numpy.random.random(1) + 0


class Layer:
    def __init__(self, previous_layer, size):
        self.matrix = numpy.empty(previous_layer.size, size)

        for i, j in self.matrix:
            self.matrix[i, j] = numpy.random.randn(-1, 1)

