import numpy


class NeuralNet:
    def __init__(self):
        self.layers = []

    def layer(self, layer):
        self.layers.append(layer)

    def train(self, data):
        pass

    def predict(self, input):
        pass


class InputLayer:
    def __init__(self, size):
        self.size = size
        self.matrix = numpy.empty(size)

        for i in range(self.matrix.shape[0]):
            self.matrix[i] = 2 * numpy.random.random(1) - 1


class OutputLayer:
    def __init__(self, previous_layer, size):
        self.size = size
        self.previous_layer = previous_layer
        self.matrix = numpy.empty(size)

        for i in range(self.matrix.shape[0]):
            self.matrix[i] = 2 * numpy.random.random(1) - 1


class HiddenLayer:
    def __init__(self, previous_layer, size):
        self.size = size
        self.matrix = numpy.empty((previous_layer.size, size))

        for i in range(0, self.matrix.shape[0]):
            for j in range(0, self.matrix.shape[1]):
                self.matrix[i, j] = 2 * numpy.random.random(1) - 1

