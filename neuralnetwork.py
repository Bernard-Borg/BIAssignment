import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


class NeuralNetwork:
    def __init__(self, error_threshold=0.2, learning_rate=0.2, epochs=500):
        self.error_threshold = error_threshold
        self.learning_rate = learning_rate
        self.max_epochs = epochs
        self.weight_matrix1 = 2 * numpy.random.random((5, 4)) - 1
        self.weight_matrix2 = 2 * numpy.random.random((4, 1)) - 1

    def train(self, data, data_labels):
        epochs = 0  # used as a counter for the number of epochs required for the network to converge
        epochs_bad_facts = []  # used to store the number of bad facts per epoch (for use in bad facts vs epochs graph)

        while epochs < self.max_epochs:
            bad_facts = 0

            for i in range(data.shape[0]):
                input_vector = data.loc[i]

                # feed forward calculations
                net_h = numpy.dot(input_vector, self.weight_matrix1)
                out_h = sigmoid(net_h)

                net_o = numpy.dot(out_h, self.weight_matrix2)
                out_o = sigmoid(net_o)

                errors = []
                bad = False

                # calculating errors and seeing if bad/good fact
                for j in range(len(out_o)):
                    error = data_labels.loc[i] - out_o[j]
                    errors.append(error)

                    if abs(error) > self.error_threshold:
                        bad_facts += 1
                        bad = True

                # if it is a bad fact, perform error back propagation
                if bad:
                    delta_values1 = []

                    # calculate delta values for output neurons
                    for k in range(len(out_o)):
                        delta_values1.append(out_o[k] * (1 - out_o[k]) * errors[k])

                    self.update_weights(self.weight_matrix2, delta_values1, out_h)
                    delta_values2 = self.calculate_hidden_layer_delta_values(self.weight_matrix2, out_h, delta_values1)
                    self.update_weights(self.weight_matrix1, delta_values2, input_vector)

            epochs_bad_facts.append((epochs, bad_facts))
            epochs += 1

        return epochs_bad_facts

    # calculates the delta values for a specific hidden layer, given
    #   the weight matrix,
    #   the current values and
    #   the delta values of the next layer
    def calculate_hidden_layer_delta_values(self, matrix, current_values, previous_deltas):
        delta_values = []

        for k in range(len(current_values)):
            summation = 0

            for m in range(len(previous_deltas)):
                summation += matrix[k][m] * previous_deltas[m]

            delta_values.append(current_values[k] * (1 - current_values[k]) * summation)

        return delta_values

    # updates the weight matrix given the delta values and the previous values
    def update_weights(self, matrix, delta_values, previous_values):
        for x in range(matrix.shape[0]):
            output = previous_values[x]

            for y in range(matrix.shape[1]):
                weight_delta = self.learning_rate * delta_values[y] * output
                matrix[x][y] += weight_delta

    # iterates over the data and tests each record, returning the accuracy
    def test(self, input_vectors, targets):
        i = 0
        correct = 0

        for i in range(input_vectors.shape[0]):
            net_h = numpy.dot(input_vectors.loc[i], self.weight_matrix1)
            out_h = sigmoid(net_h)

            net_o = numpy.dot(out_h, self.weight_matrix2)
            out_o = sigmoid(net_o)

            if abs(targets.loc[i] - out_o[0]) <= self.error_threshold:
                correct += 1

        return (correct / (i + 1)) * 100