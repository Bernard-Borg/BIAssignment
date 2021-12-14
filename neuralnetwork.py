import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


class NeuralNetwork:
    def __init__(self, error_threshold=0.2, learning_rate=0.2, epochs=500):
        self.error_threshold = error_threshold
        self.learning_rate = learning_rate
        self.max_epochs = epochs
        self.weight_matrix1 = numpy.empty((5, 4))
        self.weight_matrix2 = numpy.empty((4, 1))

        for i in range(0, self.weight_matrix1.shape[0]):
            for j in range(0, self.weight_matrix1.shape[1]):
                self.weight_matrix1[i, j] = 2 * numpy.random.random(1) - 1

        for i in range(0, self.weight_matrix2.shape[0]):
            for j in range(0, self.weight_matrix2.shape[1]):
                self.weight_matrix2[i, j] = 2 * numpy.random.random(1) - 1

    def print(self):
        print(self.weight_matrix1)
        print(self.weight_matrix2)

    def train(self, data, data_labels):
        epochs = 0  # used as a counter for the number of epochs required for the network to converge
        epochs_bad_facts = []  # used to store the number of bad facts per epoch (for use in bad facts vs epochs graph)

        while True:

            print(epochs)

            bad_facts, good_facts = 0, 0

            for i in range(0, data.shape[0]):
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
                    else:
                        good_facts += 1

                # if it is a bad fact, perform error back propagation
                if bad:
                    delta_values1 = []

                    # calculate delta values for output neurons
                    for k in range(0, len(out_o)):
                        delta_values1.append(out_o[k] * (1 - out_o[k]) * errors[k])

                    # iterate over the weights and update them (for the output neurons)
                    for m in range(0, self.weight_matrix2.shape[0]):
                        weight_delta = self.learning_rate * delta_values1[0] * out_h[m]
                        self.weight_matrix2[m][0] += weight_delta

                    delta_values2 = []

                    # calculate delta values for hidden neurons
                    for k in range(0, len(out_h)):
                        delta_values2.append(out_h[k] * (1 - out_h[k]) * (self.weight_matrix2[k][0] * delta_values1[0]))

                    for k in range(0, self.weight_matrix1.shape[0]):
                        output = input_vector[k]

                        for m in range(0, self.weight_matrix1.shape[1]):
                            weight_delta = self.learning_rate * delta_values2[m] * output
                            self.weight_matrix1[k, m] += weight_delta

            epochs_bad_facts.append((epochs, bad_facts))

            if epochs > self.max_epochs:
                break

            epochs += 1

        return epochs_bad_facts

    def test(self, input_vectors, targets):
        i = 0

        correct = 0

        for i in range(input_vectors.shape[0]):
            net_h = numpy.dot(input_vectors.loc[i], self.weight_matrix1)
            out_h = sigmoid(net_h)

            net_o = numpy.dot(out_h, self.weight_matrix2)
            out_o = sigmoid(net_o)

            print(out_o[0])
            print(targets.loc[i])

            if abs(targets.loc[i] - out_o[0]) <= self.error_threshold:
                correct += 1

        print(correct)

        return (correct / i) * 100
