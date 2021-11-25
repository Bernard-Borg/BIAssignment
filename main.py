from neuralnetwork import NeuralNetwork
import pandas


def print_neural_network(neural_net: NeuralNetwork):
    print(neural_net.weight_matrix1)
    print(neural_net.weight_matrix2)


neural_network = NeuralNetwork(0.2)
print_neural_network(neural_network)

data = pandas.read_excel("titanic_dataset.xlsx")
normalized_data = (data-data.min())/(data.max()-data.min())

rows = normalized_data.shape[0]
bound = round(rows * 0.7)

training_data = normalized_data.loc[0:bound, 'P/Class':]
training_data_labels = normalized_data['Survived']

print(training_data)
print(training_data_labels)

neural_network.train(training_data, training_data_labels)
