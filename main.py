import classes as nn
import pandas


def print_neural_network(neural_net: nn.NeuralNet):
    for layer in neural_net.layers:
        print(layer.matrix)


neural_network = nn.NeuralNet()

input_layer = nn.InputLayer(5)
neural_network.layer(input_layer)

hidden_layer = nn.HiddenLayer(input_layer, 4)
neural_network.layer(hidden_layer)

output_layer = nn.OutputLayer(hidden_layer, 1)
neural_network.layer(output_layer)

data = pandas.read_excel("titanic_dataset.xlsx")
training_data = data.loc[0:, 'P/Class':]
training_data_label = data['Survived']
print(training_data)

neural_network.train(training_data)