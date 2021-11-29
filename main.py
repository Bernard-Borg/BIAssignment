from neuralnetwork import NeuralNetwork
import pandas
import matplotlib.pyplot as plot


# Helper function to print neural network (for debugging)
def print_neural_network(neural_net: NeuralNetwork):
    print(neural_net.weight_matrix1)
    print(neural_net.weight_matrix2)


# Construct neural network object
neural_network = NeuralNetwork(0.2)
print_neural_network(neural_network)

# Get the data from excel and store as pandas DataFrame
data = pandas.read_excel("titanic_dataset.xlsx")

# Get normalised data
normalized_data = (data-data.min())/(data.max()-data.min())

# Calculate the number of rows required for 70%
rows = normalized_data.shape[0]
bound = round(rows * 0.7)

# Take the first 70% of rows for training
training_data = normalized_data.loc[0:bound, 'P/Class':]
training_data_labels = normalized_data.loc[0:bound, 'Survived']

# Take the last 30% of rows for testing
testing_data = normalized_data.loc[bound:, 'P/Class':]
testing_data_labels = normalized_data.loc[bound:, 'Survived']

# Train network
epochs_vs_bad_facts = neural_network.train(training_data, training_data_labels)
print_neural_network(neural_network)

# Output matplotlib epochs vs bad facts graph
x_axis = []
y_axis = []

for i, j in epochs_vs_bad_facts:
    x_axis.append(i)
    y_axis.append(j)

plot.plot(x_axis, y_axis)
plot.title("Bad facts vs epochs")
plot.xlabel("Epochs")
plot.ylabel("Bad facts")
plot.show()

print(epochs_vs_bad_facts)