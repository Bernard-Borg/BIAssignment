from neuralnetwork import NeuralNetwork
import pandas
import matplotlib.pyplot as plot

# Construct neural network object
neural_network = NeuralNetwork(epochs=1000)

# Get the data from excel and store as pandas DataFrame
data = pandas.read_excel("titanic_dataset.xlsx")

# Randomise the data rows
randomised_data = data.sample(frac=1).reset_index(drop=True)

# Normalise the data
normalised_data = (randomised_data - randomised_data.min()) / (randomised_data.max() - randomised_data.min())

# Calculate the number of rows required for 80%
rows = normalised_data.shape[0]
bound = round(rows * 0.8)

# Take the first 80% of rows for training
training_data = normalised_data.loc[0:bound, 'P/Class':]
training_data_labels = normalised_data.loc[0:bound, 'Survived']

# Take the last 20% of rows for testing
testing_data = normalised_data.loc[bound:, 'P/Class':].reset_index(drop=True)
testing_data_labels = normalised_data.loc[bound:, 'Survived'].reset_index(drop=True)

# Train network
epochs_vs_bad_facts = neural_network.train(training_data, training_data_labels)

# Output matplotlib epochs vs bad facts graph
x_axis = []
y_axis = []

for i, j in epochs_vs_bad_facts:
    x_axis.append(i)
    y_axis.append(j)

plot.plot(x_axis, y_axis)
plot.title("Bad facts vs epochs (learning rate: " + str(neural_network.learning_rate) +
           ", error threshold: " + str(neural_network.error_threshold) + ")")
plot.xlabel("Epochs")
plot.ylabel("Bad facts")

correct_percentage = neural_network.test(testing_data, testing_data_labels)
print("Accuracy: " + str(correct_percentage))

plot.show()
