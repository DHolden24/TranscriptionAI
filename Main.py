import ConvolutionalNeuralNetwork as cnn
import pickle
import random as rand
import gc
from DataSetGenerator import generate_data

# generate_data()
# gc.collect()

print("Starting")
with open('Music/dataset.pkl', 'rb') as f:
    data = pickle.load(f)
print("Data Loaded")
rand.shuffle(data)

x_train, y_train = [], []
for x, y in data[0:int(0.9*len(data))]:
    x.shape = (1, x.shape[0], x.shape[1])
    x_train.append(x)
    y_train.append(y)

x_test, y_test = [], []
for x, y in data[int(0.9*len(data)):]:
    x.shape = (1, x.shape[0], x.shape[1])
    x_test.append(x)
    y_test.append(y)

print("Data Split")
data.clear()
gc.collect()

cnn = cnn.ConvolutionalNeuralNetwork(128)
print("Starting Training")
cnn.train_set(x_train, y_train, 1, x_test, y_test)

with open("ai.pkl", 'wb') as f:
    pickle.dump(cnn, f)

