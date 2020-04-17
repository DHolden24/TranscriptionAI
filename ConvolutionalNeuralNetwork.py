import numpy as np

class ConvolutionalNeuralNetwork:

    def __init__(self, num_outputs):
        self.layers = []
        self.layers.append(ConvolutionLayer(1, 3, 8, 0))
        # self.layers.append(ConvolutionLayer(4, 3, 8, 0))
        self.layers.append(PoolLayer(2))
        self.layers.append(FullyConnectedLayer(300, 8 * 576 * 4))
        self.layers.append(FullyConnectedLayer(num_outputs, 300))

    def predict(self, input_data):
        result = self.layers[0].forward_prop(input_data)

        for l in self.layers[1:]:
            result = l.forward_prop(result)

        return result

    def train(self, input_data, expected):
        result = self.predict(input_data)

        errors = expected - result

        for i in range(len(self.layers), 0, -1):
            if type(self.layers[i-1]) in [FullyConnectedLayer, ConvolutionLayer]:
                errors = self.layers[i-1].backward_prop(errors, 0.005)
            else:
                errors = self.layers[i - 1].backward_prop(errors)


    def train_set(self, input_set, expected_set, epochs, test_set=None, test_expected=None):
        print("{} training points".format(len(input_set)))
        for e in range(epochs):
            print("Starting epoch {}".format(e))
            for i in range(len(input_set)):
                if i % 1000 == 0: print("Completed {} of {}".format(i, len(input_set)))
                self.train(input_set[i], expected_set[i])

            print("Starting Tests")
            if test_set is not None and test_expected is not None:
                correct_count = [0] * 5
                for i in range(len(test_set)):
                    result = self.predict(test_set[i])
                    print(result.shape)

                    norm_result = np.zeros(result.shape)
                    for j in range(result.shape[0]):
                        if result[j] > 0.7:
                            norm_result[j] = 1

                    expected = test_expected[i]
                    correct = 0
                    for j in range(len(expected)):
                        if expected[j] == norm_result[j]:
                            correct += 1

                    for j in range(len(correct_count)):
                        if correct >= 128 - j:
                            correct_count[j] += 1

                for i in range(len(correct_count)):
                    print("{} or fewer wrong notes".format(i))
                    print("\t{} correct out of {}".format(correct_count[i], len(test_set)))
                    print("\t{} % accurate".format(correct_count[i] / len(test_set) * 100))
                print('\n')


class ConvolutionLayer:

    def __init__(self, input_channels=1, filter_dim=3, num_filters=1, padding=0):
        self.num_filters = num_filters
        self.filter_dim = filter_dim
        self.filters = np.random.randn(num_filters, input_channels, filter_dim, filter_dim) / (filter_dim ** 2)
        self.biases = [0] * num_filters
        self.stride = 1
        self.padding = padding
        self.input_layers = None

    def forward_prop(self, input_layers):
        input_layers= np.pad(input_layers, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        _, height, width = input_layers.shape
        self.input_layers = input_layers.copy()

        output_layers = np.zeros((self.num_filters, (height - self.filter_dim) // self.stride + 1,
                                 (width - self.filter_dim) // self.stride + 1))

        _, out_height, out_width = output_layers.shape

        for h in range(out_height):
            for w in range(out_width):
                for f in range(self.num_filters):
                    output_layers[f, h, w] = np.sum(input_layers[:, h:h+self.filter_dim, w:w+self.filter_dim]
                                                    * self.filters[f]) + self.biases[f]
                    if output_layers[f, h, w] == 0: output_layers[f, h, w] = 0

        return output_layers

    def backward_prop(self, errors, learning_rate):
        filter_errors = np.zeros(self.filters.shape)

        input_layers = self.input_layers
        _, height, width = errors.shape

        for f in range(self.num_filters):
            delta_bias = np.mean(errors[f, :, :])
            self.biases[f] += learning_rate * delta_bias

        for h in range(height):
            for w in range(width):
                for f in range(self.num_filters):
                    filter_errors[f] += input_layers[:, h:h+self.filter_dim, w:w+self.filter_dim] * errors[f, h, w]

        self.filters -= filter_errors * learning_rate

        return_errors = np.zeros(input_layers.shape)
        channels, _, _ = input_layers.shape

        for h in range(height):
            for w in range(width):
                for f in range(channels):
                    return_errors[f, h, w] = np.sum(input_layers[:, h:h+self.filter_dim, w:w+self.filter_dim]
                                                    * filter_errors[f])
        return return_errors

class PoolLayer:

    def __init__(self, filter_dim):
        self.filter_dim = filter_dim
        self.input_layers = None

    def forward_prop(self, input_layers):
        channels, height, width = input_layers.shape
        out_height = height // self.filter_dim
        out_width = width // self.filter_dim
        self.input_layers = input_layers

        output_layers = np.zeros((channels, out_height, out_width))

        for h in range(out_height):
            for w in range(out_width):
                output_layers[:, h, w] = np.amax(input_layers[:, h:h + self.filter_dim, w:w + self.filter_dim], axis=(1, 2))

        return output_layers

    def backward_prop(self, errors):
        input_layers = self.input_layers
        channels, height, width = input_layers.shape
        out_height = height // self.filter_dim
        out_width = width // self.filter_dim

        pool_errors = np.zeros(input_layers.shape)

        for h in range(out_height):
            for w in range(out_width):
                patch = input_layers[:, h*self.filter_dim:h*self.filter_dim + self.filter_dim,
                                  w*self.filter_dim:w*self.filter_dim + self.filter_dim]
                max_val = np.amax(patch, axis=(0, 1, 2))
                patch_channels, patch_height, patch_width = patch.shape

                for eh in range(patch_height):
                    for ew in range(patch_width):
                        for f in range(patch_channels):
                            if patch[f, eh, ew] == max_val:
                                pool_errors[f, h * self.filter_dim + eh, w * self.filter_dim + ew] = errors[f, h, w]

        return pool_errors

class FullyConnectedLayer:
    def __init__(self, number_of_nodes, input_size):
        self.number_of_nodes = number_of_nodes
        self.weights = np.random.normal(0.0, pow(self.number_of_nodes, -0.5),
                                        (self.number_of_nodes, input_size))
        self.flat_inputs = None
        self.outputs = None
        self.orig_input_shape = None

    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, inputs):
        self.orig_input_shape = inputs.shape
        self.flat_inputs = inputs.flatten()
        outputs = self.activation_function(np.dot(self.weights, self.flat_inputs))
        self.outputs = outputs
        return outputs

    def backward_prop(self, errors, learning_rate):
        inputs = self.flat_inputs
        outputs = self.outputs

        return_errors = np.dot(self.weights.T, errors)

        outputs.shape = (outputs.shape[0], 1)
        errors.shape = (errors.shape[0], 1)
        inputs.shape = (inputs.shape[0], 1)

        self.weights += learning_rate * np.dot((errors * outputs * (1.0 - outputs)), np.transpose(inputs))
        return return_errors.reshape(self.orig_input_shape)