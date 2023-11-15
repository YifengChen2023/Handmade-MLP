import random
import numpy as np
import mnist_loader
import math
import matplotlib.pyplot as plt
class Network:

    def __init__(self, params, activate, activate_derivative):
        self.layer_num = len(params)
        self.params = params
        self.activate = activate
        self.activate_derivative = activate_derivative
        self.biases = [np.random.uniform(low=-math.sqrt(12 / (self.params[y - 1] + self.params[y])),
                                         high=math.sqrt(12 / (self.params[y - 1] + self.params[y])),
                                         size=(self.params[y], 1)) for y in range(1, self.layer_num)]
        self.weights = [np.random.uniform(low=-math.sqrt(12 / (self.params[y] + self.params[y - 1])),
                                          high=math.sqrt(12 / (self.params[y] + self.params[y - 1])),
                                          size=(self.params[y], self.params[y - 1])) for y in range(1, self.layer_num)]


    def feedforward(self, z):
        for b, w in zip(self.biases, self.weights):
            z = self.activate(np.dot(w, z) + b)
        return z

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activate(z)
            activations.append(activation)

        # backprop
        delta = (activations[-1] - y) * self.activate_derivative(zs[-1])
        loss = np.vdot((activations[-1]-y).transpose(), activations[-1]-y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for i in range(2, self.layer_num):
            z = zs[-i]
            delta = np.dot(self.weights[-i+1].transpose(), delta) * self.activate_derivative(z)
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
        return nabla_b, nabla_w, loss

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def update(self, mini_batch, lr):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        loss = 0
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w, loss_ = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            loss += loss_
        self.weights = [w - (lr / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (lr / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
        return loss

    def train(self, training_data, epochs, mini_batch_size, lr, test_data=None):
        test_data = list(test_data)
        training_data = list(training_data)
        if test_data:
            n_test = len(test_data)
        n =len(training_data)
        for i in range(epochs):
            loss = 0
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                loss += self.update(mini_batch, lr)
            if test_data:
                print("Epoch {0}: {1} / {2} , lr = {3}, loss = {4}".format(
                    i, self.evaluate(test_data), n_test, lr, loss))
                Loss.append(loss)
                Epoch.append(i)
            else:
                print("Epoch {0} complete".format(i))

def relu(z):
    return np.maximum(0,z)

def relu_prime(z):
    z[z > 0] = 1
    z[z <= 0] = 0
    return z

Loss = []
Epoch = []
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network([784, 30, 10], activate=relu, activate_derivative=relu_prime)
net.train(training_data=training_data, epochs=30, mini_batch_size=10, lr=0.1, test_data=test_data)
plt.title('Result Analysis')
plt.plot(Epoch, Loss,  color='skyblue', label='relu')
plt.legend()
plt.show()