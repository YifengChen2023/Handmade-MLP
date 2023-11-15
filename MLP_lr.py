import math
import random

import matplotlib.pyplot as plt
import numpy as np
import mnist_loader

class Network:

    def __init__(self, params, activate, activate_derivative):
        np.random.seed(10)
        self.layer_num = len(params)
        self.params = params
        self.activate = activate
        self. activate_derivative = activate_derivative
        self.biases = [np.random.randn(y, 1) for y in params[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(params[:-1], params[1:])]
        self.nabla_Loss = 0
        self.g = 0
        self.epsilon = 1
        self.gamma = 0.5
        self.lr = 3

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

    def update(self, mini_batch, lr, type):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        loss = 0
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w, loss_ = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            loss += loss_
        if type == 'linear':
            if lr > 0.01:
                lr /= 1.2
        elif type == 'Adagrad':
            inner = 0
            for a in nabla_w:
                inner += np.sum(a**2)
            for a in nabla_b:
                inner += np.sum(a**2)
            self.nabla_Loss += inner
            lr = 1/(math.sqrt(self.nabla_Loss)+self.epsilon) * self.lr

        elif type == 'RMSprop':
            inner = 0
            for a in nabla_w:
                inner += np.sum(a ** 2)
            for a in nabla_b:
                inner += np.sum(a ** 2)
            self.g = self.gamma * self.g + \
                     (1-self.gamma)*(inner)
            print('inner=',inner)
            lr = 1/(math.sqrt(self.g)+self.epsilon) * self.lr
            print('lr=', lr)
        self.weights = [w - (lr / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (lr / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
        return loss

    def train(self, training_data, epochs, mini_batch_size, lr, test_data=None, type='linear'):
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
                loss += self.update(mini_batch, lr, type)
            if test_data:
                print("Epoch {0}: {1} / {2} , loss = {3}".format(
                    i, self.evaluate(test_data), n_test, loss))
                if type == 'linear':
                    Loss_l.append(loss)
                elif type == 'Adagrad':
                    Loss_Adagrad.append(loss)
                elif type == 'RMSprop':
                    Loss_RMSprop.append(loss)
                Epoch.append(i)
            else:
                print("Epoch {0} complete".format(i))



def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


Loss_l = []
Loss_Adagrad = []
Loss_RMSprop = []
Epoch = []
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net1 = Network([784, 30, 10], activate=sigmoid, activate_derivative=sigmoid_prime)
# net1.train(training_data=training_data, epochs=30, mini_batch_size=10, lr=0.3, test_data=test_data, type='linear')
# net2 = Network([784, 30, 10], activate=sigmoid, activate_derivative=sigmoid_prime)
# net2.train(training_data=training_data, epochs=30, mini_batch_size=10, lr=0.3, test_data=test_data, type='Adagrad')
net3 = Network([784, 30, 10], activate=sigmoid, activate_derivative=sigmoid_prime)
net3.train(training_data=training_data, epochs=30, mini_batch_size=10, lr=0.3, test_data=test_data, type='RMSprop')


plt.title('Result Analysis')
# plt.plot(Epoch, Loss_l, color='green', label='Linear')
# plt.plot(Epoch, Loss_Adagrad, color='red', label='Adagrad')
plt.plot(Epoch, Loss_RMSprop,  color='skyblue', label='RMSprop')
plt.legend() # 显示图例
plt.show()
