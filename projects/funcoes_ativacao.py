import numpy as np

def step_function(soma):
    if soma >= 1:
        return 1
    return 0

def sigmoid_function(soma):
    return 1 / (1 + np.exp(-soma))

def tangent_function(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

def relu_function(soma):
    if soma >= 0:
        return soma
    return 0

def linear_function(soma):
    return soma


def softmax_function(x):
    ex = np.exp(x)
    return ex/ex.sum()

teste = step_function(-1)
print(teste)
teste = sigmoid_function(0.358)
print(teste)
teste = tangent_function(0.358)
print(teste)
teste = relu_function(0.358)
print(teste)
teste = linear_function(-0.358)
print(teste)

valores = [7.0,2.0,1.3]
print(softmax_function(valores))