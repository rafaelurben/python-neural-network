"Activation functions"

import math

def identity(t):
    "Identity function"
    return t

def binary_step(t):
    "Binary step function"
    return 1 if t >= 0 else 0

def relu(t):
    "Rectified linear unit"
    return max(0, t)

def leaky_relu(t):
    "Leaky rectified linear unit"
    return 0.01*t if t < 0 else t

def sigmoid(t):
    "Sigmoid function"
    return 1/(1+math.exp(-t))

ACFUNCS = {
    "identity": identity,
    "binary_step": binary_step,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "sigmoid": sigmoid,
}
