"Activation functions"

def identity(x):
    "Identity function"
    return x

def binary_step(x):
    "Binary step function"
    return 1 if x >= 0 else 0

def relu(x):
    "Rectified linear unit"
    return max(0, x)

def leaky_relu(x):
    "Leaky rectified linear unit"
    return 0.01*x if x < 0 else x

ACFUNCS = {
    "identity": identity,
    "binary_step": binary_step,
    "relu": relu,
    "leaky_relu": leaky_relu,
}
