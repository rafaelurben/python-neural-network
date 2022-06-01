import random

def randplusminus(maximum=1):
    """Generates a random number between -maximum and maximum."""
    return maximum - (random.random() * 2 * maximum)
