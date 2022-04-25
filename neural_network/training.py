"Training"

class NeuralTraining():
    "Base class for neural network training"

    def get_fitness():
        raise NotImplementedError()

class NeuroEvolution(NeuralTraining):
    "Base class for neural network training using NeuroEvolution"
    
    def __init__(self, base_network, population, generations, mutation_rate):
        pass
