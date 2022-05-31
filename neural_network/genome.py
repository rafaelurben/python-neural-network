from .network import NeuralNetwork


class Genome():
    "Genome - must be subclassed"

    def __init__(self, network: NeuralNetwork):
        self.network = network
        self.obj = None

    def setup(self, *args, **kwargs):
        raise NotImplementedError

    def run_evaluation(self, generation: int = None):
        raise NotImplementedError

    def feed_forward(self, data):
        return self.network.feed_forward(data)

    @property
    def score(self):
        if hasattr(self.obj, "score"):
            if callable(self.obj.score):
                return self.obj.score()
            return self.obj.score
        raise NotImplementedError
