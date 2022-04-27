from neural_network import NeuralNetwork

net = NeuralNetwork([6, 4, 2])
print(net.feed_forward([0.5] * 6))
print(net.to_json())
net.mutate(0.35)
print(net.to_json())
net.to_json_file("test.json")
