from neural_network import NeuralNetwork

net = NeuralNetwork([6, 7, 4, 2])
net.to_json_file("test.json")
print(net.feed_forward([0.5] * 6))
print(net.to_json())
net.mutate(0.35)
print(net.to_json())
