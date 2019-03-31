import network
import mnist_loader
import os.path


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data, 8, 10, test_data=test_data)
