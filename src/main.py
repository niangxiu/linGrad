import network
import mnist_loader
import os.path
import data_generator


# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = network.Network([784, 30, 10])
# net.SGD(training_data, 8, 10, test_data=test_data)


net_sizes = [50, 50, 50, 50]
data_sizes = [50000, 1, 10000]
training_data, validation_data, test_data = data_generator.load_data(net_sizes,data_sizes)
net = network.Network(net_sizes)
# net.SGD(training_data, epochs=50, mini_batch_size=10, test_data=test_data, case='DIST', const_eta=None)
net.SGD(training_data, epochs=50, mini_batch_size=10, test_data=test_data, case='DIST', const_eta=1)
