from NeuralNetwork.MlpBackpropagation import MlpBackpropagation

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

image_flatten_vector_length = 784 # the i. Came from image's dimension: 28 x 28 which equals to 784
num_possible_outcome = 10 # the j. Total number of images to recognise: number 0, 1, 2, 3, 4, 5, 6, 7, 8, & 9.
learning_rate = 0.5
epochs = 1000 # how many times training loop process will be taken
batch_size = 100 

test_inputs = mnist.test.images
test_labels = mnist.test.labels

def fn_batches_xs_ys():
    """ A callback which feeds training dataset during each of training loop phase """
    return mnist.train.next_batch(batch_size)

nn_backprop = MlpBackpropagation(image_flatten_vector_length, num_possible_outcome, learning_rate)

nn_backprop.do_training(fn_batches_xs_ys, epochs)

accuracy = nn_backprop.get_prediction_accuracy(test_inputs, test_labels)

print("Prediction accuracy: ", accuracy, " %")