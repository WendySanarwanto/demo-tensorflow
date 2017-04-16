import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# A helper for initialising required models to train the NN and also NN's inputs, outputs, & hidden layers model
def initialise_nn_models():
    # 1. Define NN input layer (x) which accepts n dimensional vector and the dimension can be of any length (None).
    #    n here is i, or 28 x 28 = 784
    x = tf.placeholder(tf.float32, [None, image_flatten_vector_length ])

    # 2. Define Weight params vector (NN Hidden layer) & Bias parameters of the linear regression model, by using tf.Variable.
    # The linear regression model is defined as in this following equation: sum(Wij * xj + bi)
    # where i = length of flatten vector of 28x28 image and j = number of possible outcome that NN should identify/classify
    # W is the weight parameters, represents the neurons in the Hidden Layer & b is the bias parameter. 
    W = tf.Variable(tf.zeros([image_flatten_vector_length, num_possible_outcome]), name="Weights")
    b = tf.Variable(tf.zeros(num_possible_outcome), name="Biases")

    # 3. Define linear model of Wij * xj + bi 
    linear_model = tf.matmul(x, W) + b

    # 4. We use softmax regression model so that we could obtain matched result's probabilities during evaluation later.
    # softmax is defined as in this following equation: y = softmax(Wij * xj + bi)
    y = tf.nn.softmax(linear_model)

    # 5. Define y', the NN output layer vector
    y_ = tf.placeholder(tf.float32, [None, num_possible_outcome])

    # 6. Define loss model, the 'cross-entropy': E =  mean( -sum(y_ * log(y)) ) where E is a scalar
    # 'Loss' represent how far off our model is from our desired outcome (error). 
    # The more it reach near to 0, the more accurate our NN would give expected outcome
    cross_entrophy = tf.reduce_mean(-tf.reduce_sum( y_ * tf.log(y), reduction_indices=[1]))

    # 7. Define Training model for the Wij & bi ( the weight & bias vectors) using backpropagation - gradient descent algorithm and TF already have these, so we'll just use them.
    training_model = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entrophy)

    return training_model, x, y, y_

# A helper for training 
def do_training_nn(training_model, x, y):
    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    for _ in range(epochs):
        # Pick batches of inputs & outputs data from the pre-loaded MNIST data
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Run NN Training using the training model
        session.run(training_model, feed_dict = {x: batch_xs, y_: batch_ys})    
    return session

# A helper to measure trained NN's prediction accuracy
def measure_nn_prediction_accuracy(y, y_):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    

# Constants
image_flatten_vector_length = 784 # the i. Came from image's dimension: 28 x 28 which equals to 784
num_possible_outcome = 10 # the j. Total number of images to recognise: number 0, 1, 2, 3, 4, 5, 6, 7, 8, & 9.
learning_rate = 0.5
epochs = 1000 # how many times training loop process will be taken
batch_size = 100 

# Create NN models 
training_model, x, y, y_ = initialise_nn_models()

# Train the NN using inputs & outputs data sets
session = do_training_nn(training_model, x, y)

# Get the trained NN's prediction accuracy
accuracy = measure_nn_prediction_accuracy(y, y_)

print("Prediction accuracy: ", ( session.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100 ), "%"  )

# TODO: Save the model to disk

# TODO: Load model from the disk and use it for classifying handwriting images of 0-9 numbers.