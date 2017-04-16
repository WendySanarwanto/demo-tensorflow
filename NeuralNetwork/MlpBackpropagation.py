import tensorflow as tf
import numpy as np


class MlpBackpropagation:
    """
        Represents Multi Layer Perceptron NN models and also provide training method which use
        Backpropagation algorithm, provided behind scene by TensorFlow.
    """

    def __init__(self, image_flatten_vector_length, num_possible_outcome, learning_rate):
        """ Constructor of this class """
        self.image_flatten_vector_length = image_flatten_vector_length
        self.num_possible_outcome = num_possible_outcome
        self.learning_rate = learning_rate

        # 1. Define NN input layer (x) which accepts n dimensional vector and the dimension can be of any length (None).
        #    n here is i, or 28 x 28 = 784
        self.x = tf.placeholder(tf.float32, [None, image_flatten_vector_length ])

        # 2. Define Weight params vector (NN Hidden layer) & Bias parameters of the linear regression model, by using tf.Variable.
        # The linear regression model is defined as in this following equation: sum(Wij * xj + bi)
        # where i = length of flatten vector of 28x28 image and j = number of possible outcome that NN should identify/classify
        # W is the weight parameters, represents the neurons in the Hidden Layer & b is the bias parameter. 
        self.W = tf.Variable(tf.zeros([image_flatten_vector_length, num_possible_outcome]))
        self.b = tf.Variable(tf.zeros(num_possible_outcome))

        # 3. Define linear model of Wij * xj + bi 
        self.linear_model = tf.matmul(self.x, self.W) + self.b

        # 4. We use softmax regression model so that we could obtain matched result's probabilities during evaluation later.
        # softmax is defined as in this following equation: y = softmax(Wij * xj + bi)
        self.y = tf.nn.softmax(self.linear_model)

        # 5. Define y', the NN output layer vector
        self.y_ = tf.placeholder(tf.float32, [None, num_possible_outcome])

        # 6. Define loss model, the 'cross-entropy': E =  mean( -sum(y_ * log(y)) ) where E is a scalar
        # 'Loss' represent how far off our model is from our desired outcome (error). 
        # The more it reach near to 0, the more accurate our NN would give expected outcome
        self.cross_entrophy = tf.reduce_mean(-tf.reduce_sum( self.y_ * tf.log(self.y), reduction_indices=[1]))

        # 7. Define Training model for the Wij & bi ( the weight & bias vectors) using backpropagation - gradient descent algorithm and TF already have these, so we'll just use them.
        self.training_model = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cross_entrophy)

    def do_training(self, fn_batches_xs_ys, epochs):
        """ Train the NN by specified data set batches, for a number of steps (epochs) """
        self.session = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        for _ in range(epochs):
            batch_xs, batch_ys = fn_batches_xs_ys()
            # Run NN Training using the training model
            self.session.run(self.training_model, feed_dict = {self.x: batch_xs, self.y_: batch_ys})

    def get_prediction_accuracy(self, test_xs, test_ys):
        """ Get trained NN's prediction accuracy """
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        correct_prediction_mean = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        final_correct_prediction = self.session.run(correct_prediction_mean, feed_dict={self.x: test_xs, self.y_: test_ys}) * 100
        return final_correct_prediction
