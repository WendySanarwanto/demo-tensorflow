"""
Getting started sample of TensorFlow, picked from this following url: https://www.tensorflow.org/get_started/get_started
"""

import tensorflow as tf
import numpy as np

def do_train(train_loop, train_op, x_inputs, y_inputs):
    """
    Train the x_inputs to match y_inputs, by predefined train_op model.
    Args:
        train_loop (int): define number of training loop (e.g. 1000 )
        train_op (obj): tf's train operation object
        x_inputs: training data input
        y_inputs: training data input to match against x_inputs
    """
    # Open Tf session & initialise all defined global variables (W & b)    
    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    # begin training the model
    for i in range(train_loop):
        session.run(train_op, {x: x_inputs, y: y_inputs})

    # Display training's results
    current_w, current_b, current_loss = session.run([W, b, loss], {x: x_inputs, y: y_inputs})
    print("W:%s b:%s loss:%s"%(current_w, current_b, current_loss))

#constants
init_w = .3
init_b = -.3
gradient_descent = 0.01
train_loop = 1000

"""
We have this sample learning model: W * x + b,  where x is an array of inputs to train, W & b are model parameters to tweak later.
"""
# Define the parameters
W = tf.Variable([init_w], tf.float32)
b = tf.Variable([init_b], tf.float32)

# Define Input & output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Define the model's equation
linear_model = W * x + b

"""
Loss Operation: operation which sums the squares result of linear_model against y inputs, into a reduced scalar value.
The scalar value indicates accuracy rate of our NN when giving results (indicated by y_inputes), from taking the x_inputs. 
The more lower it is near to 0, the more accurate the actual y results would be, guessed by NN later.
"""
# Define loss operation
loss = tf.reduce_sum(tf.square(linear_model - y))

# Define operation for training our x & y input outputs so that the loss result become near to 0, which indicates accuracy rate when inferencing the model later.
optimiser = tf.train.GradientDescentOptimizer(gradient_descent)
train = optimiser.minimize(loss)

# Prepare x & y inputs
x_inputs = [1, 2, 3, 4]
y_inputs = [0, -1, -2, -3]

# Execute the training
do_train(train_loop, train, x_inputs, y_inputs)
