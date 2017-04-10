import tensorflow as tf
import datetime
import numpy as np

def current_timestamp():
    return datetime.datetime.now().replace(microsecond=0)

#1. Define input layer for input array: x (input tensor) and its dimension (input's rank)
input_layers = [tf.contrib.layers.real_valued_column("x", dimension=1)]

#2. Instantiate default linear model W * x + b
linear_model = tf.contrib.learn.LinearRegressor(feature_columns=input_layers)

#3. Setup input training data sets using numpy
x = np.array( [1., 2., 3., 4.] )
y = np.array([0., -1., -2., -3.])
training_steps = 10000
input_datasets = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, batch_size = 4, num_epochs=training_steps)

# #4. Do training against input datasets for a number of training steps
before_train_timestamp = current_timestamp()
linear_model.fit(input_fn=input_datasets, steps=training_steps)
after_train_timestamp = current_timestamp()

print ("I Finished traning the model for: "  + str(after_train_timestamp - before_train_timestamp)) 

#5. When training is done, we evaluate it using training input.
eval = np.array([1., 2., 3., 4.])
evaluation_dataset = tf.contrib.learn.io.numpy_input_fn({"x": eval}, y)
print(linear_model.evaluate(input_fn=evaluation_dataset))

#6. TODO: Save the trained model to somewhere. 

#7. TODO: Have a test load the trained model and re-evaluate it using separate datasets again.

