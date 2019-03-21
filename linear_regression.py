import tensorflow as tf

# y = Wx + b

# training input data
x_train = [1.0, 2.0, 3.0, 4.0]

# The correct answers for the input values which will be used by the Model to evaluate the value of W and b
y_train = [-1.0, -2.0, -3.0, -4.0]  # expected output.

# Model will try to guess the values of W and b for each input in x_train to match the corresponding value in y_train
# These are variables because the model will change the values of W and b
W = tf.Variable(initial_value=[1.0], dtype=tf.float32)
b = tf.Variable(initial_value=[1.0], dtype=tf.float32)

# this is going to be the input for the model
x = tf.placeholder(dtype=tf.float32)

# This is the Node which will tell the model what is the correct output for each input
y_actual = tf.placeholder(dtype=tf.float32)

# this is the formula y = Wx + b
multiply = tf.multiply(x=W, y=x)
y_output = tf.add(x=multiply, y=b)

# Loss function and optimiser which aim to minimise the difference between expected output and actual output
loss = tf.reduce_sum(input_tensor=tf.square(x=y_output - y_actual))
optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = optimiser.minimize(loss=loss)

# Sessions are used to evaluate the tensor value of a node or nodes
session = tf.Session()
session.run(tf.global_variables_initializer())

# Total loss function before training. This will be huge because the initial values of the variables are far off from
# the actual values.
print(session.run(fetches=loss, feed_dict={x: x_train, y_actual: y_train}))

# Training the model, this will run the train step 1000 times.
for _ in range(1000):
    session.run(fetches=train_step, feed_dict={x: x_train, y_actual: y_train})

# Prints the calculated value of loss, W and b after training
print(session.run(fetches=[loss, W, b], feed_dict={x: x_train, y_actual: y_train}))

# Test the model with some test data
print(session.run(fetches=y_output, feed_dict={x: [5.0, 10.0, 15.0]}))
