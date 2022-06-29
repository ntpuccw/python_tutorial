# pip install --user tensorflow 
# need to install some cuda libraries
import tensorflow as tf

# create TensorFlow variables
const = tf.Variable(2.0, name="const")
b = tf.Variable(2.0, name='b')
c = tf.Variable(1.0, name='c')