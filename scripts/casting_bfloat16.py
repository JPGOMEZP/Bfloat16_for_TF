#
# This script performs a matrix-matrix addition for real numbers in single
# precision.
#

# imports

import tensorflow as tf
import os
import sys

#Casting x to int32 ans bfloat16
x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
tf.cast(x, tf.int32)
tf.cast(x, tf.bfloat16)

#Casting y to int32 ans bfloat16
y = tf.constant([[7, 8, 9], [10, 11, 12]], dtype=tf.float32)
tf.cast(x, tf.int32)
tf.cast(x, tf.bfloat16)
