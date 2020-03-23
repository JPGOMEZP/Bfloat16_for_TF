#
# This script performs a matrix-matrix addition for real numbers in single
# precision.
#

# imports
import tensorflow as tf
import os
import sys

# first constant for the addition in float32
a = tf.constant(5.0, dtype=tf.float32)

# casting from flot32 to int32 to bfloat16
tf.cast(a, tf.int32)
tf.cast(a, tf.bfloat16)

# second constant for the addition in float32
b = tf.constant(6.0, dtype=tf.float32)

# casting from flot32 to int32 to bfloat16
tf.cast(b, tf.int32)
tf.cast(b, tf.bfloat16)

# addition
c = tf.add(a, b)

# print and validate the resutl
ses = tf.compat.v1.Session()
print(ses.run(c))
val = ses.run(c)

if val.all():
    sys.exit(0)
else:
    sys.exit(1)
