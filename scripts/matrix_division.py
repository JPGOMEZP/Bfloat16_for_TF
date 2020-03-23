#
# This script performs a matrix-matrix division for real numbers in single
# precision.
#

# imports
import tensorflow as tf
import os
import sys

# first matrix in float32
a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)

# casting from flot32 to int32 to bfloat16
tf.cast(a, tf.int32)
tf.cast(a, tf.bfloat16)

# second matrix in float32
b = tf.constant([[7, 8, 9], [10, 11, 12]], dtype=tf.float32)

# casting from flot32 to int32 to bfloat16
tf.cast(b, tf.int32)
tf.cast(b, tf.bfloat16)

# matrix operation
c = tf.divide(b, a)

# print and validate the resutl
ses = tf.compat.v1.Session()
print(ses.run(c))
val = ses.run(c)

if val.all():
    sys.exit(0)
else:
    sys.exit(1)
