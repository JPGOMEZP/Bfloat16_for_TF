#
# This script performs a matrix-matrix dot product for real numbers in single
# precision.
#

# imports
import tensorflow as tf
import os
import sys

# first matrix in float32
a = tf.constant([1, 2, 3], dtype=tf.float32)

# casting from flot32 to int32 to bfloat16
tf.cast(a, tf.int32)
tf.cast(a, tf.bfloat16)

# second matrix in float32
b = tf.constant([[11, 12, 13], [14, 15, 16], [17, 18, 19]], dtype=tf.float32)

# casting from flot32 to int32 to bfloat16
tf.cast(b, tf.int32)
tf.cast(b, tf.bfloat16)

# matrix opoeration
c = tf.tensordot(b, a, 1)

# print and validate the resutl
ses = tf.compat.v1.Session()
print(ses.run(c))
val = ses.run(c)

if val.all():
    sys.exit(0)
else:
    sys.exit(1)
