#
# This script performs a logarithmic operations for real numbers in single
# precision.
#

# imports
import tensorflow as tf
import os
import sys

# first matrix in float32
a = tf.constant(15, dtype=tf.float32)

# casting from flot32 to int32 to bfloat16
tf.cast(a, tf.int32)
tf.cast(a, tf.bfloat16)

# logarithmic operation
c = tf.log(a)

# print and validate the resutl
ses = tf.compat.v1.Session()
print(ses.run(c))
val = ses.run(c)

if val.all():
    sys.exit(0)
else:
    sys.exit(1)
