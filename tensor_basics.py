import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # random succesfully opened messages no longer seen
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(physical_devices[0], True)



# initialization of tensors
x = tf.constant(4.0, shape=(1, 1), dtype=tf.float32)
x = tf.constant([[1, 2, 3], [4, 5, 6]])  # 2Dtensor
x = tf.ones((3, 3))
x=tf.zeros((2, 3))
x=tf.eye(3) # I for identity matrix
x = tf.random.normal((3, 3), mean=0, stddev=1) # normal distribution
x = tf.random.uniform((1, 3), minval=0, maxval=1) # uniform distribution
x = tf.range(9)
x = tf.range(start=1, limit=10, delta=2)
x = tf.cast(x, dtype=tf.float64)



# mathematical operations
x = tf.constant([1,2,3])
y = tf.constant([9,8,7])

z = tf.add(x, y) # element wise addition,same as z = x + y
# tf.subtract(x,y) i.e x-y
# tf.divide(x,y) i.e x/y
# z =  tf.multiply(x,y) i.e x*y elementwise

z = tf.tensordot(x,y, axes=1) #dot product
z = tf.reduce_sum(x*y, axis=0) # same as above

z = x ** 5 # element wise exponentiation

x = tf.random.normal((2,3))
y = tf.random.normal((3,4))
z = tf.matmul(x, y) # i.e z = x@y
# print(z)

# Indexing
x = tf.constant([0,1, 1, 2, 3, 1, 2, 3])
#print(x[:])
#print(x[::2]) # skip every second element
#print(x[::-1]) # reverse order

indices = tf.constant([0,3])
x_ind = tf.gather(x, indices) # get specific indices from x


x = tf.constant([[1, 2],
                 [2, 3],
                 [4, 5]])

# print(x[0, :])
# print(x[0:2, :])


# reshaping
x = tf.range(9)
print(x)

x = tf.reshape(x, (3,3))
print(x)

x = tf.transpose(x, perm=[1, 0])
print(x)