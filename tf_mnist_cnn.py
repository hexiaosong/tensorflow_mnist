#coding:utf-8

import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


"""
权重初始化
初始化为一个接近0的很小的正数
"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1], padding = 'SAME')

start = time.clock()
file = "/Users/apple/MNIST_data/"
mnist = input_data.read_data_sets(file, one_hot=True)

# 第一层卷积
# x_image(batch, 28, 28, 1) -> h_pool1(batch, 14, 14, 32)
x = tf.placeholder(tf.float32,[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1]) #最后一维代表通道数目，如果是rgb则为3


W_conv1 = weight_variable([5, 5, 1, 32])  # 32个5x5的卷积核
b_conv1 = bias_variable([32])             # 32个偏置项
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)           # 2x2的池化

# 第二层卷积
# h_pool1(batch, 14, 14, 32) -> h_pool2(batch, 7, 7, 64)
W_conv2 = weight_variable([5, 5, 32, 64]) # 64个5x5的卷积
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第三层全连接(Flatten)
# h_pool2(batch, 7, 7, 64) -> h_fc1(1, 1024)
W_fc1 = weight_variable([7 * 7 * 64, 1024]) # 第二层的卷积核为64，经过两次池化后尺寸 28==>14==>7
b_fc1 = bias_variable([1024])  # 扁平化为1024的一维向量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第四层Softmax输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 训练和评估模型
# ADAM优化器来做梯度最速下降
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 计算准确率
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))


# 构建会话
sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(5000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0})
        print("step %d, train_accuracy %g" % (i, train_accuracy))

    train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1],keep_prob:0.5})

print("test accuracy %g" %accuracy.eval(session = sess,feed_dict = {x:mnist.test.images, y_:mnist.test.labels,keep_prob:1.0}))

end = time.clock()
print("running time is %g s") % (end-start)
