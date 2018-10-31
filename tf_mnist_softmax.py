# coding:utf-8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

file = '/Users/apple/MNIST_data/'
mnist = input_data.read_data_sets(file, one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x, W) + b)
loss = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

for i in range(2000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    if i % 200 == 0:
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1]})
        print "step %d,train_accuracy= %g"%(i,train_accuracy)
	# if train_accuracy > 0.95:
	#     saver.save(sess, "/Users/apple/MNIST_data/model.model.ckpt")
     #        break
    train_step.run(feed_dict={x:batch[0],y_:batch[1]})
