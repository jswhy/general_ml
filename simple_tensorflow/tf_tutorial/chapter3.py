import tensorflow as tf
from numpy.random import RandomState
# tf.enable_eager_execution()

#3.1
# g1 = tf.Graph()
# with g1.as_default():
#     v = tf.get_variable('v', shape = [1], initializer=tf.zeros_initializer())
#
# g2 = tf.Graph()
# with g2.as_default():
#     v = tf.get_variable('v',shape = [1], initializer=tf.ones_initializer())
# with tf.Session(graph = g1) as sess:
#     tf.global_variables_initializer().run()
#     with tf.variable_scope("",reuse = True):
#         print(sess.run(tf.get_variable("v")))
# with tf.Session(graph = g2) as sess:
#     tf.global_variables_initializer().run()
#     with tf.variable_scope("",reuse = True):
#         print(sess.run(tf.get_variable("v")))

#3.4

# w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
# w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))
# x = tf.constant([[0.7, 0.9]])
# a = tf.matmul(x, w1)
# y = tf.matmul(a, w2)
#
# sess= tf.Session()
# sess.run(w1.initializer)
# sess.run(w2.initializer)
# print(sess.run(y))
# sess.close()

batch_size = 8
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
x = tf.placeholder(tf.float32, shape = (None, 2), name = 'x_input')
y_ = tf.placeholder(tf.float32, shape = (None, 1), name = 'y_input')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) +
                                (1 - y_) * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)




rdm = RandomState(1)
X = rdm.rand(128, 2)
Y = [[int(x1 + x2 < 1) ]for (x1, x2) in X]
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2), "\n")

    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % 128
        end = start + batch_size
        sess.run([train_step, y, y_], feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
            print("after %d train step(s), cross entropy on all data is %g" % (i,total_cross_entropy))


    print(sess.run(w1))
    print(sess.run(w2))