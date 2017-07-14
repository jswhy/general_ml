#coding=utf-8
import tensorflow as tf
from numpy.random import RandomState

#设置batch大小为8
batch_size = 9

# 初始化数据，和点
x=tf.placeholder(name="x-iunput",dtype=tf.float32,shape=(None,2))
y_=tf.placeholder(name="y-iunput",dtype=tf.float32,shape=(None,1))
w1 = tf.Variable(tf.random_normal([2,3],seed=1,stddev=1))
w2 = tf.Variable(tf.random_normal([3,1],seed=1,stddev=1))



#定义神经网络流动方向
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#代价函数
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)



#随机产生一个训练集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
Y = [[int(x1+x2<1)] for (x1,x2) in X]# 如果x1+x2<1即为1，反之为0



with tf.Session() as sess:
# 定义会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    STEPS = 5000 #设定训练轮数
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
    sess.run(train_step,feed_dict={x : X[start:end],y:Y[start:end]})
    if i%100 ==0:
        total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
        print("After %d training steps ,cross entropy on all data is %g" % (i,total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))