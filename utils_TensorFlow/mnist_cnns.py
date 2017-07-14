#coding = utf-8
from tensorflow.examples.tutorials.mnist import input_data
# 载入mnist数据
import numpy as np
import tensorflow as tf

# mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
# print("Training dataset size: ", mnist.train.num_examples)#训练集
# print("Validating dataset size: ", mnist.validation.num_examples)#交叉验证集
# print("Test dataset size: ", mnist.test.num_examples)#测试集
# print("Training dataset shape: ", np.shape(mnist.train.images))


INPUT_NODE = 784 #输入层数
OUTPUT_NODE = 10 #输出层数
LAYER1_NODE = 500#第一层隐含层节点数
BATCH_SIZE = 100
LEARNING_RATE_ALPHA = 0.8 #初始化学习率
LEARNING_RATE_DECAY = 0.96 #学习率衰减率
REGULARIZATION_RATE = 0.00001 #正则化系数
TRAINING_STEPS = 3000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99#滑动平均衰减率


def inference(input_tensor,avg_class,weights1,bises1,weights2,bises2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+bises1)
        return tf.matmul(layer1,weights2)+bises2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1)))+avg_class.average(bises1)
        return tf.matmul(input_tensor,avg_class.average(weights2))+avg_class.average(bises2)

def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name = "x-input")
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name = "y-input")

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    bises1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    bises2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))


    y = inference(x,None,weights1,bises1,weights2,bises2)
    global_step = tf.Variable(0,trainable = False)
    variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    variables_averages_op = variables_averages.apply(tf.trainable_variables())
    average_y = inference(x,variables_averages,weights1,bises1,weights2,bises2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1)+regularizer(weights2)
    loss = cross_entropy_mean+regularization
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_ALPHA,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY,staircase=False)


    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op = tf.no_op(name="train")
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    with tf.Session() as sess:
        tf.initialize_all_Variables().run()
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i % 100 == 0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("after %d training step(s), validation accuracy using average model is %g " %(i,validate_acc))

            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y:ys})

        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print("after %d training step(s), test accuracy using average model is %g " %(TRAINING_STEPS,test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    train(mnist)


if __name__ =='__main__':
    tf.app.run()





