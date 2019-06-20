#import tensorflow as tf
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from model import *
import numpy as np
def train(batchsize,epochs):
    (X_train,Y_train),(X_test,Y_test)=cifar10.load_data()
    Y_train,Y_test=to_categorical(Y_train,10),to_categorical(Y_test,10)
    nums=int(X_train.shape[0]/batchsize)
    input=tf.placeholder(tf.float32,shape=[batchsize,X_train.shape[1],X_train.shape[2],X_train.shape[3]])
    labels=tf.placeholder(tf.float32,shape=[batchsize,10])
    logits,output=build(input,num_layer=18,filters=64,repetitions=[2,2,2,2],n_classes=10,is_train=True)
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    optimizer=tf.train.AdamOptimizer(0.01).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(output, 1),tf.argmax(labels, 1))
    accuaracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(epochs):
            for n in range(nums):
                predict1,logits1, _, loss1,acc = sess.run([output, logits,optimizer, loss,accuaracy],
                                                   feed_dict={input: X_train[n * batchsize:(n + 1) * batchsize],labels: Y_train[n * batchsize:(n + 1) *batchsize]})
                print('acc:',acc)
            predict1, logits1, _, loss1, acc = sess.run([output, logits, optimizer, loss, accuaracy],
                                                        feed_dict={input: X_test[0: batchsize],
                                                                   labels: Y_test[0: batchsize]})
            #acc = sess.run(accuaracy,feed_dict={input: X_train[:batchsize], labels: Y_train[:batchsize]})
            print('step', step, acc)
if __name__ == '__main__':
    batchsize=32
    train_epochs=300
    train(batchsize,train_epochs)
