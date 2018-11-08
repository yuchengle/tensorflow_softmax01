#coding:utf-8

import numpy as np
import tensorflow as tf
import os.path  
    
def softmaxmain(model_version,inputfile1,inputfile2,learning_rate,training_epochs,num_labels,batch_size):    
    #训练好的模型分类器保存位置
    MODEL_DIR = "../softmaxmodel"
    if not tf.gfile.Exists(MODEL_DIR):  
        tf.gfile.MakeDirs(MODEL_DIR)  
    MODEL_NAME = "softmaxmodel" 
    output_path = os.path.join(tf.compat.as_bytes(MODEL_DIR),tf.compat.as_bytes(str(model_version)))
    builder = tf.saved_model.builder.SavedModelBuilder(output_path)
    #训练样本读入
    data1 = inputfile1.readlines()
    xs = []
    labels = []
    for data11 in data1:
        dt1 = data11.strip().split(',')
        p = int(len(dt1) - num_labels)
        xs.append(dt1[1:p])
        labels.append(dt1[p:])
    #测试样本读入
    data2 = inputfile2.readlines()
    test_xs = []
    test_labels = []
    for data21 in data2:
        dt2 = data21.strip().split(',')
        q = int(len(dt2) - num_labels)
        test_xs.append(dt2[1:q])
        test_labels.append(dt2[q:])
    #数组类型转换
    xs = np.array(xs)
    labels = np.array(labels)
    test_xs = np.array(test_xs)
    test_labels = np.array(test_labels)
    #打乱训练样本顺序
    arr = np.arange(xs.shape[0])
    np.random.shuffle(arr) 
    xs = xs[arr, :] 
    labels = labels[arr, :] 
    #softmax参数设置
    train_size, num_features = xs.shape 

    print('~~~~~~~~~~开始设计计算图~~~~~~~~')
    X = tf.placeholder("float", shape=[None, num_features], name = 'X')
    Y = tf.placeholder("float", shape=[None, num_labels], name = 'Y') 
    W = tf.Variable(tf.zeros([num_features, num_labels]), name = 'W')  
    b = tf.Variable(tf.zeros([num_labels]), name = 'b')
    # 函数y=w*x+b
    y_model = tf.nn.softmax(tf.matmul(X, W) + b)
    cost = -tf.reduce_sum(Y * tf.log(y_model)) 
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 
    correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 
    print('把计算图写入事件文件，在TensorBoard里面查看')
    writer = tf.summary.FileWriter(logdir='../tensorboard_softmax', graph=tf.get_default_graph())
    writer.close()
    print('~~~~~~~~~~开始运行计算图~~~~~~~~') 
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for step in range(training_epochs * train_size // batch_size):
            offset = (step * batch_size) % train_size
            batch_xs = xs[offset:(offset + batch_size), :] 
            batch_labels = labels[offset:(offset + batch_size)] 
            err, _ = sess.run([cost, train_op], feed_dict={X:batch_xs, Y:batch_labels}) 
            #print (step, err) 
        print("accuracy", accuracy.eval(feed_dict={X:test_xs, Y:test_labels}))   
        saver.save(sess, os.path.join(MODEL_DIR, MODEL_NAME)) 
        builder.save()
    inputfile1.close()
    inputfile2.close()

if __name__ == '__main__':
    model_version=1#模型版本
    inputfile1=open('train.txt', 'r')#训练样本
    inputfile2=open('test.txt', 'r')#测试样本
    learning_rate = 0.01 #学习率
    training_epochs = 1000 #训练次数
    num_labels = 3 #分类数
    batch_size = 100 #每批训练的数据量大小    
    softmaxmain(model_version,inputfile1,inputfile2,learning_rate,training_epochs,num_labels,batch_size)
