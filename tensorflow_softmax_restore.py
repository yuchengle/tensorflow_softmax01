# coding=utf-8
import numpy as np
import tensorflow as tf
import json

def getresult(file):
    #测试样本读入
    inputfile=open(file, 'r')
    data1 = inputfile.readlines()
    imei = []
    test_xs = []
    for data11 in data1:
        dt = data11.strip().split(',')
        imei.append(dt[0])
        test_xs.append(dt[1:])
    inputfile.close()
    #数组类型转换
    test_xs = np.array(test_xs)
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('../softmaxmodel/softmaxmodel.meta')
        saver.restore(sess,tf.train.latest_checkpoint('../softmaxmodel/'))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        W = graph.get_tensor_by_name("W:0")
        b = graph.get_tensor_by_name("b:0")
        y_model = tf.nn.softmax(tf.matmul(X, W) + b)
        pred_class_index=tf.argmax(y_model, 1)
        feed_dict={X:test_xs}
        pred_value = sess.run(pred_class_index, feed_dict)
        pred_prob = sess.run(y_model, feed_dict)
        jsonList = []
        for i in range(len(pred_value)):
            pred_prob_list = pred_prob[i]
            confidence = max(pred_prob_list)
            Item = {}
            Item["id"] = imei[i]
            Item["pred_value"] = float(pred_value[i])
            Item["pred_prob"] = float(confidence)
            jsonList.append(Item)
        jsonArr = json.dumps(jsonList, sort_keys=True, ensure_ascii=False)
        return jsonArr
'''    
if __name__ == '__main__':
    file='test0.txt'
    print(getresult(file))
'''
    