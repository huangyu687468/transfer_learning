# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:58:50 2019

@author: Administrator
"""
import os
import numpy as np
import tensorflow as tf
import csv

from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

data_dir = 'flower_photos/'
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]


# 首先设置计算batch的值，如果运算平台的内存越大，这个值可以设置得越高
batch_size = 10
# 用codes_list来存储特征值
codes_list = []
# 用labels来存储花的类别
labels = []
# batch数组用来临时存储图片数据
batch = []

codes = None

with tf.Session() as sess:
    # 构建VGG16模型对象
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.name_scope("content_vgg"):
        # 载入VGG16模型
        vgg.build(input_)
    
    # 对每个不同种类的花分别用VGG16计算特征值
    for each in classes:
        print("Starting {} images".format(each))
        class_path = data_dir + each
        files = os.listdir(class_path)
        for ii, file in enumerate(files, 1):
            # 载入图片并放入batch数组中
            img = utils.load_image(os.path.join(class_path, file))
            batch.append(img.reshape((1, 224, 224, 3)))
            labels.append(each)
            
            # 如果图片数量到了batch_size则开始具体的运算
            if ii % batch_size == 0 or ii == len(files):
                images = np.concatenate(batch)

                feed_dict = {input_: images}
                # 计算特征值
                codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)
                
                # 将结果放入到codes数组中
                if codes is None:
                    codes = codes_batch
                else:
                    codes = np.concatenate((codes, codes_batch))
                
                # 清空数组准备下一个batch的计算
                batch = []
                print('{} images processed'.format(ii))


# read codes and labels from file
with open('codes', 'w') as f:
    codes.tofile(f)
    
import csv
with open('labels', 'w') as f:
    writer = csv.writer(f, delimiter='\n')
    writer.writerow(labels)
