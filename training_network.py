# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:51:09 2019

@author: Administrator
"""

# 输入数据的维度
inputs_ = tf.placeholder(tf.float32, shape=[None, codes.shape[1]])
# 标签数据的维度
labels_ = tf.placeholder(tf.int64, shape=[None, labels_vecs.shape[1]])

# 加入一个256维的全连接的层
fc = tf.contrib.layers.fully_connected(inputs_, 256)

# 加入一个5维的全连接层
logits = tf.contrib.layers.fully_connected(fc, labels_vecs.shape[1], activation_fn=None)

# 计算cross entropy值
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits)

# 计算损失函数
cost = tf.reduce_mean(cross_entropy)

# 采用用得最广泛的AdamOptimizer优化器
optimizer = tf.train.AdamOptimizer().minimize(cost)

# 得到最后的预测分布
predicted = tf.nn.softmax(logits)

# 计算准确度
correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def get_batches(x, y, n_batches=10):
    """ 这是一个生成器函数，按照n_batches的大小将数据划分了小块 """
    batch_size = len(x)//n_batches
    
    for ii in range(0, n_batches*batch_size, batch_size):
        # 如果不是最后一个batch，那么这个batch中应该有batch_size个数据
        if ii != (n_batches-1)*batch_size:
            X, Y = x[ii: ii+batch_size], y[ii: ii+batch_size] 
        # 否则的话，那剩余的不够batch_size的数据都凑入到一个batch中
        else:
            X, Y = x[ii:], y[ii:]
        # 生成器语法，返回X和Y
        yield X, Y


#######################开始训练

# 运行多少轮次
epochs = 20
# 统计训练效果的频率
iteration = 0
# 保存模型的保存器
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for x, y in get_batches(train_x, train_y):
            feed = {inputs_: x,
                    labels_: y}
            # 训练模型
            loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            print("Epoch: {}/{}".format(e+1, epochs),
                  "Iteration: {}".format(iteration),
                  "Training loss: {:.5f}".format(loss))
            iteration += 1
            
            if iteration % 5 == 0:
                feed = {inputs_: val_x,
                        labels_: val_y}
                val_acc = sess.run(accuracy, feed_dict=feed)
                # 输出用验证机验证训练进度
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Validation Acc: {:.4f}".format(val_acc))
    # 保存模型
    saver.save(sess, "checkpoints/flowers.ckpt")