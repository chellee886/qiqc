# import tensorflow as tf
# import numpy as np
from sklearn.metrics import f1_score
#
# a_pl = tf.placeholder(name="a",
#                     shape=[None, 2, 2],
#                     dtype=tf.float32)
#
# b_pl = tf.placeholder(name="b",
#                    shape=[None, 1],
#                    dtype=tf.int32)
#
# aa_pl = tf.placeholder(name="aa",
#                     shape=[None, 4],
#                     dtype=tf.float32)
#
#
# feed_dict = {
#     a_pl: np.array([[[0.3, 0.2],[0.7, 3.]],
#                  [[0.0, 4.],[4., 5.]],
#                  [[5., 6.],[6., 7.]]]),
#     b_pl: np.array([[1],
#                  [0],
#                  [1]]),
#     aa_pl: np.array([[0.,1.,1.,1.],
#                     [0.,1.,1.,1.],
#                     [1.,1.,1.,1.]])
# }
#
#
#
# initializer = tf.contrib.layers.xavier_initializer()
# a_pl = tf.reshape(a_pl, [tf.shape(a_pl)[0], -1])
#
# W = tf.get_variable(name="W",
#                     shape=[4, 1],
#                     initializer=initializer,
#                     dtype=tf.float32)
# b = tf.get_variable(name="b",
#                     shape=[1],
#                     dtype=tf.float32)
#
# condition = tf.less(a_pl, 0.5)
# result_max = tf.where(condition,tf.zeros_like(a_pl), tf.ones_like(a_pl))
# eq = tf.reduce_mean(tf.cast(tf.equal(aa_pl, result_max), tf.float32))
#
# summ = tf.math.segment_sum
#
#
#
# logits = tf.matmul(a_pl, W) + b
# predictions = tf.nn.sigmoid(logits)
# softmax = tf.nn.softmax(a_pl)
# mul = tf.matmul(softmax, tf.transpose(a_pl))
# losses = tf.losses.log_loss(labels=b_pl, predictions=predictions)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     predictions, loss, W, softmax, mul, result_max, eq, bb = sess.run([predictions, losses, W, softmax, mul, result_max, eq, b_pl], feed_dict=feed_dict)
#
#     # predictions, b_pl = sess.run([predictions, b], feed_dict=feed_dict)
#     print(bb)
#     print(f1_score(y_pred=bb, y_true=np.array([[0],
#                  [0],
#                  [0]]), labels=1))
#     print(np.shape(bb))
#     # print(mul)


import numpy as np
import pandas as pd
x = 155 + (300 - 155) * np.random.random(size=[100000, 1])
y = 180 * np.random.random(size=[100000, 1])
b = np.random.random(size=[100000, 1])
result = np.add(np.add(3.027902 * x, 0.668345 * y), b)

all = np.concatenate([x, y, result], axis=1)
idx = []
for i in range(100000):
    if all[i, 2] > 500 and all[i, 2] < 800:
        idx.append(i)
    if(len(idx) == 3000):
        break
all = all[idx]
pd.DataFrame(all, columns=["x", "y", "z"]).to_csv("x3027902.txt", sep=" ", index=None)

