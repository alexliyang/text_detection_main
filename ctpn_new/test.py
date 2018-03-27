"""这个文件用于写代码时候的测试"""
import numpy as np
import tensorflow as tf
a = np.array([1, 8, 5, 6, 4, 2, 9, 5, 4, 4])
gg = tf.equal(a, 4)
rpn_keep = tf.where(gg)
with tf.Session() as sess:
    print(rpn_keep.eval())
    print(sess.run(tf.shape(rpn_keep)[0]))
    b = tf.gather(a, rpn_keep)
    print("=====================")
    print(b.eval())
    print("=======================")
    print(gg.eval())

