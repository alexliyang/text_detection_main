"""这个文件用于写代码时候的测试"""
import tensorflow as tf
a = tf.placeholder(shape=[2,3], dtype=tf.float16)
print(a.get_shape()[-1])
