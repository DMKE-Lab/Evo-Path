# coding=UTF-8
from __future__ import division
from __future__ import absolute_import
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


class baseline(object):
    def get_baseline_value(self):
        pass

    def update(self, target):
        pass


class ReactiveBaseline(baseline):
    def __init__(self, l):
        self.l = l
        self.b = tf.Variable(0.0, trainable=False)

    def get_baseline_value(self):
        return self.b

    def update(self, target):
        self.b = tf.add((1 - self.l) * self.b, self.l * target)
