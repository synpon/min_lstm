#! /usr/bin/env python
#-*- coding: utf-8
""" File: lstm.py
    Author: Thomas Wood, (thomas@synpon.com)
    Description: A minimal LSTM layer for use in TensorFlow networks.
"""
import tensorflow as tf
import numpy as np
from pprint import pprint

def weight_variable(shape):
    """
    Function: weight_variable(shape)
    Args:
        shape: a list of integers that define the shape of TensorFlow weights.
    Returns:
        tf.Variable(): a TensorFlow tensor of weights.
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    Function: bias_variable(shape)
    Args:
        shape: a list of integers that define shape of bias tensor.
    Returns:
        tf.Variable(): a TensorFlow tensor of bias constants.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class LayerLSTM(object):
    """
    Class: LayerLSTM
    Args:
        x: input to LayerLSTM
        init_c: a placeholder containing initial c state.
        init_h: a placeholder containing initial h state.
        hidden_dim: dimension of the LSTM hidden layer.
        first_run: boolean to say if initial states are placeholders.
    Returns:
        None, but LayerLSTM.h and LayerLSTM.c are useful to pull out.
    """
    def __init__(self, xs, init_c, init_h):
        """
        Function: __init__(self, args)
        Args:
            All the args passed through to instantiate LayerLSTM.
        Returns:
            None
        """
        self.xs = xs
        # Be sure not to assign init_c and init_h any values

        self.init_c = init_c
        self.init_h = init_h
        # Common parameters
        # input_dim = tf.shape(x)[-1]
        # hidden_dim = tf.shape(init_c)[-1]
        self.input_dim = xs.get_shape().as_list()[-1]
        n_steps = xs.get_shape().as_list()[0]
        self.hidden_dim = init_c.get_shape().as_list()[-1]
        self.counter = 0
        self.hs = []
        self.cs = []
        for k in range(n_steps):
            self.step(tf.expand_dims(xs[k,:], 0))
            self.hs.append(self.h)
            self.cs.append(self.c)
        self.H = tf.concat(0, self.hs)
        self.C = tf.concat(0, self.cs)

    def step(self, x):
        input_dim = self.input_dim
        hidden_dim = self.hidden_dim


        # Input gate.
        W_i = weight_variable([input_dim, hidden_dim])
        U_i = weight_variable([hidden_dim, hidden_dim])
        b_i = bias_variable([hidden_dim])
        # Forget gate.
        W_f = weight_variable([input_dim, hidden_dim])
        U_f = weight_variable([hidden_dim, hidden_dim])
        b_f = bias_variable([hidden_dim])
        # Candidate gate.
        W_c = weight_variable([input_dim, hidden_dim])
        U_c = weight_variable([hidden_dim, hidden_dim])
        b_c = bias_variable([hidden_dim])
        # Output gate.
        W_o = weight_variable([input_dim, hidden_dim])
        U_o = weight_variable([hidden_dim, hidden_dim])
        b_o = bias_variable([hidden_dim])
        # Candidate weight in output gate.
        V_o = weight_variable([hidden_dim, hidden_dim])


        ##############################################
        ###          Compute Activations           ###
        ##############################################
        # We have to define expressions self.h and self.c
        # The initial h and c states are possibly placeholders.

        W = tf.concat(1,[W_i, W_f, W_c, W_o])

        U = tf.concat(1,[U_i, U_f, U_c, U_o])

        B = tf.concat(0, [b_i, b_f, b_c, b_o])
        if self.counter < 1:
            H = tf.matmul(x, W) + tf.matmul(self.init_h, U) + B
        else:
            H = tf.matmul(x, W) + tf.matmul(self.h, U) + B


        i, f, c, o = tf.split(1,4,H)

        # Input gate activation.
        igate = tf.nn.sigmoid(i)
        fgate = tf.nn.sigmoid(f)
        cgate = tf.nn.tanh(c)
        if self.counter < 1:
            self.c = tf.mul(igate, cgate) + tf.mul(fgate, self.init_c)
        else:
            self.c = tf.mul(igate, cgate) + tf.mul(fgate, self.c)

        ogate = tf.nn.sigmoid(o + tf.matmul(self.c, V_o))
        # Compute a new value of h to expose to class.
        self.h = tf.mul(ogate, tf.nn.tanh(self.c))
        self.counter += 1

def test_LayerLSTM():
    n_in = 400
    n_hid = 40
    n_steps = 25
    xs = tf.placeholder(tf.float32, shape=[n_steps, n_in])
    init_c = tf.placeholder(tf.float32, shape=[1,n_hid])
    init_h = tf.placeholder(tf.float32, shape=[1,n_hid])
    lstm = LayerLSTM(xs, init_c, init_h)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    feed_dict={lstm.init_c:np.random.rand(1,n_hid),
               lstm.init_h:np.random.rand(1,n_hid),
               lstm.xs:np.random.rand(n_steps,n_in)}
    C, H = sess.run([lstm.H, lstm.C],feed_dict=feed_dict)
    print(C)
    print(H)
    print(H.shape)


if __name__ == "__main__":
    test_LayerLSTM()
