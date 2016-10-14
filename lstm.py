#! /usr/bin/env python
#-*- coding: utf-8
""" File: lstm.py
    Author: Thomas Wood, (thomas@synpon.com)
    Description: A minimal LSTM layer for use in TensorFlow networks.
"""

import tensorflow as tf

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
        input_dim,
    """
    def __init__(
                 self,
                 x, # Input to LSTM. Might be a placeholder, might not.
                 init_c, # Initial C state of LSTM. Definite placeholder.
                 init_h,
                 hidden_dim
                 ):
        """
        """
        # Placholders.
        self.x = x
        self.init_c = init_c
        self.init_h = init_h
        # Common parameters
        self.input_dim = int(x.get_shape()[-1])
        self.hidden_dim = hidden_dim
        # Define counters.
        self.current_step = 0
        self.step()


    def step(self):
        """
        """
        input_dim = self.input_dim
        hidden_dim = self.hidden_dim
        x = self.x
        init_h = self.init_h
        init_c = self.init_c

        # Define weight matrices and bias vectors.
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



        # We have to define expressions self.h and self.c
        if self.current_step < 1:
            # The initial h and c states are placeholders.
            ingate = tf.nn.sigmoid(tf.matmul(x, W_i) + \
                tf.matmul(init_h, U_i) + b_i)
            cgate = tf.nn.tanh(tf.matmul(x, W_c) + \
                tf.matmul(init_h, U_c) + b_c)
            fgate = tf.nn.sigmoid(tf.matmul(x, W_f) + \
                tf.matmul(init_h, U_f) + b_f)
            self.c = tf.mul(ingate, cgate) + tf.mul(fgate, init_c)
            ogate = tf.nn.sigmoid(tf.matmul(x, W_o) + tf.matmul(init_h, U_o) + \
                tf.matmul(self.c, V_o) + b_o)
        else:
            # These values for h and c aren't placeholders, but expressions.
            ingate = tf.nn.sigmoid(tf.matmul(x, W_i) + \
                tf.matmul(self.h, U_i) + b_i)
            cgate = tf.nn.tanh(tf.matmul(x, W_c) + \
                tf.matmul(self.h, U_c) + b_c)
            fgate = tf.nn.sigmoid(tf.matmul(x, W_f) + \
                tf.matmul(self.h, U_f) + b_f)
            self.c = tf.mul(ingate, cgate) + tf.mul(fgate, self.c)
            ogate = tf.nn.sigmoid(tf.matmul(x, W_o) + tf.matmul(self.h, U_o) + \
                tf.matmul(self.c, V_o) + b_o)

        # We've finally computed the output/hidden state of the lstm.
        self.h = tf.mul(ogate, tf.nn.tanh(self.c))
        self.current_step += 1