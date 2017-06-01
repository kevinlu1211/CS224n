#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3(d): Grooving with GRUs
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np

class LSTMCell(tf.contrib.rnn.RNNCell):
    """Wrapper around our GRU cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size):
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, cell_state, hidden_state, scope=None):
        """Updates the state using the previous @state and @inputs.
        Remember the GRU equations are:

        z_t = sigmoid(x_t U_z + h_{t-1} W_z + b_z)
        r_t = sigmoid(x_t U_r + h_{t-1} W_r + b_r)
        o_t = tanh(x_t U_o + r_t * h_{t-1} W_o + b_o)
        h_t = z_t * h_{t-1} + (1 - z_t) * o_t

        TODO: In the code below, implement an GRU cell using @inputs
        (x_t above) and the state (h_{t-1} above).
            - Define W_r, U_r, b_r, W_z, U_z, b_z and W_o, U_o, b_o to
              be variables of the apporiate shape using the
              `tf.get_variable' functions.
            - Compute z, r, o and @new_state (h_t) defined above
        Tips:
            - Remember to initialize your matrices using the xavier
              initialization as before.
        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        scope = scope or type(self).__name__

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope, initializer = tf.contrib.layers.xavier_initializer()):
        # with tf.variable_scope(scope):
            ### YOUR CODE HERE (~20-30 lines)

            # Define the variables needed for the input gate
            W_i = tf.get_variable("W_i", shape = (self.input_size, self.state_size)) 
            U_i = tf.get_variable("U_i", shape = (self.state_size, self.state_size))
            b_i = tf.get_variable("b_i", shape = (self.state_size), initializer = tf.constant_initializer(0.0))

            # Define the variables needed for the forget gate
            W_f = tf.get_variable("W_f", shape = (self.input_size, self.state_size)) 
            U_f = tf.get_variable("U_f", shape = (self.state_size, self.state_size))
            b_f = tf.get_variable("b_f", shape = (self.state_size), initializer = tf.constant_initializer(0.0))
            forget_bias = tf.constant(1.0)

            # Define the variables needed for the pre-output gate
            U_o = tf.get_variable("U_o", shape = (self.state_size, self.state_size))
            W_o = tf.get_variable("W_o", shape = (self.input_size, self.state_size))
            b_o = tf.get_variable("b_o", shape = (self.state_size), initializer = tf.constant_initializer(0.0))

            # Define the new memory cell state
            U_c = tf.get_variable("U_c", shape = (self.state_size, self.state_size))
            W_c = tf.get_variable("W_c", shape = (self.input_size, self.state_size))
            b_c = tf.get_variable("b_c", shape = (self.state_size))

            # Define the components of the LSTM
            i_t = tf.nn.sigmoid(tf.matmul(inputs, W_i) + tf.matmul(hidden_state, U_i) + b_i) 
            f_t = tf.nn.sigmoid(tf.matmul(inputs, W_f) + tf.matmul(hidden_state, U_f) + b_f + forget_bias)
            o_t = tf.nn.sigmoid(tf.matmul(inputs, W_o) + tf.matmul(hidden_state, U_o) + b_o)
            new_cell_state = tf.nn.tanh(tf.matmul(inputs, W_c) + tf.matmul(hidden_state, U_c) + b_c)

            final_cell_state = f_t * cell_state + i_t * new_cell_state
            final_hidden_state = o_t * tf.nn.tanh(final_cell_state)

        # initially I was doing:
        # return o_t, final_cell_state, final_hidden_state
        return final_hidden_state, final_cell_state, final_hidden_state