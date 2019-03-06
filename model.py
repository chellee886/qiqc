#!usr/bin/env python3
# -*- utf-8 -*-
# Created by Chellee on 2018.12.23

import tensorflow as tf
from utils import set_rnn_cell

class SelfAttention(object):
    def __init__(self, config, embedding_table, rnn_hidden_size):
        self.max_sentence_length = config.max_sentence_length
        self.embedding_table = embedding_table
        self.input_sentence_fw_pl = tf.placeholder(name='input_sentence_fw_pl',
                                                   shape=[None, self.max_sentence_length],
                                                   dtype=tf.int32)
        self.labels_pl = tf.placeholder(name="labels_pl",
                                        shape=[None, 1],
                                        dtype=tf.int32)
        self.config = config
        self.fw_cell = None
        self.bw_cell = None
        self.hidden_size = rnn_hidden_size

    def set_cell(self, name):
        with tf.variable_scope("rnn_cell"):
            self.fw_cell = set_rnn_cell(name, self.hidden_size)
            self.bw_cell = set_rnn_cell(name, self.hidden_size)

    def _encode(self):
        input_vector = tf.nn.embedding_lookup(self.embedding_table, self.input_sentence_fw_pl)
        outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fw_cell,
                                                         cell_bw=self.bw_cell,
                                                         inputs=input_vector,
                                                         dtype=tf.float64)
        self.outputs = tf.concat(outputs, axis=-1)

    def self_attention(self, d_a=128, r=30):
        """
        :param d_a: The first dimension of W_s1
        :param r: The number of attention
        """
        initializer = tf.contrib.layers.xavier_initializer()
        W_s1 = tf.get_variable(name="W_s1",
                               shape=[d_a, 2 * self.hidden_size],
                               initializer=initializer,
                               dtype=tf.float64)
        W_s2 = tf.get_variable(name="W_s2",
                               shape=[r, d_a],
                               initializer=initializer,
                               dtype=tf.float64)

        A = tf.nn.softmax(tf.map_fn(lambda x: tf.matmul(W_s2, x),
                                    tf.tanh(tf.map_fn(
                                        lambda x: tf.matmul(W_s1, tf.transpose(x)), self.outputs))), axis=-1)

        M = tf.reduce_mean(tf.matmul(A, self.outputs), axis=-2)

        A_t = tf.transpose(A, perm=[0, 2, 1])
        tile_eye = tf.reshape(tf.tile(tf.eye(r, dtype=tf.float64), [tf.shape(self.outputs)[0], 1]), [-1, r, r])
        AA_t = tf.matmul(A, A_t) - tile_eye
        self.P = tf.square(tf.norm(AA_t, axis=[-2, -1], ord="fro"), name="P")

        # flatten = tf.reshape(self.outputs, shape=[tf.shape(self.outputs)[0], -1])

        with tf.variable_scope("hidden_layer"):
            W_hidden = tf.get_variable(name="W_hidden",
                                shape=[2 * self.hidden_size, self.hidden_size],
                                # shape=[self.config.max_sentence_length * 2 * self.hidden_size, 1024],
                                initializer=initializer,
                                dtype=tf.float64)
            b_hidden = tf.get_variable(name="b_hidden",
                                shape=[self.hidden_size],
                                dtype=tf.float64)
            logits_hidden = tf.nn.relu(tf.nn.xw_plus_b(M, W_hidden, b_hidden, name="logits_hidden"))
            logits_dropout = tf.nn.dropout(logits_hidden, keep_prob=0.5)

        with tf.variable_scope("output_layer"):
            W_out = tf.get_variable(name="W_out",
                                    shape=[self.hidden_size, 1],
                                    initializer=initializer,
                                    dtype=tf.float64)
            b_out = tf.get_variable(name="b_out",
                                    shape=[1],
                                    dtype=tf.float64)
            logits_out = tf.nn.xw_plus_b(logits_dropout, W_out, b_out, name="logits_out")
            self.prediction = tf.nn.sigmoid(logits_out)



        with tf.name_scope("loss"):
            self.losses = tf.losses.log_loss(predictions=self.prediction, labels=self.labels_pl)
            self.P = tf.cast(self.P * 0.004, tf.float64)
            self.losses = tf.reduce_mean(tf.cast(self.losses, tf.float64) + self.P)

        with tf.name_scope("accuracy"):
            condition = tf.less(self.prediction, 0.3)
            self.pred_label = tf.cast(tf.where(condition, tf.zeros_like(self.prediction), tf.ones_like(self.prediction), name="accuracy"), tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred_label, self.labels_pl), tf.float32))

class RnnCnnAtt(object):
    def __init__(self, config, embedding_table, rnn_hidden_size):
        self.max_sentence_length = config.max_sentence_length
        self.embedding_table = embedding_table
        self.input_sentence_fw_pl = tf.placeholder(name='input_sentence_fw_pl',
                                                   shape=[None, self.max_sentence_length],
                                                   dtype=tf.int32)
        self.labels_pl = tf.placeholder(name="labels_pl",
                                        shape=[None, 1],
                                        dtype=tf.int32)
        self.config = config
        self.fw_cell = None
        self.bw_cell = None
        self.hidden_size = rnn_hidden_size

    def set_cell(self, name):
        with tf.variable_scope("rnn_cell"):
            self.fw_cell = set_rnn_cell(name, self.hidden_size)
            self.bw_cell = set_rnn_cell(name, self.hidden_size)

    def _encode(self):
        input_vector = tf.nn.embedding_lookup(self.embedding_table, self.input_sentence_fw_pl)
        outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fw_cell,
                                                         cell_bw=self.bw_cell,
                                                         inputs=input_vector,
                                                         dtype=tf.float64)
        outputs = tf.concat(outputs, axis=-1)

        self.outputs = tf.layers.conv1d(inputs=outputs,
                                        filters=self.hidden_size,
                                        strides=1,
                                        kernel_size=3,
                                        padding="same")

    def self_attention(self, d_a=128, r=30):
        """
        :param d_a: The first dimension of W_s1
        :param r: The number of attention
        """
        initializer = tf.contrib.layers.xavier_initializer()
        W_s1 = tf.get_variable(name="W_s1",
                               shape=[d_a, self.hidden_size],
                               initializer=initializer,
                               dtype=tf.float64)
        W_s2 = tf.get_variable(name="W_s2",
                               shape=[r, d_a],
                               initializer=initializer,
                               dtype=tf.float64)

        A = tf.nn.softmax(tf.map_fn(lambda x: tf.matmul(W_s2, x),
                                    tf.tanh(tf.map_fn(
                                        lambda x: tf.matmul(W_s1, tf.transpose(x)), self.outputs))), axis=-1)

        M = tf.reduce_mean(tf.matmul(A, self.outputs), axis=-2)

        A_t = tf.transpose(A, perm=[0, 2, 1])
        tile_eye = tf.reshape(tf.tile(tf.eye(r, dtype=tf.float64), [tf.shape(self.outputs)[0], 1]), [-1, r, r])
        AA_t = tf.matmul(A, A_t) - tile_eye
        self.P = tf.square(tf.norm(AA_t, axis=[-2, -1], ord="fro"), name="P")

        # flatten = tf.reshape(self.outputs, shape=[tf.shape(self.outputs)[0], -1])

        with tf.variable_scope("hidden_layer"):
            W_hidden = tf.get_variable(name="W_hidden",
                                shape=[self.hidden_size, self.hidden_size],
                                # shape=[self.config.max_sentence_length * 2 * self.hidden_size, 1024],
                                initializer=initializer,
                                dtype=tf.float64)
            b_hidden = tf.get_variable(name="b_hidden",
                                shape=[self.hidden_size],
                                dtype=tf.float64)
            logits_hidden = tf.nn.relu(tf.nn.xw_plus_b(M, W_hidden, b_hidden, name="logits_hidden"))
            logits_dropout = tf.nn.dropout(logits_hidden, keep_prob=0.5)

        with tf.variable_scope("output_layer"):
            W_out = tf.get_variable(name="W_out",
                                    shape=[self.hidden_size, 1],
                                    initializer=initializer,
                                    dtype=tf.float64)
            b_out = tf.get_variable(name="b_out",
                                    shape=[1],
                                    dtype=tf.float64)
            logits_out = tf.nn.xw_plus_b(logits_dropout, W_out, b_out, name="logits_out")
            self.prediction = tf.nn.sigmoid(logits_out)


        with tf.name_scope("loss"):
            self.losses = tf.losses.log_loss(predictions=self.prediction, labels=self.labels_pl)
            self.P = tf.cast(self.P * 0.004, tf.float64)
            self.losses = tf.reduce_mean(tf.cast(self.losses, tf.float64) + self.P)

        with tf.name_scope("accuracy"):
            condition = tf.less(self.prediction, 0.3)
            self.pred_label = tf.cast(tf.where(condition, tf.zeros_like(self.prediction), tf.ones_like(self.prediction), name="accuracy"), tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred_label, self.labels_pl), tf.float32))


