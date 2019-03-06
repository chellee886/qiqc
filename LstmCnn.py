#!usr/bin/env python3
# -*- utf-8 -*-
# Created by Chellee on 2018.12.23
import tensorflow as tf
from utils import set_rnn_cell

class LSTM_CNN(object):
    def __init__(self, config, embedding_table, rnn_hidden_size):
        self.max_sentence_length = config.max_sentence_length
        self.embedding_table = embedding_table
        self.input_sentence_fw_pl = tf.placeholder(name="input_sentence_fw_pl",
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

    def _encoder(self):
        input_vector = tf.nn.embedding_lookup(self.embedding_table, self.input_sentence_fw_pl)
        outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw = self.fw_cell,
                                                         cell_bw = self.bw_cell,
                                                         inputs=input_vector,
                                                         dtype=tf.float64)
        self.outputs = tf.concat(outputs, axis=-1)
