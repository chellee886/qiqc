#!/user/bin/env python3
# -*- coding: utf-8 -*-
# Created by Chellee on 2018.12.5

import tensorflow as tf
import numpy as np
from model import SelfAttention, RnnCnnAtt
from data_prepare import Data
from config import Config
import time
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from utils import clean_text, MISPELL_DICT
import warnings


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
warnings.filterwarnings("ignore")

def train(training_data, config, embedding_table):
    """"""
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        selfatt = RnnCnnAtt(config=config,
                                embedding_table=embedding_table,
                                rnn_hidden_size=128)

        # Model
        selfatt.set_cell(name="LSTM")
        selfatt._encode()
        selfatt.self_attention(d_a=128, r=30)

        train_summary_writer = tf.summary.FileWriter(config.home + "summary", sess.graph)
        loss_summary_op = tf.summary.scalar("loss", selfatt.losses)

        # global_step = tf.Variable(0, name="global_step", trainable=False)
        checkpoints_dir = os.path.join(config.home, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(selfatt.losses)

        skf = StratifiedKFold(n_splits=config.cross_validation, shuffle=True)
        model_count = 0
        for idx_train, idx_val in skf.split(training_data, np.array(training_data[:, 2], dtype=int)):

            sess.run(tf.global_variables_initializer())
            print("Model : {}".format(model_count))
            for epoch in range(config.epoch_num):
                step = 0
                print("epoch: ", epoch)
                batches = data.batch_iter(data_set=training_data[idx_train], shuffle=True)
                start_time = time.time()
                for batch in batches:
                    x_batch = np.array(list(batch[:, 1]))
                    y_batch = batch[:, 2].reshape(-1, 1)
                    feed_dict = {
                        selfatt.input_sentence_fw_pl: x_batch,
                        selfatt.labels_pl: y_batch
                    }

                    _, loss, loss_summary, accuracy, pred = sess.run([train_op, selfatt.losses, loss_summary_op, selfatt.accuracy, selfatt.pred_label], feed_dict=feed_dict)
                    train_summary_writer.add_summary(loss_summary)
                    step += 1
                    if(step % 5 == 0):
                        print("{}/{}, time: {}, loss: {}, accuracy: {}, true_f1: {}".
                              format(min(step*config.batch_size, len(idx_train)),
                                     len(idx_train),
                                     time.time() - start_time,
                                     loss,
                                     accuracy,
                                     f1_score(y_true=batch[:, 2].astype(int), y_pred=pred.astype(int), labels=0)))
                        start_time = time.time()

                print("\nEvalucation:")
                pred_list = []
                batches_val = data.batch_iter(data_set=training_data[idx_val], shuffle=False)
                for batch_val in batches_val:
                    start_time_dev = time.time()
                    fedd_dict_dev = {
                        selfatt.input_sentence_fw_pl: np.array(list(batch_val[:, 1])),
                        selfatt.labels_pl: batch_val[:, 2].reshape(-1, 1)
                    }
                    loss, accuracy, pred = sess.run([selfatt.losses, selfatt.accuracy, selfatt.pred_label], feed_dict=fedd_dict_dev)
                    print("{}, time: {}, loss: {}, accuracy: {}".format(len(idx_val),
                                                                            time.time() - start_time_dev,
                                                                            loss,
                                                                            accuracy
                                                                            ))
                    pred_list.append(pred.astype(int))
                print("f1_val: {}".format(f1_score(y_true=training_data[idx_val][:, 2].astype(int),y_pred=np.concatenate(pred_list))))
                # Model checkpoint
                chk_model = os.path.join(checkpoints_dir, "model{}".format(model_count))
                saver.save(sess, chk_model, global_step=epoch)

                epoch += 1

            model_count += 1

if __name__ == "__main__":
    config = Config("./config.json")

    data = Data(config)

    print("Getting training and testing data...")
    training_data = data.get_data(data_set_name="train.csv", is_train_data=True)
    testing_data = data.get_data(data_set_name="test.csv", is_train_data=False)
    # print(training_data["question_text"])

    print("Cleaning data...")
    training_data["question_text"] = training_data["question_text"].apply(lambda x: clean_text(x, MISPELL_DICT))
    testing_data["question_text"] = testing_data["question_text"].apply(lambda x: clean_text(x, MISPELL_DICT))
    # print(training_data["question_text"])

    print("Getting word embedding...")
    word2idx = data.word_to_idx(training_data)
    # print("before : {}".format(len(word2idx)))

    # print(len(word2idx))
    emb_dict = data.get_embedding_dict(embedding_name="newglove.840B.300d.txt",
                                       reset_embedding_table=False,
                                       word_set=word2idx.keys())
    # print("After : {}".format(len(emb_dict)))

    emb_table = data.get_embedding_table(word2idx=word2idx, embedding_dict=emb_dict)

    print("Sentences tokenizer...")
    training_data["question_text"] = training_data["question_text"].apply(lambda x: data.a_sent2token(word2idx, x))
    testing_data["question_text"] = testing_data["question_text"].apply(lambda x: data.a_sent2token(word2idx, x))
    # print(training_data["question_text"])

    print("Padding...")
    training_data["question_text"] = training_data["question_text"].apply(lambda x: data.padding_a_sentence(x))
    testing_data["question_text"] = testing_data["question_text"].apply(lambda x: data.padding_a_sentence(x))
    train(training_data=training_data.values, config=config, embedding_table=emb_table)
