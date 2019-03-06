#!/user/bin/env python
# -*- coding: utf-8 -*-
# Created by Chellee on 2018.12.5

import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

class Data:
    def __init__(self, config):
        self.config = config

    def get_data(self, data_set_name, is_train_data=True):
        """
        Getting train or test data from data_set_name
        :param data_path: train or test data set name
        :param is_train_data: Ture : get train_data;
                              False: get test_data
        :return: train or test data set of type dataframe
        """
        data_set_path = os.path.join(self.config.home + "data/" + data_set_name)
        if os.path.exists(data_set_path):
            data_set = pd.read_csv(data_set_path, encoding="utf-8", sep=',')
            if is_train_data:
                data_set.columns = ["qid", "question_text", "target"]
            else:
                data_set.columns = ["qid", "question_text"]
            return data_set
        else:
            print("data_path: {} does not exist".format(data_set_path))
            exit()

    def get_embedding_dict(self, embedding_name, reset_embedding_table=False, word_set=None):
        """
        Getting embedding table from embedding_name file
        :param embedding_path: word embedding file
        :param word_set: words in data set
        :return: a dict: {word: embedding}
        :
        """
        embedding_path = os.path.join(self.config.home + "embedding_data/" + embedding_name)
        if (os.path.exists(embedding_path)):
            embedding_set = pd.read_csv(embedding_path, encoding="utf-8", sep=' ', header=None)
            word_embedding = dict()
            if reset_embedding_table:
                if word_set == None:
                    print("Please pass list to word_set")
                    raise NameError
                for emb in embedding_set.values:
                    word = emb[0]
                    if word in word_set and word not in word_embedding.keys():
                        word_emb = np.array(emb[1:], dtype=np.float32)
                        assert len(emb) == self.config.embedding_dim + 1
                        word_embedding[word] = word_emb

                new_path = os.path.join(self.config.home + "embedding_data/" + "new" + embedding_name)
                with open(new_path, 'w+', encoding="utf8") as f:
                    for word, emb in word_embedding.items():
                        line = word + ' ' + str(' '.join([str(i) for i in list(emb)])) + '\n'
                        f.write(line)
            else:
                for emb in embedding_set.values:
                    word = emb[0]
                    word_emb = np.array(emb[1:], dtype=np.float32)
                    assert len(emb) == self.config.embedding_dim + 1
                    word_embedding[word] = word_emb
            return word_embedding
        else:
            print("embedding_path: {} does not exist".format(embedding_path))
            exit()

    def word_to_idx(self, data_set):
        """
        :param data_set: a DataFrame data_set
        :return: a dict :{word: index}
        """
        all_sentences = pd.Series(data_set["question_text"].tolist()).unique()
        vectorizer = CountVectorizer(lowercase=True, min_df=self.config.min_word_occurrence)
        vectorizer.fit(all_sentences)
        word2idx = {}
        for idx, word in enumerate(vectorizer.vocabulary_.keys()):
            if word not in word2idx.keys():
                word2idx[word] = idx + 1
        return word2idx

    def a_sent2token(self, word2idx, sentence):
        """
        Transforming a sentence to token
        :param word2idx: a dict of {word: idx}
        :param sentence: a sentence
        :return: a list of token
        """
        sent2token = []
        for sen in sentence.lower().strip().split():
            if sen in word2idx.keys():
                sent2token.append(word2idx[sen])
            else:
                sent2token.append(0)  # zero or unk , I still have to think about it later
        return sent2token

    # def sents2token(self, word2idx, data_set):
    #     """
    #     Transforming all sentences to token
    #     :param word2idx: a dict of {word: idx}
    #     :param data_set: a DataFrame data_set
    #     :return: a DataFrame tokenized data_set
    #     """
    #     return data_set["question_text"].apply(lambda x: self.a_sent2token(word2idx, x))

    def padding_a_sentence(self, sentence):
        mst = self.config.max_sentence_length
        if len(sentence) > mst:
            assert len(sentence[:mst]) == mst, sentence
            return sentence[:mst]
        pad_zero = [0] * (mst - len(sentence))
        assert len(pad_zero + sentence) == mst, pad_zero + sentence
        return sentence + pad_zero

    # def padding_sentences(self, data_set):
    #     return data_set["question_text"].apply(lambda x: self.padding_a_sentence(x))

    def get_embedding_table(self, word2idx, embedding_dict):
        """
        Getting embedding table with shape [vocab_size, embedding_dim]
        :param word2idx: a dict of {word: idx}
        :param embedding_dict: a dict of {word: embedding}
        :return: a embedding table
        """
        embedding_table = np.random.normal(size=(len(word2idx) + 1, self.config.embedding_dim))
        for word, idx in word2idx.items():
            if word in embedding_dict:
                embedding_table[idx] = embedding_dict[word]
        return embedding_table

    def batch_iter(self, data_set, shuffle=True):
        """
        Generate batch
        :param data_set: a data with type of numpy
        """
        batch_num = int((np.shape(data_set)[0]-1) / self.config.batch_size) + 1
        if shuffle:
            shuffle_num = np.random.permutation(np.shape(data_set)[0])
            data_set = data_set[shuffle_num]
        for bat in range(batch_num):
            start_idx = bat * self.config.batch_size
            end_idx = min((bat + 1) * self.config.batch_size, np.shape(data_set)[0])
            yield data_set[start_idx: end_idx, :]

