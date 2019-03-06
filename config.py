#!/user/bin/env python
# -*- coding: utf-8 -*-
# Created by Chellee on 2018.12.5

import json
import os

class Config:
    def __init__(self, config_file_path):
        if os.path.exists(config_file_path):
            cfg = json.load(open(config_file_path, 'r'))
            self.home = os.path.dirname(os.path.abspath(__file__)) + "/"
            self.learning_rate = cfg["learning_rate"]
            self.embedding_dim = cfg["embedding_dim"]
            self.min_word_occurrence = cfg["min_word_occurrence"]
            self.max_sentence_length = cfg["max_sentence_length"]
            self.epoch_num = cfg["epoch_num"]
            self.batch_size = cfg["batch_size"]
            self.cross_validation = cfg["cross_validation"]
        else:
            print("Config file path {} does not exist".format(config_file_path))
            exit()
