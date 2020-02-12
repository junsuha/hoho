import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import random
import pandas as pd
import os

Drugdata = []
druganwser = []
EPAdata = []
epaanwser = []
with open("../logP_labels.smi", "r") as EPA:
    with open("../alogptest.csv", "r") as DRUG:
        dlines = DRUG.readlines()
        Elines = EPA.readlines()
        for dline in dlines:
            dsl = dline.split()
            Drugdata.append(" ".join(dsl[0]))
            druganwser.append(dsl[1])
        for Eline in Elines:
            Esl = Eline.split()
            EPAdata.append(" ".join(Esl[0]))
            epaanwser.append(Esl[1])

x_train = np.array(EPAdata, dtype=np.str)
y_train = np.array(epaanwser, dtype=np.str)
x_test = np.array(Drugdata, dtype=np.str)
y_test = np.array(druganwser, dtype=np.str)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
MAX_LENGTH = 400
SOS_token = 1
EOS_token = 2
PAD_token = 0


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # PAD 포함

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word

            self.n_words += 1
        else:
            self.word2count[word] += 1


EPAlang = Lang("EPA")
Druglang = Lang("Drug")
E = []
D = []


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence_encond(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    while len(indexes) >= MAX_LENGTH - 1:
        del indexes[MAX_LENGTH - 2]

    indexes.append(EOS_token)
    while len(indexes) < MAX_LENGTH:
        indexes.append(PAD_token)
    return indexes


def tensorsFromPair(EP):
    input_tensor = tensorFromSentence_encond(EPAlang, EP)
    return input_tensor


for EP in x_train:
    EPAlang.addSentence(EP)
    Druglang.addSentence(EP)
    ab = tensorsFromPair(EP)
    a = [str(ab[i]) for i in range(len(ab))]
    E.append(" ".join(a))
for Dru in x_test:
    EPAlang.addSentence(Dru)
    Druglang.addSentence(Dru)
    ab = tensorsFromPair(Dru)
    a = [str(ab[i]) for i in range(len(ab))]
    D.append(" ".join(a))
import tensorflow as tf


def correlation_coefficient(y_true, y_pred):
    pearson_r, update_op = tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)
    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'correlation_coefficient' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        pearson_r = tf.identity(pearson_r)
        return pearson_r
x_train = np.array(E, dtype=np.str)
x_test = np.array(D, dtype=np.str)

import autokeras as ak

# Initialize the text classifier.
clf = ak.TextRegressor(max_trials=2,objective="val_mean_squared_error", overwrite=False, loss="mean_absolute_error") # It tries 10 different models.
# Feed the text classifier with training data.
clf.fit(x_train, y_train)
# Predict with the best model.
predicted_y = clf.predict(x_test)
# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))
