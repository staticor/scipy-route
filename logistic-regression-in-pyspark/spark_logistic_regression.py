#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: staticor0828@gmail.com
# @Date:   2016-02-26 12:51:58
# @Last Modified by:   staticor0828@gmail.com
# @Last Modified time: 2016-02-26 19:28:07


from time import time

from numpy import array

from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS



def parse(line):
    line_split = line.split(',')
    # first 17 columns: features
    # last column: Y label,
    # 1: lost user; 0: staying user (for now)

    # clean_line_split, keep last one as y. and previous are xs
    lost = 0.0
    clean_line_split = line_split[:-1]
    if line_split[-1] == '1':
        lost = 1.0
    return LabeledPoint(lost, array([float(x) for x in clean_line_split]))

# Initialize SparkContext
sc = SparkContext('local')
# Load and Extract Data
rawdata = sc.textFile('/Users/jinyongyang/Dropbox/Ricebook/user_lost/output_training.csv')

parsed_data = rawdata.map(parse)

# Building model
# time counts

t0 = time()
logit_model = LogisticRegressionWithLBFGS.train(parsed_data)
t_cost = time() - t0

print("Classifier trained in {} seconds.".format(round(t_cost, 3)))

# load test data
testdata_raw = sc.textFile('/Users/jinyongyang/Dropbox/Ricebook/user_lost/output_test.csv')

testdata = testdata_raw.map(parse)

# Evaluating model
labels_and_preds = testdata.map(lambda p: (p.label, logit_model.predict(p.features)) )
t0 = time()

test_accuracy = labels_and_preds.filter(lambda t: t[0] == t[1]).count() / float(testdata.count())

t_cost2 = time() - t0
print("Test accuracy is {}".format( round(test_accuracy, 4)))
