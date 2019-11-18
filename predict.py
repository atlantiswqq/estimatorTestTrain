# _*_ coding:utf-8 _*_
# Author:Atlantis
# Date:2019-11-11

import numpy as np
import pandas as pd
import tensorflow as tf


class PredictByPB(object):
    def __init__(self):
        self.np_path = "./output/1pb/1"
        self.pd_path = "./output/2pb/1"

    def get_predict_fn(self, path):
        predict_fn = tf.contrib.predictor.from_saved_model(path)
        return predict_fn

    def predict_np(self):
        predict_fn = self.get_predict_fn(self.np_path)
        inputs = np.array([[6.4, 3.2, 4.5, 1.5], [6.4, 3.2, 4.5, 1.5]])
        for item in inputs:
            examples = []
            feature = {"your_input": tf.train.Feature(float_list=tf.train.FloatList(value=item))}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            examples.append(example.SerializeToString())
            predictions = predict_fn({"inputs": examples})
            print(predictions)

    def predict_pd(self):
        predict_fn = self.get_predict_fn(self.pd_path)
        inputs = pd.DataFrame(np.array([[6.4, 3.2, 4.5, 1.5], [6.4, 3.2, 4.5, 1.5]]), columns=list("abcd"))
        for row_index, row in inputs.iterrows():
            examples = []
            feature = {}
            for col, value in row.iteritems():
                feature[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            examples.append(example.SerializeToString())
            predictions = predict_fn({"inputs": examples})
            print(predictions)


if __name__ == '__main__':
    pb = PredictByPB()
    pb.predict_np()
    pb.predict_pd()
