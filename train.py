# _*_ coding:utf-8 _*_
# Author:Atlantis
# Date:2019-11-08

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class EStrain(object):

    def __init__(self):
        self.iris = load_iris()

    def get_train_test(self):
        data = self.iris.data
        target = self.iris.target
        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def get_feature_columns_by_numpy(self):
        columns = [
            tf.feature_column.numeric_column("your_input", shape=(4,))
        ]
        return columns

    def get_feature_columns_by_pandas(self):
        columns = [
            tf.feature_column.numeric_column(name,shape=(1,)) for name in list("abcd")
        ]
        return columns

    def input_fn_by_numpy(self, x, y):
        return tf.estimator.inputs.numpy_input_fn(
            x={"your_input": x},
            y=y,
            batch_size=512,
            num_epochs=1,
            shuffle=False,
            queue_capacity=1000,
            num_threads=1
        )

    def input_fn_by_pandas(self, x, y):
        return tf.estimator.inputs.pandas_input_fn(
            x,
            y,
            batch_size=32,
            num_epochs=1,
            shuffle=False,
            queue_capacity=1000,
            num_threads=1
        )

    def to_pandas(self, arr, columns):
        return pd.DataFrame(arr, columns=columns)

    def get_est(self, path, feature_columns):
        est = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=[10, 20, 10],
            n_classes=3,
            model_dir=path
        )
        return est

    def train_by_numpy(self):
        x_train, x_test, y_train, y_test = self.get_train_test()
        feature_columns = self.get_feature_columns_by_numpy()
        est = self.get_est("./output/1", feature_columns)
        train_input = self.input_fn_by_numpy(x_train, y_train)
        test_input = self.input_fn_by_numpy(x_test, y_test)
        est.train(input_fn=train_input)
        accuracy_score = est.evaluate(input_fn=test_input)["accuracy"]
        print("accuracy:%s\n" % accuracy_score)
        """ a test example"""
        samples = np.array([[6.4, 3.2, 4.5, 1.5],
                            [6.4, 3.2, 4.5, 1.5]
                            ])
        samples_input = self.input_fn_by_numpy(samples, None)
        predictions = list(est.predict(samples_input))
        print(predictions)
        predicted_classes = int(predictions[0]["classes"])
        print("predict result is %s\n" % predicted_classes)

    def train_by_pandas(self):
        x_train, x_test, y_train, y_test = self.get_train_test()
        feature_columns = self.get_feature_columns_by_pandas()
        est = self.get_est("./output/2", feature_columns)
        x_train_pd = self.to_pandas(x_train, columns=list("abcd"))
        x_test_pd = self.to_pandas(x_test, columns=list("abcd"))
        y_train_pd = pd.Series(y_train)
        y_test_pd = pd.Series(y_test)
        train_input = self.input_fn_by_pandas(x_train_pd, y_train_pd)
        test_input = self.input_fn_by_pandas(x_test_pd, y_test_pd)
        est.train(input_fn=train_input)
        accuracy_score = est.evaluate(input_fn=test_input)["accuracy"]
        print("accuracy:%s\n" % accuracy_score)
        """ a test example"""
        samples = pd.DataFrame(
            [[6.4, 3.2, 4.5, 1.5]], columns=list("abcd")
        )
        samples_input = self.input_fn_by_pandas(samples, None)
        predictions = list(est.predict(samples_input))
        print(predictions)
        predicted_classes = int(predictions[0]["classes"])
        print("predict result is %s\n" % predicted_classes)


if __name__ == '__main__':
    et = EStrain()
    et.train_by_pandas()
    et.train_by_numpy()
