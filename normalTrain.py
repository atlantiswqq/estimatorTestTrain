# _*_ coding:utf-8 _*_
# Author:Atlantis
# Date:2019-11-18

import pickle
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC


class NormalTrain(object):
    def __init__(self):
        self.data = load_iris().data

    def get_clf(self):
        clf = LinearSVC
        return clf

    def save_model(self):
        clf = self.get_clf()
        clf.fit("xx","xx")
        with open("test.pkl","wb") as f:
            f.write(pickle.dumps(clf,2))
