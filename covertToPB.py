# _*_ coding:utf-8 _*_
# Author:Atlantis
# Date:2019-11-08

from train import EStrain
import tensorflow as tf


class ConvertToPB(object):

    def __init__(self):
        self.model_dir_np = "./output/1"
        self.model_dir_pd = "./output/2"

    def serving_input_receiver_fn(self, feature_spec):
        serizlized_ft_example = tf.placeholder(dtype=tf.float64, shape=[None, 4], name="input_tensor")
        receiver_tensors = {"input": serizlized_ft_example}
        features = tf.parse_example(serizlized_ft_example, feature_spec)
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    def convert_np(self):
        es = EStrain()
        feature_columns = es.get_feature_columns_by_numpy()
        est = es.get_est(self.model_dir_np, feature_columns)
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        est.export_saved_model("./output/1pb", export_input_fn, as_text=True)

    def convert_pd(self):
        es = EStrain()
        feature_columns = es.get_feature_columns_by_pandas()
        est = es.get_est(self.model_dir_pd, feature_columns)
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        est.export_saved_model("./output/2pb", export_input_fn, as_text=True)


if __name__ == '__main__':
    ct = ConvertToPB()
    ct.convert_np()
    ct.convert_pd()
