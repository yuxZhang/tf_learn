#coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

IRIS_TRAINING = os.path.join(os.path.dirname(__file__), "iris_training.csv")
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = os.path.join(os.path.dirname(__file__), "iris_test.csv")
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"


def main(unused_argv):

    tf.logging.set_verbosity(tf.logging.INFO)
    #training_set = my_input_fn(IRIS_TRAINING)
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(test_set.data,
                                                                     test_set.target,
                                                                     every_n_steps=50)

    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=3,
                                                model_dir="/home/zhangyuxiang/dl_learn/tf_learn/contrib_learn/tmp/iris_model",
                                                config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))

    # Define the training inputs
    # def get_train_inputs():
    #     x = tf.constant(training_set.data)
    #     y = tf.constant(training_set.target)
    #
    #     return x, y
    # print(training_set.data)
    # return
    # Fit model.
    classifier.fit( x=training_set.data,
                    y=training_set.target,
                    steps=2000,
                    monitors=[validation_monitor])

    # Define the test inputs
    # def get_test_inputs():
    #     x = tf.constant(test_set.data)
    #     y = tf.constant(test_set.target)
    #
    #     return x, y

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(x=test_set.data,
                                         y=test_set.target)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    # Classify two new flower samples.
    def new_samples():
        return np.array(
            [[6.4, 3.2, 4.5, 1.5],
             [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

    predictions = list(classifier.predict(input_fn=new_samples))

    print(
        "New Samples, Class Predictions:    {}\n"
            .format(predictions))


if __name__ == "__main__":
    tf.app.run()
