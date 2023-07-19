import copy
import math
import os
import numpy as np
import sys
import pickle

sys.path.append("../")
import tensorflow as tf
from tensorflow.python.platform import flags
from data.census import census_predict_data
from data.bank import bank_predict_data
from data.credit import credit_predict_data
from data.compas import compas_predict_data
from data.meps import meps_predict_data
from utils.utils_tf import model_train, model_eval
from table_model.dnn_models import dnn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
FLAGS = flags.FLAGS


def predicting(dataset, dataset_path, model_path, batch_size):
    """
    Train the model
    :param dataset: the name of testing dataset
    :param model_path: the path to save trained model
    """
    data = {
        "census": census_predict_data,
        "credit": credit_predict_data,
        "bank": bank_predict_data,
        "compas": compas_predict_data,
        "meps": meps_predict_data,
    }
    # prepare the data and model
    X, Y, input_shape, nb_classes = data[dataset](dataset_path)
    tf.set_random_seed(1234)
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)
    preds = model(x)
    saver = tf.train.Saver()
    model_path = os.path.join(model_path, "dnn", "best.model")
    print("load model from ", model_path)
    saver.restore(sess, model_path)

    with sess.as_default():
        nb_batches = int(math.ceil(float(len(X)) / batch_size))
        assert nb_batches * batch_size >= len(X)
        pros_all = np.zeros(shape=(X.shape[0], nb_classes), dtype="float32")

        X_cur = np.zeros((batch_size,) + X.shape[1:], dtype=X.dtype)
        for batch in range(nb_batches):
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))
            start = batch * batch_size
            end = min(len(X), start + batch_size)
            cur_batch_size = end - start
            X_cur[:cur_batch_size] = X[start:end]
            # feed_dict = {x: X_cur}
            feed_dict = {x: X[start:end]}
            pros = sess.run(preds, feed_dict)
            for i in range(start, end):
                pros_all[i] = pros[i - start]
    print("X[0]", X[0])
    print("len(pros_all)", len(pros_all))
    print("pros_all[0]", pros_all[0])
    labels = np.argmax(pros_all, axis=1)
    print("labels[0]", labels[0])
    return pros_all, labels


def main(argv=None):
    pros_all, labels = predicting(
        dataset=FLAGS.dataset,
        dataset_path=FLAGS.dataset_path,
        model_path=FLAGS.model_path,
        batch_size=FLAGS.batch_size,
    )

    with open(FLAGS.output_path, "wb+") as handle:
        pickle.dump(pros_all, handle)
    with open(FLAGS.output_path2, "wb+") as handle:
        pickle.dump(labels, handle)


if __name__ == "__main__":
    flags.DEFINE_string("dataset", "census", "the name of dataset")
    flags.DEFINE_string(
        "dataset_path", "../datasets/census", "the path of test dataset"
    )

    flags.DEFINE_string(
        "model_path", "../logs/census", "the name of path for saving model"
    )
    flags.DEFINE_integer("batch_size", 64, "Size of training batches")
    flags.DEFINE_string(
        "output_path", "../logs/census/predict_scores.npy", "Size of training batches"
    )
    flags.DEFINE_string(
        "output_path2", "../logs/census/labels.npy", "Size of training batches"
    )
    tf.app.run()
