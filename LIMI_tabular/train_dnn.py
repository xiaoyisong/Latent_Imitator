import argparse
import copy
import math
import os
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import sys

import tensorflow as tf
from data import census, credit, bank, meps
from data.census import census_data, census_predict_data, census_eval_data
from data.bank import bank_data, bank_predict_data, bank_eval_data
from data.credit import credit_data, credit_predict_data, credit_eval_data
from data.meps import meps_data, meps_predict_data, meps_eval_data

from utils.utils_tf import model_train, model_eval
from table_model.dnn_models import dnn
from utils import utils_base
from utils.logger import setup_logger
from utils import metrics

import logging
import joblib

EXP_DIR = "./exp/train_dnn"


def numpy2log(arr, output_path):
    f = open(output_path, 'w')
    for item in arr:
        if isinstance(item, list):
            line = ','.join(list(map(str, item)))
        else:
            line = str(item)
        f.write(line + '\n')
    return


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="train_dnn")
    parser.add_argument("--exp_name", type=str, default="census_gender")
    parser.add_argument("--dataset", type=str, default="census")
    parser.add_argument("--nb_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--sample_add_rate", type=float, default=0.3)
    parser.add_argument("--datapath", type=str, default="census_aug.txt")
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument(
        "--protected_index",
        type=int,
        default=8,
        help='the protected attr used in eval',
    )
    parser.add_argument(
        "--method_type", type=str, default='adf', choices=['adf', 'nf', 'expga', 'ours']
    )
    parser.add_argument(
        "--evaluate_path",
        type=str,
        default="./exp/train_dnn/retrain/cencus_gender_5_ours",
    )
    parser.add_argument(
        '--exp_flag',
        choices=['train', 'retrain', 'evaluate'],
        default='train',
        required=True,
    )
    parser.add_argument("--des", type=str, default="description")
    parser.add_argument('--evaluate_log', default='evaluate_train')
    opt = vars(parser.parse_args())
    if opt["exp_flag"] == 'evaluate':
        opt["expdir"] = os.path.join(opt['evaluate_path'], opt["exp_flag"])
        opt['model_path'] = os.path.join(opt["evaluate_path"])  # use for
    elif opt['exp_flag'] == 'retrain':
        if opt['des'] != 'description':
            opt["expdir"] = os.path.join(
                EXP_DIR,
                opt["exp_flag"] + '_' + opt['des'],
                opt['method_type'],
                opt["exp_name"],
            )
        else:
            opt["expdir"] = os.path.join(
                EXP_DIR, opt["exp_flag"], opt['method_type'], opt["exp_name"]
            )
        opt['model_path'] = os.path.join(opt["expdir"])
    else:
        opt["expdir"] = os.path.join(EXP_DIR, opt["exp_flag"], opt["exp_name"])
        opt['model_path'] = os.path.join(opt["expdir"])

    opt['aug_store_path'] = os.path.join(opt["expdir"], 'aug.txt')
    opt['predict_path'] = os.path.join(opt["expdir"], 'predict_label.txt')

    if not os.path.exists(opt["expdir"]):
        utils_base.make_dir(opt["expdir"])
    if opt["exp_flag"] == 'evaluate':
        logger = setup_logger(
            opt["expdir"],
            logfile_name=opt['evaluate_log'] + '.txt',
            logger_name="logger",
            mode="w",
        )
    else:
        logger = setup_logger(opt["expdir"], logger_name="logger", mode="w")

    np.random.seed(opt["random_seed"])
    logger.info(f"the random_seed is set as {opt['random_seed']}\n")
    logger.info(f"setting {opt}\n")
    return opt


def training(dataset, model_path, nb_epochs, batch_size, learning_rate):
    """
    Train the model
    :param dataset: the name of testing dataset
    :param model_path: the path to save trained model
    """
    data = {
        "census": census_data,
        "credit": credit_data,
        "bank": bank_data,
        "meps": meps_data,
    }
    # prepare the data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    tf.set_random_seed(opt["random_seed"])
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)
    preds = model(x)
    sess.run(tf.global_variables_initializer())

    # training parameters
    train_params = {
        "nb_epochs": nb_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_dir": model_path + "/dnn/",
        "filename": "best.model",
    }

    rng = np.random.RandomState([2021, 3, 15])
    model_train(sess, x, y, preds, X, Y, args=train_params, rng=rng, save=True)

    # evaluate the accuracy of trained model
    eval_params = {"batch_size": 128}
    accuracy = model_eval(sess, x, y, preds, X, Y, args=eval_params)
    logger.info("Test accuracy on legitimate test examples: {0}".format(accuracy))


def retraining(opt: dict):
    # prepare the data origin
    data = {
        "census": census_data,
        "credit": credit_data,
        "bank": bank_data,
        "meps": meps_data,
    }

    X_train_raw, Y_train_raw, input_shape, nb_classes = data[opt['dataset']]()

    # just need to prepare the aug_data
    # sample_add_rate, datapath
    ensemble_clf = joblib.load(
        os.path.join('./exp/ensemble_models/' + opt['dataset'] + '_ensemble.pkl')
    )

    data_aug = {
        "census": census_predict_data,
        "credit": credit_predict_data,
        "bank": bank_predict_data,
        "meps": meps_predict_data,
    }
    # prepare the data and model
    X_aug, Y_aug, _input_shape, _nb_classes = data_aug[opt['dataset']](opt['datapath'])
    protected_attribs = None
    if opt['dataset'] == 'census':
        protected_attribs = census.protected_attribs
    elif opt['dataset'] == 'credit':
        protected_attribs = credit.protected_attribs
    elif opt['dataset'] == 'bank':
        protected_attribs = bank.protected_attribs
    elif opt['dataset'] == 'meps':
        protected_attribs = meps.protected_attribs

    ids_aug = np.empty(shape=(0, len(X_train_raw[0])))
    num_aug = int(len(X_train_raw) * opt['sample_add_rate'])  # percentage of raw data
    for _ in range(num_aug):
        rand_index = np.random.randint(len(X_aug))
        ids_aug = np.append(ids_aug, [X_aug[rand_index]], axis=0)
    label_vote = ensemble_clf.predict(np.delete(ids_aug, protected_attribs, axis=1))
    Y_aug = []
    for _label in label_vote:
        if _label == 0:
            Y_aug.append([1, 0])
        else:
            Y_aug.append([0, 1])
    Y_aug = np.array(Y_aug, dtype=float)
    logger.info(f"Y_aug.shape is {Y_aug.shape}")
    logger.info(f"label_vote is {label_vote}")
    logger.info(f"sum of label_vote is {sum(label_vote)}")
    logger.info(f"Y_aug is {Y_aug}")

    X_train = np.append(X_train_raw, ids_aug, axis=0)
    Y_train = np.append(Y_train_raw, Y_aug, axis=0)

    _arr = np.append(ids_aug, Y_aug, axis=1)
    numpy2log(_arr, opt['aug_store_path'])

    # then just do the training
    tf.set_random_seed(opt["random_seed"])
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)
    preds = model(x)
    sess.run(tf.global_variables_initializer())

    # training parameters
    train_params = {
        "nb_epochs": opt['nb_epochs'],
        "batch_size": opt['batch_size'],
        "learning_rate": opt['learning_rate'],
        "train_dir": opt['model_path'] + "/dnn/",
        "filename": "best.model",
    }

    rng = np.random.RandomState([2021, 3, 15])
    model_train(
        sess, x, y, preds, X_train, Y_train, args=train_params, rng=rng, save=True
    )

    # evaluate the accuracy of trained model
    eval_params = {"batch_size": 128}
    accuracy = model_eval(sess, x, y, preds, X_train_raw, Y_train_raw, args=eval_params)
    logger.info("Test accuracy on legitimate test examples: {0}".format(accuracy))


def eval_ori(sess, dataset, x, y, preds):
    data = {
        "census": census_data,
        "credit": credit_data,
        "bank": bank_data,
        "meps": meps_data,
    }
    # prepare the data and model
    X, Y, input_shape, nb_classes = data[dataset]()
    eval_params = {"batch_size": 128}
    accuracy = model_eval(sess, x, y, preds, X, Y, args=eval_params)
    logger.info("Test accuracy on legitimate test examples: {0}".format(accuracy))
    return accuracy


def evaluate(opt):
    # prepare the data origin
    data = {
        "census": census_eval_data,
        "credit": credit_eval_data,
        "bank": bank_eval_data,
        "meps": meps_eval_data,
    }
    batch_size = opt['batch_size']
    X, Y, input_shape, nb_classes = data[opt['dataset']](
        protected_index=opt['protected_index']
    )
    # then just do the training
    tf.set_random_seed(opt["random_seed"])
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model = dnn(input_shape, nb_classes)
    preds = model(x)

    saver = tf.train.Saver()
    model_path = os.path.join(opt['model_path'], 'dnn', "best.model")
    logger.info(f"load model from {model_path}")
    saver.restore(sess, model_path)

    _accuracy = eval_ori(sess, opt['dataset'], x, y, preds)

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

            feed_dict = {x: X[start:end]}
            pros = sess.run(preds, feed_dict)
            for i in range(start, end):
                pros_all[i] = pros[i - start]

    labels = np.argmax(pros_all, axis=1)
    numpy2log(labels, opt['predict_path'])

    _domain, _targets, _pred = Y[:, 1], Y[:, 0], labels
    print('sum of _domain', sum(_domain))
    print('len of _domain', len(_domain))

    spd, di, spd_abs, di_abs = metrics.SPD_DI(_domain, _targets, _pred)
    eod, eod_abs = metrics.EOD(_domain, _targets, _pred)
    aod, aod_abs = metrics.AOD(_domain, _targets, _pred)
    erd, erd_abs = metrics.ERD(_domain, _targets, _pred)

    test_results = {
        'SPD': spd,
        'DI': di,
        'EOD': eod,
        'AOD': aod,
        'ERD': erd,
        #
        'SPD_S': spd_abs,
        'DI_S': di_abs,
        'EOD_S': eod_abs,
        'AOD_S': aod_abs,
        'ERD_S': erd_abs,
    }
    logger.info('test results: ')
    logger.info(f"SPD : {spd} , DI : {di}")
    logger.info(f"EOD : {eod}")
    logger.info(f"AOD : {aod}")
    logger.info(f"ERD : {erd}")

    logger.info(f"SPD : {spd_abs} , DI : {di_abs}")
    logger.info(f"EOD : {eod_abs}")
    logger.info(f"AOD : {aod_abs}")
    logger.info(f"ERD : {erd_abs}")

    logger.info(f"test_results, {test_results}")

    logger.info(
        f"copy: {_accuracy:.5f},{spd_abs:.5f},{eod_abs:.5f},{aod_abs:.5f},{erd_abs:.5f},"
    )


if __name__ == "__main__":
    opt = _parse_args()
    logger = logging.getLogger("logger")
    if opt['exp_flag'] == 'retrain':
        retraining(opt)
    elif opt['exp_flag'] == 'train':
        training(
            dataset=opt['dataset'],
            model_path=opt['model_path'],
            nb_epochs=opt['nb_epochs'],
            batch_size=opt['batch_size'],
            learning_rate=opt['learning_rate'],
        )
    elif opt['exp_flag'] == 'evaluate':
        evaluate(opt)
