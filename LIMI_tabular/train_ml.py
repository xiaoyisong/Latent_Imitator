import argparse
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from utils import utils_base
from utils.logger import setup_logger
from utils import metrics
from data.census import census_data, census_predict_data
from data.bank import bank_data, bank_predict_data
from data.credit import credit_data, credit_predict_data
from data.meps import meps_data, meps_predict_data
import logging
import joblib

EXP_DIR = "./exp/train_ml"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="svm")
    parser.add_argument("--exp_name", type=str, default="census_gender")
    parser.add_argument("--dataset", type=str, default="census")
    parser.add_argument("--datapath", type=str, default="census.txt")
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument(
        "--protected_index",
        type=int,
        default=8,
        help='the protected attr used in eval',
    )
    parser.add_argument(
        "--evaluate_path", type=str, default="./exp/train_ml/train/census",
    )
    parser.add_argument(
        '--exp_flag', choices=['train', 'predict'], default='train',
    )

    opt = vars(parser.parse_args())
    if opt["exp_flag"] == 'predict':
        opt["expdir"] = os.path.join(opt['evaluate_path'], opt["exp_flag"])
        opt['predict_path'] = os.path.join(opt["expdir"], 'predict_label.txt')
        opt['predict_path2'] = os.path.join(opt["expdir"], 'labels.npy')
        opt['predict_path3'] = os.path.join(opt["expdir"], 'predict_scores.npy')
        opt['model_path'] = os.path.join(
            opt["evaluate_path"], opt['experiment'] + '.pkl'
        )
    else:
        opt["expdir"] = os.path.join(
            EXP_DIR, opt['experiment'], opt["exp_flag"], opt["exp_name"]
        )
        opt['model_path'] = os.path.join(opt["expdir"], opt['experiment'] + '.pkl')

    if not os.path.exists(opt["expdir"]):
        utils_base.make_dir(opt["expdir"])
    if opt["exp_flag"] == 'evaluate':
        logger = setup_logger(
            opt["expdir"], logfile_name='evaluate.txt', logger_name="logger", mode="w",
        )
    else:
        logger = setup_logger(opt["expdir"], logger_name="logger", mode="w")

    np.random.seed(opt["random_seed"])
    logger.info(f"the random_seed is set as {opt['random_seed']}\n")
    logger.info(f"setting {opt}\n")
    return opt


def train(opt, logger):
    data = {
        "census": census_data,
        "credit": credit_data,
        "bank": bank_data,
        "meps": meps_data,
    }
    X_train_raw, Y_train_t, input_shape, nb_classes = data[opt['dataset']]()
    Y_train_raw = 1 - Y_train_t[:, 0]
    print(Y_train_t)
    print(Y_train_raw)
    pos_count, neg_count, length = (
        sum(Y_train_raw),
        len(Y_train_raw) - sum(Y_train_raw),
        len(Y_train_raw),
    )
    logger.info(f"{pos_count}, {neg_count}, {length}")
    if opt['experiment'] == 'rf':
        model = RandomForestClassifier()
    elif opt['experiment'] == 'svm':
        model = SVC(probability=True)
    model.fit(X_train_raw, Y_train_raw)
    joblib.dump(model, opt['model_path'])

    predictions = model.predict(X_train_raw)
    correct_num = np.sum(Y_train_raw == predictions)
    acc = correct_num / len(predictions)
    logger.info(f"acc is {acc}")
    logger.info(f"acc is {acc*100:.2f}")


def predict(opt, logger):
    ## opt['dataset']
    ## opt['datapath']

    data = {
        "census": census_predict_data,
        "credit": credit_predict_data,
        "bank": bank_predict_data,
        "meps": meps_predict_data,
    }
    X_test, _, input_shape, nb_classes = data[opt['dataset']](opt['datapath'])
    model = joblib.load(opt['model_path'])
    logger.info(f"model load from {opt['model_path']}")
    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)
    logger.info(f"len(predictions) is {len(predictions)}")
    logger.info(f"scores is {scores}")
    with open(opt['predict_path'], "w") as handle:
        for _item in predictions:
            handle.write(str(_item) + '\n')
    with open(opt['predict_path2'], "wb+") as handle:
        pickle.dump(predictions, handle)
    with open(opt['predict_path3'], "wb+") as handle:
        pickle.dump(scores, handle)


if __name__ == "__main__":
    opt = _parse_args()
    logger = logging.getLogger("logger")
    if opt['exp_flag'] == 'train':
        train(opt, logger)
    elif opt['exp_flag'] == 'predict':
        predict(opt, logger)
