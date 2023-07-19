import argparse
import pickle
import logging
import numpy as np
from sklearn import svm
from sklearn import metrics
from utils.logger import setup_logger
from sklearn.preprocessing import label_binarize
from utils import util
import torch
import os

EXP_DIR = './exp'


def _collect_args_boundary():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='train_boundaries')
    parser.add_argument('--exp_name', type=str, default='_')
    parser.add_argument('--latent_file', type=str, default='_')
    parser.add_argument('--score_file', type=str, default='_')
    parser.add_argument('--label_file', type=str, default='_')
    parser.add_argument('--attribute', type=int, default=31)
    parser.add_argument('--random_seed', type=int, default=2333)
    parser.add_argument('--train_num', type=int, default=5000)
    parser.set_defaults(cuda=True)

    opt = vars(parser.parse_args())

    attr_list = util.get_all_attr()
    attr_name = attr_list[opt['attribute']]
    opt['attr_name'] = attr_name
    opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
    opt['dtype'] = torch.float32

    if opt['exp_name'] == '_':
        opt['exp_name'] = str(opt['attribute']) + '_' + attr_name

    opt['expdir'] = os.path.join(EXP_DIR, opt['experiment'])
    util.make_dir(opt['expdir'])

    # save the boundary of the attr
    opt['output_file'] = os.path.join(opt['expdir'], opt['exp_name'] + '.npy')
    opt['output_file2'] = os.path.join(opt['expdir'], opt['exp_name'] + '_svm.npy')

    logger = setup_logger(opt['expdir'], logger_name='logger', mode='a+')

    # Uncomment if deterministic run required
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(opt['random_seed'])
    np.random.seed(opt['random_seed'])
    logger.info(f"the random_seed is set as {opt['random_seed']}\n")
    logger.info(f"setting {opt}\n")

    return opt


def reset(latents, scores):
    index_1 = np.where(scores == 1)
    index_0 = np.where(scores == 0)
    latent_1 = latents[index_1]
    latent_0 = latents[index_0]
    logger.info(f'len of latent_0 {len(latent_0)}, len of latent_1 {len(latent_1)}')

    latent_1 = list(latent_1)
    latent_0 = list(latent_0)
    if len(latent_0) < len(latent_1):
        for item in latent_0:
            if len(latent_0) < len(latent_1):
                latent_0.append(item)
            else:
                break
    elif len(latent_1) < len(latent_0):
        for item in latent_1:
            if len(latent_1) < len(latent_0):
                latent_1.append(item)
            else:
                break
    len0, len1 = len(latent_0), len(latent_1)
    logger.info(f'len0: {len0}, len1: {len1}')
    latent_new = []
    score_new = []
    latent_new.extend(latent_0)
    score_new.extend(np.zeros(len(latent_0)))
    latent_new.extend(latent_1)
    score_new.extend(np.ones(len(latent_1)))
    latent_new, score_new = np.array(latent_new), np.array(score_new)
    logger.info(
        f'latent_new.shape: {latent_new.shape}, score_new.shape: {score_new.shape}'
    )
    return np.array(latent_new), np.array(score_new)


def valid(latent, score, line_svm, name: str):
    latent_val = latent
    score_val = score
    val_prediction = line_svm.predict(latent_val)
    correct_num = np.sum(score_val == val_prediction)
    logger.info(
        f"Accuracy for {name} set: "
        f"{correct_num} / {len(latent_val) } = "
        f"{correct_num / len(latent_val) :.6f}"
    )

    y_test_hot = label_binarize(score_val, classes=(0, 1))
    # print(y_test_hot)
    svm_y_score = line_svm.decision_function(latent_val)
    svm_fpr, svm_tpr, svm_threasholds = metrics.roc_curve(
        y_test_hot.ravel(), svm_y_score.ravel()
    )
    svm_auc = metrics.auc(svm_fpr, svm_tpr)
    logger.info(f"AUC for validation set: {svm_auc:.6f}\n")
    A1 = (score_val[score_val == val_prediction] == 1).sum()
    B1 = (score_val[score_val != val_prediction] == 1).sum()
    C1 = (score_val[score_val != val_prediction] == 0).sum()
    D1 = (score_val[score_val == val_prediction] == 0).sum()
    logger.info(f"TP: {A1}, " f"FN: {B1}, " f"FP: {C1}, " f"TN: {D1}\n")


def train_equal():
    logger.info(f"train the boundary of {opt['attr_name']}")

    logger.info(f"load the latent_file from {opt['latent_file']}")
    with open(opt['latent_file'], 'rb') as f:
        latents = pickle.load(f)

    logger.info(f"load the label_file from {opt['label_file']}")
    with open(opt['label_file'], 'rb') as f:
        labels = pickle.load(f)
    logger.info(f"load the score_file from {opt['score_file']}")
    with open(opt["score_file"], "rb") as f:
        scores = pickle.load(f)

    positive_train_id = np.where(labels == 1)[0]
    negative_train_id = np.where(labels == 0)[0]
    logger.info(f"label=1 {len(positive_train_id)}, label=0 {len(negative_train_id)}")

    score_1, latent_1 = scores[positive_train_id], latents[positive_train_id]
    score_0, latent_0 = scores[negative_train_id], latents[negative_train_id]

    ## deal 1
    sorted_idx1 = np.argsort(score_1, axis=0)[::-1]
    logger.info(f"sorted_score1 {score_1[sorted_idx1]}")
    # threshold
    tt_idx1 = []
    for i in sorted_idx1:
        if score_1[i] > 0.7:
            tt_idx1.append(i)
    logger.info(f"pos > 0.7 {len(tt_idx1)}")
    tt_idx1 = np.array(tt_idx1)

    sorted_idx0 = np.argsort(score_0, axis=0)[::1]
    logger.info(f"sorted_score0 {score_0[sorted_idx0]}")
    # threshold
    tt_idx0 = []
    for i in sorted_idx0:
        if score_0[i] < 0.3:
            tt_idx0.append(i)
    logger.info(f"neg > 0.7 {len(tt_idx0)}")
    tt_idx0 = np.array(tt_idx0)

    positive_train = latent_1[tt_idx1[: min(len(tt_idx1), opt["train_num"])]]
    negative_train = latent_0[tt_idx0[: min(len(tt_idx0), opt["train_num"])]]
    logger.info(f"len of positive_train: {len(positive_train)}")
    logger.info(f"len of negative_train: {len(negative_train)}")

    latent_train = np.concatenate([positive_train, negative_train], axis=0)

    score_train = np.concatenate(
        [
            np.ones(len(positive_train), dtype=np.int),
            np.zeros(len(negative_train), dtype=np.int),
        ],
        axis=0,
    )
    logger.info(f'len of latent_train: {len(latent_train)}')

    latent_train, score_train = reset(latent_train, score_train)
    opt['val_num'] = len(labels) - opt['train_num']

    logger.info(f'Training boundary.')
    line_svm = svm.LinearSVC(max_iter=500000)  
    line_svm.fit(latent_train, score_train)
    logger.info(f'Finish training.')

    if opt['val_num']:
        valid(latent=latent_train, score=score_train, line_svm=line_svm, name='train')
        valid(latent=latents, score=labels, line_svm=line_svm, name='all')

    with open(opt['output_file2'], 'wb+') as handle:
        pickle.dump(line_svm, handle)

    line_svm_norm = np.linalg.norm(line_svm.coef_)
    line_svm.coef_ = line_svm.coef_ / (line_svm_norm)
    line_svm.intercept_ = line_svm.intercept_ / line_svm_norm

    save_svm = {'coef_': line_svm.coef_, 'intercept_': line_svm.intercept_}

    with open(opt['output_file'], 'wb+') as handle:
        pickle.dump(save_svm, handle)


if __name__ == '__main__':
    opt = _collect_args_boundary()
    logger = logging.getLogger('logger')
    train_equal()

