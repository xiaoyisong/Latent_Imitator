import argparse
import os
import sys
import numpy as np
import pandas as pd
from sdv.metrics.tabular import CSTest, KSComplement
from sdmetrics.reports.single_table import QualityReport
from sdv.demo import load_tabular_demo
from utils import utils_base
from utils.logger import setup_logger

EXP_DIR = "./exp"


def _parse_args():
    parser = argparse.ArgumentParser(description="evaluate naturalness")
    parser.add_argument("--experiment", choices=['natural'], default="natural")
    parser.add_argument("--dataset", type=str, default="census")
    parser.add_argument("--exp_name", type=str, default="census")
    parser.add_argument("--record", type=str, default="xx.txt")
    parser.add_argument("--method", type=str, default="adf")
    parser.add_argument("--path", type=str, default="census")
    parser.add_argument("--random_state", type=int, default=2333)
    opt = vars(parser.parse_args())

    opt["expdir"] = os.path.join(EXP_DIR, opt['experiment'], opt["exp_name"])
    utils_base.make_dir(opt["expdir"])
    logger = open(os.path.join(opt["expdir"], 'log.txt'), mode="a+")

    record = open(os.path.join(opt["expdir"], opt['record']), mode="a+")

    sys.stdout = logger
    return opt, logger, record


def load_data(path, opt, flag=False):
    dataset_name = opt['dataset']
    if dataset_name == 'census':
        _str = 'a,b,c,d,e,f,g,h,i,j,k,l,m'
    elif dataset_name == 'credit':
        _str = 'a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t'
    elif dataset_name == 'bank':
        _str = 'a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p'
    elif dataset_name == 'meps':
        _str = 'REGION,AGE,SEX,RACE,MARRY,FTSTU,ACTDTY,HONRDC,RTHLTH,MNHLTH,CHDDX,ANGIDX,MIDX,OHRTDX,STRKDX,EMPHDX,CHBRON,CHOLDX,CANCERDX,DIABDX,JTPAIN,ARTHDX,ARTHTYPE,ASTHDX,ADHDADDX,PREGNT,WLKLIM,ACTLIM,SOCLIM,COGLIM,DFHEAR42,DFSEE42,ADSMOK42,PCS42,MCS42,K6SUM42,PHQ242,EMPST,POVCAT,INSCOV'

    names = _str.split(',')
    print(names)
    if flag:
        data = pd.read_csv(path, header=None, names=names)
    else:
        data = pd.read_csv(path, header='infer')
    print(data.head())
    return data


def _print_results(report: QualityReport, out=sys.stdout):
    _ans = ''
    """Print the quality report results."""
    if np.isnan(report._overall_quality_score) & any(report._property_errors.values()):
        out.write('\nOverall Quality Score: Error computing report.\n\n')
    else:
        out.write(
            f'\nOverall Quality Score: {round(report._overall_quality_score * 100, 2)}%\n\n'
        )
        _ans += (
            f"Overall Quality Score: {round(report._overall_quality_score * 100, 2)}%\n"
        )

    if len(report._property_breakdown) > 0:
        out.write('Properties:\n')

    for prop, score in report._property_breakdown.items():
        if not np.isnan(score):
            out.write(f'{prop}: {round(score * 100, 2)}%\n')
            _ans += f"{prop}: {round(score * 100, 2)}%\n"
        elif report._property_errors[prop] > 0:
            out.write(f'{prop}: Error computing property.\n')
        else:
            out.write(f'{prop}: NaN\n')

    return _ans


def compute(opt, logger, record):
    dataset_name = opt['dataset']
    ### prepare metadata_dict
    if dataset_name == 'census':
        metadata_dict = {
            'fields': {
                'a': {'type': 'numerical'},
                'b': {'type': 'categorical'},
                'c': {'type': 'numerical'},
                'd': {'type': 'categorical'},
                'e': {'type': 'categorical'},
                'f': {'type': 'categorical'},
                'g': {'type': 'categorical'},
                'h': {'type': 'categorical'},
                'i': {'type': 'categorical'},
                'j': {'type': 'numerical'},
                'k': {'type': 'numerical'},
                'l': {'type': 'numerical'},
                'm': {'type': 'categorical'},
            }
        }
    elif dataset_name == 'credit':
        metadata_dict = {
            'fields': {
                'a': {'type': 'categorical'},
                'b': {'type': 'numerical'},
                'c': {'type': 'categorical'},
                'd': {'type': 'categorical'},
                'e': {'type': 'numerical'},
                'f': {'type': 'categorical'},
                'g': {'type': 'categorical'},
                'h': {'type': 'numerical'},
                'i': {'type': 'categorical'},
                'j': {'type': 'categorical'},
                'k': {'type': 'numerical'},
                'l': {'type': 'categorical'},
                'm': {'type': 'numerical'},
                'n': {'type': 'categorical'},
                'o': {'type': 'categorical'},
                'p': {'type': 'numerical'},
                'q': {'type': 'categorical'},
                'r': {'type': 'numerical'},
                's': {'type': 'categorical'},
                't': {'type': 'categorical'},
            }
        }
    elif dataset_name == 'bank':
        metadata_dict = {
            'fields': {
                'a': {'type': 'numerical'},
                'b': {'type': 'categorical'},
                'c': {'type': 'categorical'},
                'd': {'type': 'categorical'},
                'e': {'type': 'categorical'},
                'f': {'type': 'numerical'},
                'g': {'type': 'categorical'},
                'h': {'type': 'categorical'},
                'i': {'type': 'categorical'},
                'j': {'type': 'categorical'},
                'k': {'type': 'categorical'},
                'l': {'type': 'numerical'},
                'm': {'type': 'numerical'},
                'n': {'type': 'numerical'},
                'o': {'type': 'numerical'},
                'p': {'type': 'categorical'},
            }
        }
    elif dataset_name == 'meps':
        metadata_dict = {
            'fields': {
                'REGION': {'type': 'categorical'},
                'AGE': {'type': 'numerical'},
                'SEX': {'type': 'categorical'},
                'RACE': {'type': 'numerical'},
                'MARRY': {'type': 'categorical'},
                'FTSTU': {'type': 'categorical'},
                'ACTDTY': {'type': 'categorical'},
                'HONRDC': {'type': 'categorical'},
                'RTHLTH': {'type': 'categorical'},
                'MNHLTH': {'type': 'categorical'},
                'CHDDX': {'type': 'categorical'},
                'ANGIDX': {'type': 'categorical'},
                'MIDX': {'type': 'categorical'},
                'OHRTDX': {'type': 'categorical'},
                'STRKDX': {'type': 'categorical'},
                'EMPHDX': {'type': 'categorical'},
                'CHBRON': {'type': 'categorical'},
                'CHOLDX': {'type': 'categorical'},
                'CANCERDX': {'type': 'categorical'},
                'DIABDX': {'type': 'categorical'},
                'JTPAIN': {'type': 'categorical'},
                'ARTHDX': {'type': 'categorical'},
                'ARTHTYPE': {'type': 'categorical'},
                'ASTHDX': {'type': 'categorical'},
                'ADHDADDX': {'type': 'categorical'},
                'PREGNT': {'type': 'categorical'},
                'WLKLIM': {'type': 'categorical'},
                'ACTLIM': {'type': 'categorical'},
                'SOCLIM': {'type': 'categorical'},
                'COGLIM': {'type': 'categorical'},
                'DFHEAR42': {'type': 'categorical'},
                'DFSEE42': {'type': 'categorical'},
                'ADSMOK42': {'type': 'categorical'},
                'PCS42': {'type': 'numerical'},
                'MCS42': {'type': 'numerical'},
                'K6SUM42': {'type': 'numerical'},
                'PHQ242': {'type': 'categorical'},
                'EMPST': {'type': 'categorical'},
                'POVCAT': {'type': 'categorical'},
                'INSCOV': {'type': 'categorical'},
            }
        }
    baseroot = '../LIMI_tabular/'

    if dataset_name == 'census':
        path1 = os.path.join(baseroot, 'datasets/census_train.csv')
    elif dataset_name == 'credit':
        path1 = os.path.join(baseroot, 'datasets/credit_train.csv')
    elif dataset_name == 'bank':
        path1 = os.path.join(baseroot, 'datasets/bank_train.csv')
    elif dataset_name == 'meps':
        path1 = os.path.join(baseroot, 'datasets/meps_train.csv')

    _str = opt['method'] + ':'
    logger.write(_str + '\n')
    print('load raw')
    real_data = load_data(path1, opt)

    path2 = opt['path']
    print(f"load eval_path from {opt['method']} + ':'")
    synthetic_data = load_data(path2, opt, flag=True)

    print(f'real data length {len(real_data)}')
    if len(synthetic_data) > len(real_data):
        print(f'synthetic_data data length {len(synthetic_data)}')
        synthetic_data = synthetic_data.sample(
            n=len(real_data), random_state=opt['random_state']
        )
    report = QualityReport()
    report.generate(real_data, synthetic_data, metadata_dict)
    _res = _print_results(report, logger)
    _detail = report.get_details(property_name='Column Shapes')
    logger.write(f"\n{_detail}\n")
    logger.write('----------------spilt_line----------------' + '\n\n')

    _str = opt['method'] + ':'
    record.write(_str + '\n')
    record.write(_res)
    record.write('----------------spilt_line----------------' + '\n\n')


if __name__ == '__main__':
    opt, logger, record = _parse_args()
    if opt['experiment'] == 'natural':
        compute(opt, logger, record)
