import sys
import os

import pickle

from DBR.logger import Logger as Log
Log.VERBOSE = True

import DBR.evaluation as evaluation
from DBR.plotting import *

def sort_by_config(results, configs, key):
    vals = np.array([cfg[key] for cfg in configs])
    I_vals = np.argsort(vals)

    for k in results['train'].keys():
        results['train'][k] = results['train'][k][I_vals,]
        results['valid'][k] = results['valid'][k][I_vals,]

        if k in results['test']:
            results['test'][k] = results['test'][k][I_vals,]

    configs_sorted = []
    for i in I_vals:
        configs_sorted.append(configs[i])

    return results, configs_sorted

def load_config(config_file):
    with open(config_file, 'r') as f:
        cfg = [l.split(': ') for l in f.read().split('\n') if ': ' in l]
        cfg = dict([(kv[0], eval(kv[1])) for kv in cfg])
    return cfg

def evaluate(config_file, overwrite=False, filters=None):

    if not os.path.isfile(config_file):
        raise Exception('Could not find config file at path: %s' % config_file)

    cfg = load_config(config_file)
    output_dir = cfg['outdir']

    if not os.path.isdir(output_dir):
        raise Exception('Could not find output at path: %s' % output_dir)

    data_train = cfg['datadir']+'/'+cfg['dataform']
    data_test = cfg['datadir']+'/'+cfg['data_test']
    binary = False
    if cfg['loss'] == 'log':
        binary = True

    eval_path = '%s/evaluation.npz' % output_dir
    if overwrite or (not os.path.isfile(eval_path)):
        eval_results, configs = evaluation.evaluate(output_dir,
                                data_path_train=data_train,
                                data_path_test=data_test,
                                binary=binary)
        # Save evaluation
        pickle.dump((eval_results, configs), open(eval_path, "wb"))
    else:
        if Log.VERBOSE:
            print('Loading evaluation results from %s...' % eval_path)
        # Load evaluation
        eval_results, configs = pickle.load(open(eval_path, "rb"))
    data_train_find = load_data(data_train)



    if binary:
        if data_train_find['HAVE_TRUTH']:
            plot_evaluation_cont(eval_results, configs, output_dir, data_train, data_test, filters)
        else:
            plot_evaluation_bin(eval_results, configs, output_dir, data_train, data_test, filters)
    else:
        plot_evaluation_cont(eval_results, configs, output_dir, data_train, data_test, filters)


if __name__ == "__main__":
    config_file = r'config_ihdp.txt'
    overwrite = 1
    evaluate(config_file, overwrite, filters=None)