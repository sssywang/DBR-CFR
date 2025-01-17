import sys
import os
import numpy as np
import matplotlib as mpl
import pickle
mpl.use('Agg')
import matplotlib.pyplot as plt
# from MBRL.evaluation import eval_ate
from DBR.loader import *

LINE_WIDTH = 2
FONTSIZE_LGND = 8
FONTSIZE = 16

EARLY_STOP_SET_CONT = 'valid'
EARLY_STOP_CRITERION_CONT = 'rmse_fact'
CONFIG_CHOICE_SET_CONT = 'valid'
CONFIG_CRITERION_CONT = 'pehe'
CORR_CRITERION_CONT = 'pehe'
CORR_CHOICE_SET_CONT = 'test'

EARLY_STOP_SET_BIN = 'valid'
EARLY_STOP_CRITERION_BIN = 'rmse_fact'
CONFIG_CHOICE_SET_BIN = 'valid'
CONFIG_CRITERION_BIN = 'bias_ate'
CORR_CRITERION_BIN = 'bias_ate'
CORR_CHOICE_SET_BIN = 'test'

CURVE_TOP_K = 7

def fix_log_axes(x):
    ax = plt.axes()
    plt.draw()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[1] = r'0'
    ax.set_xticklabels(labels)
    d=0.025
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((0.04-0.25*d, 0.04+0.25*d), (-d, +d), **kwargs)
    ax.plot((0.06-0.25*d, 0.06+0.25*d), (-d, +d), **kwargs)
    plt.xlim(np.min(x), np.max(x))

def plot_format():
    plt.grid(linestyle='-', color=[0.8,0.8,0.8])
    ax = plt.gca()
    ax.set_axisbelow(True)

def fill_bounds(data, axis=0, std_error=False):
    if std_error:
        dev = np.std(data, axis)/np.sqrt(data.shape[axis])
    else:
        dev = np.std(data, axis)

    ub = np.mean(data, axis) + dev
    lb = np.mean(data, axis) - dev

    return lb, ub

def plot_with_fill(x, y, axis=0, std_error=False, color='r'):
    plt.plot(x, np.mean(y, axis), '.-', linewidth=2, color=color)
    lb, ub = fill_bounds(y, axis=axis, std_error=std_error)
    plt.fill_between(x, lb, ub, linewidth=0, facecolor=color, alpha=0.1)

def cap(s):
    t = s[0].upper() + s[1:]
    return t

def table_str_bin(result_set, row_labels, labels_long=None, binary=False):
    if binary:
        cols = ['pehe', 'bias_ate','bias_ate_dml', 'rmse_fact', 'objective', 'auc', 'fact_auc']
    else:
        cols = ['pehe', 'DR_pehe', 'bias_ate', 'rmse_fact', 'rmse_fact2', 'objective']

    cols = [c for c in cols if c in result_set[0]]

    head = [cap(c) for c in cols]
    colw = np.max([16, np.max([len(h)+1 for h in head])])
    col1w = np.max([len(h)+1 for h in row_labels])

    def rpad(s):
        return s+' '*(colw-len(s))

    def r1pad(s):
        return s+' '*(col1w-len(s))

    head_pad = [r1pad('')]+[rpad(h) for h in head]

    head_str = '| '.join(head_pad)
    s = head_str + '\n' + '-'*len(head_str) + '\n'

    for i in range(len(result_set)):
        vals = [np.mean(np.abs(result_set[i][c])) for c in cols]
        stds = [np.std(result_set[i][c])/np.sqrt(result_set[i][c].shape[0]) for c in cols]
        val_pad = [r1pad(row_labels[i])] + [rpad('%.3f +/- %.3f ' % (vals[j], stds[j])) for j in range(len(vals))]
        val_str = '| '.join(val_pad)

        if labels_long is not None:
            s += labels_long[i] + '\n'

        s += val_str + '\n'

    return s

def evaluation_summary(result_set, row_labels, output_dir, labels_long=None, binary=False):
    s = ''
    for i in ['train', 'valid', 'test']:
        s += 'Mode: %s\n' % cap(i)
        s += table_str_bin([results[i] for results in result_set], row_labels, labels_long, binary)
        s += '\n'

    return s

def select_parameters(output_dir, results, configs, stop_set, stop_criterion, choice_set, choice_criterion):

    if stop_criterion == 'objective' and 'objective' not in results[stop_set]:
        if 'err_fact' in results[stop_set]:
            stop_criterion = 'err_fact'
        else:
            stop_criterion = 'rmse_fact'

    ''' Select early stopping for each repetition '''
    n_exp = results[stop_set][stop_criterion].shape[1]
    if stop_criterion in ['auc', 'fact_auc', 'pert_fact_auc']:
        i_sel = np.argmax(results[stop_set][stop_criterion],2)
    else:
        i_sel = np.argmin(results[stop_set][stop_criterion],2)
    results_sel = {'train': {}, 'valid': {}, 'test': {}}

    for k in results['valid'].keys():
        # To reduce dimension
        results_sel['train'][k] = np.sum(results['train'][k],2)
        results_sel['valid'][k] = np.sum(results['valid'][k],2)

        if k in results['test']:
            results_sel['test'][k] = np.sum(results['test'][k],2)

        for ic in range(len(configs)):
            for ie in range(n_exp):
                results_sel['train'][k][ic,ie,] = results['train'][k][ic,ie,i_sel[ic,ie],]
                results_sel['valid'][k][ic,ie,] = results['valid'][k][ic,ie,i_sel[ic,ie],]

                if k in results['test']:
                    results_sel['test'][k][ic,ie,] = results['test'][k][ic,ie,i_sel[ic,ie],]
    # results = load_results(output_dir)
    # for ic in range(len(configs)):
    #     cfg = configs[ic]
    #     data_path_train = cfg['datadir'] + '/' + cfg['dataform']
    #     data_path_test = cfg['datadir'] + '/' + cfg['data_test']
    #     data_train = load_data(data_path_train)
    #     data_test = load_data(data_path_test)
    #     result = results[ic]
    #     for ie in range(n_exp):
    #         ate_dml_train, ate_rcl_train_1, ate_rcl_train_2, ate_dml_valid, ate_rcl_valid_1, ate_rcl_valid_2, ate_dml_test, ate_rcl_test_1, ate_rcl_test_2 = eval_ate(result, data_train, data_test, ie, i_sel[ic,ie])
    #         results_sel['train']['bias_ate_dml'][ic,ie,] = ate_dml_train
    #         results_sel['valid']['bias_ate_dml'][ic,ie,] = ate_dml_valid
    #         results_sel['test']['bias_ate_dml'][ic,ie,] = ate_dml_test
    #         results_sel['train']['bias_ate_rcl_1'][ic,ie,] = ate_rcl_train_1
    #         results_sel['valid']['bias_ate_rcl_1'][ic,ie,] = ate_rcl_valid_1
    #         results_sel['test']['bias_ate_rcl_1'][ic,ie,] = ate_rcl_test_1
    #         results_sel['train']['bias_ate_rcl_2'][ic,ie,] = ate_rcl_train_2
    #         results_sel['valid']['bias_ate_rcl_2'][ic,ie,] = ate_rcl_valid_2
    #         results_sel['test']['bias_ate_rcl_2'][ic,ie,] = ate_rcl_test_2
    pickle.dump(results_sel, open(output_dir + '/' + 'results_sel.npz', 'wb'))
    pickle.dump(i_sel, open(output_dir + '/' +'i_sel.npz', 'wb'))
    print('Early stopping:')
    print(np.mean(i_sel,1))

    ''' Select configuration '''
    results_all = [dict([(k1, dict([(k2, v[i,]) for k2,v in results_sel[k1].items()]))
                        for k1 in results_sel.keys()]) for i in range(len(configs))]

    labels = ['%d' % i for i in range(len(configs))]

    sort_key = np.argsort([np.mean(r[choice_set][choice_criterion]) for r in results_all])
    results_all = [results_all[i] for i in sort_key]
    configs_all = [configs[i] for i in sort_key]
    labels = [labels[i] for i in sort_key]

    return results_all, configs_all, labels, sort_key

def plot_option_correlation(output_dir, diff_opts, results, configs,
        choice_set, choice_criterion, filter_str=''):

    topk = int(np.min([CURVE_TOP_K, len(configs)]))

    opts_dir = '%s/opts%s' % (output_dir, filter_str)

    try:
        os.mkdir(opts_dir)
    except:
        pass

    for k in diff_opts:

        x_range = sorted(list(set([configs[i][k] for i in range(len(configs))])))

        x_range_bins = [None]*len(x_range)
        x_range_bins_top = [None]*len(x_range)

        plt.figure()
        for i in range(0, len(configs)):
            x = x_range.index(configs[i][k])
            y = np.mean(results[i][choice_set][choice_criterion])

            if x_range_bins[x] is None:
                x_range_bins[x] = []
            x_range_bins[x].append(y)

            plt.plot(x + 0.2*np.random.rand()-0.1, y , 'ob')

        for i in range(topk):
            x = x_range.index(configs[i][k])
            y = np.mean(results[i][choice_set][choice_criterion])

            if x_range_bins_top[x] is None:
                x_range_bins_top[x] = []
            x_range_bins_top[x].append(y)

            plt.plot(x + 0.2*np.random.rand()-0.1, y , 'og')

        for i in range(len(x_range)):
            m1 = np.mean(x_range_bins[i])
            plt.plot([i-0.2, i+0.2], [m1, m1], 'r', linewidth=LINE_WIDTH)

            if x_range_bins_top[i] is not None:
                m2 = np.mean(x_range_bins_top[i])
                plt.plot([i-0.1, i+0.1], [m2, m2], 'g', linewidth=LINE_WIDTH)

        plt.xticks(range(len(x_range)), x_range)
        plt.title(r'$\mathrm{Influence\/of\/%s\/on\/%s\/on\/%s}$' % (k, choice_criterion, choice_set))
        plt.ylabel('%s' % (choice_criterion))
        plt.xlabel('options')
        plt.xlim(-0.5, len(x_range)-0.5)
        plt.savefig('%s/opt.%s.%s.%s.pdf' % (opts_dir, choice_set, choice_criterion, k))
        plt.close()

def plot_evaluation_cont(results, configs, output_dir, data_train_path, data_test_path, filters=None):
    data_train = load_data(data_train_path)
    data_test = load_data(data_test_path)

    propensity = {}
    propensity['train'] = np.mean(data_train['t'])
    propensity['valid'] = np.mean(data_train['t'])
    propensity['test'] = np.mean(data_test['t'])

    ''' Select by filter '''
    filter_str = ''
    if filters is not None:
        filter_str = '.'+'.'.join(['%s.%s' % (k,filters[k]) for k in sorted(filters.keys())])

        N = len(configs)
        I = [i for i in range(N) if np.all( \
                [configs[i][k]==filters[k] for k in filters.keys()] \
            )]

        results = dict([(s,dict([(k,results[s][k][I,]) for k in results[s].keys()])) for s in ['train', 'valid', 'test']])
        configs = [configs[i] for i in I]

    ''' Do parameter selection and early stopping '''
    results_all, configs_all, labels, sort_key = select_parameters(output_dir, results,
        configs, EARLY_STOP_SET_CONT, EARLY_STOP_CRITERION_CONT,
        CONFIG_CHOICE_SET_CONT, CONFIG_CRITERION_CONT)

    ''' Save sorted configurations by parameters that differ '''
    diff_opts = sorted([k for k in configs[0] if len(set([cfg[k] for cfg in configs]))>1])
    labels_long = [str(i) + ' | ' +', '.join(['%s=%s' % (k,str(configs[i][k])) for k in diff_opts]) for i in sort_key]

    with open('%s/configs_sorted%s_%s.txt' % (output_dir, filter_str, EARLY_STOP_CRITERION_CONT), 'w') as f:
        f.write('\n'.join(labels_long))

    ''' Compute evaluation summary and store'''
    eval_str = evaluation_summary(results_all, labels, output_dir, binary=False)

    with open('%s/results_summary%s_%s.txt' % (output_dir, filter_str, EARLY_STOP_CRITERION_CONT), 'w') as f:
        f.write('Selected early stopping based on individual \'%s\' on \'%s\'\n' % (EARLY_STOP_CRITERION_CONT, EARLY_STOP_SET_CONT))
        f.write('Selected configuration based on mean \'%s\' on \'%s\'\n' % (CONFIG_CRITERION_CONT, CONFIG_CHOICE_SET_CONT))
        f.write(eval_str)

    ''' Plot option correlation '''
    plot_option_correlation(output_dir, diff_opts, results_all, configs_all,
        CORR_CHOICE_SET_CONT, CORR_CRITERION_CONT, filter_str)


def plot_evaluation_bin(results, configs, output_dir, data_train_path, data_test_path, filters=None, epsilon=0, early_stop_criterion='rmse_fact'):
    EARLY_STOP_CRITERION_BIN = early_stop_criterion
    data_train = load_data(data_train_path)
    data_test = load_data(data_test_path)

    propensity = {}
    propensity['train'] = np.mean(data_train['t'][data_train['e']==1,])
    propensity['valid'] = np.mean(data_train['t'][data_train['e']==1,])
    propensity['test'] = np.mean(data_test['t'][data_test['e']==1,])

    ''' Select by filter '''
    filter_str = ''
    if filters is not None:
        filter_str = '.'+'.'.join(['%s.%s' % (k,filters[k]) for k in sorted(filters.keys())])

        def cmp(u,v):
            if isinstance(u, basestring):
                return u.lower()==v.lower()
            else:
                return u==v

        N = len(configs)
        I = [i for i in range(N) if np.all( \
                [cmp(configs[i][k],filters[k]) for k in filters.keys()] \
            )]

        results = dict([(s,dict([(k,results[s][k][I,]) for k in results[s].keys()])) for s in ['train', 'valid', 'test']])
        configs = [configs[i] for i in I]

    ''' Do parameter selection and early stopping '''
    results_all, configs_all, labels, sort_key = select_parameters(output_dir, results,
        configs, EARLY_STOP_SET_BIN, EARLY_STOP_CRITERION_BIN,
        CONFIG_CHOICE_SET_BIN, CONFIG_CRITERION_BIN)

    ''' Save sorted configurations by parameters that differ '''
    diff_opts = sorted([k for k in configs[0] if len(set([cfg[k] for cfg in configs]))>1])
    labels_long = [str(i) + ' | ' +', '.join(['%s=%s' % (k,str(configs[i][k])) for k in diff_opts]) for i in sort_key]

    with open('%s/configs_sorted%s_%s_%s.txt' % (output_dir, filter_str, early_stop_criterion, str(epsilon)), 'w') as f:
        f.write('\n'.join(labels_long))

    ''' Compute evaluation summary and store'''
    eval_str = evaluation_summary(results_all, labels, output_dir, binary=True)

    with open('%s/results_summary%s_%s_%s.txt' % (output_dir, filter_str, early_stop_criterion, str(epsilon)), 'w') as f:
        f.write('Selected early stopping based on individual \'%s\' on \'%s\'\n' % (EARLY_STOP_CRITERION_BIN, EARLY_STOP_SET_BIN))
        f.write('Selected configuration based on mean \'%s\' on \'%s\'\n' % (CONFIG_CRITERION_BIN, CONFIG_CHOICE_SET_BIN))
        f.write(eval_str)

    ''' Policy curve for top-k configurations '''
    colors = 'rgbcmyk'
    topk = int(np.min([CURVE_TOP_K, len(configs)]))
    try:
        for eval_set in ['train', 'valid', 'test']:
            pc = np.mean(results_all[0][eval_set]['policy_curve'],0)
            x = np.array(range(len(pc))).astype(np.float32)/(len(pc)-1)
            for i in range(topk):
                plot_with_fill(x, results_all[i][eval_set]['policy_curve'], axis=0, std_error=True, color=colors[i])
            plt.plot([0,1], [pc[0], pc[-1]], '--k', linewidth=2)


            p = propensity[eval_set]
            x_lim = plt.xlim()
            y_lim = plt.ylim()
            plt.plot([p,p], y_lim, ':k')
            plt.text(p+0.01*(x_lim[1]-x_lim[0]),y_lim[0]+0.05*(y_lim[1]-y_lim[0]), r'$p(t)$', fontsize=14)
            plt.ylim(y_lim)

            plt.xlabel(r'$\mathrm{Inclusion\/rate}$', fontsize=FONTSIZE)
            plt.ylabel(r'$\mathrm{Policy\/value}$', fontsize=FONTSIZE)
            plt.title(r'$\mathrm{Policy\/curve\/%s\/(w.\/early\/stopping)}$' % eval_set)
            plt.legend(['Configuration %d' % i for i in range(topk)], fontsize=FONTSIZE_LGND)
            plot_format()
            plt.savefig('%s/policy_curve%s.%s.pdf' % (output_dir, filter_str, eval_set))
            plt.close()
    except:
        print('No policy curve')
    ''' Plot option correlation '''
    plot_option_correlation(output_dir, diff_opts, results_all, configs_all,
        CORR_CHOICE_SET_BIN, CORR_CRITERION_BIN, filter_str)
