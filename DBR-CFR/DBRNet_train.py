import tensorflow as tf
import numpy as np
import sys, os
import getopt
import random
import datetime
import traceback

import DBR.DBR_Net as DBRnet
from DBR.util import *
from sklearn import metrics
import joblib

''' Define parameter flags '''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('loss', 'mse', """Which loss function to use (l1/log/mse/rmse)""")
tf.app.flags.DEFINE_integer('n_in', 4, """Number of representation layers. """)
tf.app.flags.DEFINE_integer('n_out', 3, """Number of regression layers. """)
tf.app.flags.DEFINE_integer('n_dc', 3, """Number of discriminator layers. """)
tf.app.flags.DEFINE_float('p_alpha1', 0.1, """MI in I regularization. """)
tf.app.flags.DEFINE_float('p_alpha2', 0.2, """distance regularization. """)
tf.app.flags.DEFINE_float('p_alpha3', 0.1, """MI independence regularization. """)
tf.app.flags.DEFINE_float('p_lambda', 0.01, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_float('r_ror', 0.1, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_float('r_rfa', 0.1, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_float('r_ydis', 0.1, """regularization for y. """)
tf.app.flags.DEFINE_float('p_DR', 0.01, """regularization for t. """)
tf.app.flags.DEFINE_float('p_ydis', 0.1, """regularization for y. """)
tf.app.flags.DEFINE_integer('rep_weight_decay', 0, """Whether to penalize representation layers with weight decay""")
tf.app.flags.DEFINE_float('dropout_in', 1, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 1, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_string('nonlin', 'elu', """Kind of non-linearity. Default relu. """)
tf.app.flags.DEFINE_float('lrate', 1e-3, """Learning rate. """)
tf.app.flags.DEFINE_float('lr_ad', 1e-3, """Learning rate. """)
tf.app.flags.DEFINE_float('decay', 0.3, """RMSProp decay. """)
tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size. """)
tf.app.flags.DEFINE_integer('dim_in', 100, """Pre-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_out', 100, """Post-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_d', 100, """Discriminator layer dimensions. """)
tf.app.flags.DEFINE_integer('batch_norm', 0, """Whether to use batch normalization. """)
tf.app.flags.DEFINE_string('normalization', 'divide', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_integer('experiments_start', 1, """Number of experiments. """)
tf.app.flags.DEFINE_integer('experiments_end', 100, """Number of experiments. """)
tf.app.flags.DEFINE_integer('iterations', 1000, """Number of iterations. """)
tf.app.flags.DEFINE_float('weight_init', 0.1, """Weight initialization scale. """)
tf.app.flags.DEFINE_float('lrate_decay', 0.99, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
tf.app.flags.DEFINE_string('outdir', './result/ihdp', """Output directory. """)
tf.app.flags.DEFINE_string('datadir', 'D:/siyi_timeturner/disentanglement/dataset/', """Data directory. """)
tf.app.flags.DEFINE_string('dataform', 'ihdp_npci_1-1000.train.npz', """Training data filename form. """)
tf.app.flags.DEFINE_string('data_test', 'ihdp_npci_1-1000.test.npz', """Test data filename form. """)
tf.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
tf.app.flags.DEFINE_integer('seed', 1, """Seed. """)
tf.app.flags.DEFINE_integer('repetitions', 1, """Repetitions with different seed.""")
tf.app.flags.DEFINE_integer('use_p_correction', 0, """Whether to use population size p(t) in mmd/disc/wass.""")
tf.app.flags.DEFINE_integer('wass_iterations', 10, """Number of iterations in Wasserstein computation. """)
tf.app.flags.DEFINE_float('wass_lambda', 10, """Wasserstein lambda. """)
tf.app.flags.DEFINE_integer('wass_bpt', 1, """Backprop through T matrix? """)
tf.app.flags.DEFINE_string('optimizer', 'Adam', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_integer('output_csv',0,"""Whether to save a CSV file with the results""")
tf.app.flags.DEFINE_integer('output_delay', 1, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_integer('pred_output_delay', 1, """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
tf.app.flags.DEFINE_integer('save_rep', 1, """Save representations after training. """)
tf.app.flags.DEFINE_float('val_part', 0.3, """Validation part. """)
tf.app.flags.DEFINE_boolean('split_output', 1, """Whether to split output layers between treated and control. """)
tf.app.flags.DEFINE_integer('reweight_sample', 1, """Whether to reweight sample for prediction loss with average treatment probability. """)
tf.app.flags.DEFINE_integer('reweight_sample_t', 0, """Whether to reweight sample for adversarial loss with average treatment probability. """)
tf.app.flags.DEFINE_integer('NUM_ITERATIONS_PER_DECAY', 20, """iter """)
tf.app.flags.DEFINE_integer('safelog_t', 0, """ safelog t? """)
tf.app.flags.DEFINE_integer('safelog_y', 0, """ safelog y? """)
tf.app.flags.DEFINE_integer('t_pre_smooth', 1, """ smooth t? """)
tf.app.flags.DEFINE_integer('y_pre_smooth', 0, """ smooth y if y is binary? """)

if FLAGS.sparse:
    import scipy.sparse as sparse
config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config.gpu_options.per_process_gpu_memory_fraction = 0.1
tf.Session(config=config)
NUM_ITERATIONS_PER_DECAY = FLAGS.NUM_ITERATIONS_PER_DECAY


def train(DBRNet, sess, train_step_A, train_step_I_1, train_step_I_2, train_step_C_1, train_step_C_2, train_step_B, D, I_valid, D_test, logfile, i_exp):

    ''' Train/validation split '''
    n = D['x'].shape[0]
    I = range(n); I_train = list(set(I)-set(I_valid))
    n_train = len(I_train)

    ''' Compute treatment probability'''
    p_treated = np.mean(D['t'][I_train,:])
    p_treated_test = np.mean(D_test['t'])
    z_norm = np.random.normal(0.,1.,(1,FLAGS.dim_in))

    ''' Set up loss feed_dicts'''
    dict_factual = {DBRNet.x: D['x'][I_train,:], DBRNet.t: D['t'][I_train,:], DBRNet.y_: D['yf'][I_train,:], \
      DBRNet.do_in: 1.0, DBRNet.do_out: 1.0, DBRNet.r_lambda: FLAGS.p_lambda, DBRNet.p_t: p_treated, DBRNet.z_norm: z_norm}

    if FLAGS.val_part > 0:
        dict_valid = {DBRNet.x: D['x'][I_valid,:], DBRNet.t: D['t'][I_valid,:], DBRNet.y_: D['yf'][I_valid,:], \
          DBRNet.do_in: 1.0, DBRNet.do_out: 1.0, DBRNet.r_lambda: FLAGS.p_lambda, DBRNet.p_t: p_treated, DBRNet.z_norm: z_norm}

    if D['HAVE_TRUTH']:
        dict_cfactual = {DBRNet.x: D['x'][I_train,:], DBRNet.t: 1-D['t'][I_train,:], DBRNet.y_: D['ycf'][I_train,:], \
          DBRNet.do_in: 1.0, DBRNet.do_out: 1.0, DBRNet.z_norm: z_norm}

    ''' Initialize TensorFlow variables '''
    sess.run(tf.global_variables_initializer())

    ''' Set up for storing predictions '''
    preds_train = []
    preds_test = []
    tpre_train = []
    tpre_test = []
    ITE_pehe_train = []
    ITE_pehe_test = []
    DR_ITE_train = []
    DR_ITE_test = []
    ferror_test = []
    balloss_test = []

    ''' Compute losses '''
    losses = []
    obj_loss, f_error, loss_A, loss_I, loss_C = sess.run(
        [DBRNet.obj_loss, DBRNet.weighted_factual_loss, DBRNet.loss_A, DBRNet.loss_I, DBRNet.loss_C], feed_dict=dict_factual)

    cf_error = np.nan
    if D['HAVE_TRUTH']:
        cf_error = sess.run(DBRNet.factual_loss, feed_dict=dict_cfactual)

    valid_obj = np.nan; valid_f_error = np.nan
    if FLAGS.val_part > 0:
        valid_obj, valid_f_error, valid_loss_A, valid_loss_I, valid_loss_C = sess.run(
            [DBRNet.obj_loss, DBRNet.weighted_factual_loss, DBRNet.loss_A, DBRNet.loss_I, DBRNet.loss_C], feed_dict=dict_valid)

    losses.append([obj_loss, f_error, cf_error,  valid_f_error, valid_obj])

    objnan = False

    reps = []
    reps_test = []

    ''' Train for multiple iterations '''
    for i in range(FLAGS.iterations):
        I = list(range(0, n_train))
        np.random.shuffle(I)
        for i_batch in range(n_train // FLAGS.batch_size):
            if i_batch < n_train // FLAGS.batch_size - 1:
                I_b = I[i_batch * FLAGS.batch_size:(i_batch+1) * FLAGS.batch_size]
            else:
                I_b = I[i_batch * FLAGS.batch_size:]
            x_batch = D['x'][I_train,:][I_b,:]
            t_batch = D['t'][I_train,:][I_b]
            y_batch = D['yf'][I_train,:][I_b]

            z_norm_batch = np.random.normal(0.,1.,(1,FLAGS.dim_in))
            ''' Do one step of gradient descent '''
            if not objnan:
                sess.run(train_step_A, feed_dict={DBRNet.x: x_batch, DBRNet.t: t_batch, \
                                                DBRNet.y_: y_batch, DBRNet.do_in: FLAGS.dropout_in,
                                                DBRNet.do_out: FLAGS.dropout_out, \
                                                DBRNet.r_lambda: FLAGS.p_lambda, DBRNet.p_t: p_treated})
                sess.run(train_step_I_1, feed_dict={DBRNet.x: x_batch, DBRNet.t: t_batch, \
                                                DBRNet.y_: y_batch, DBRNet.do_in: FLAGS.dropout_in,
                                                DBRNet.do_out: FLAGS.dropout_out, \
                                                DBRNet.r_lambda: FLAGS.p_lambda, DBRNet.p_t: p_treated})
                sess.run(train_step_I_2, feed_dict={DBRNet.x: x_batch, DBRNet.t: t_batch, \
                                                DBRNet.y_: y_batch, DBRNet.do_in: FLAGS.dropout_in,
                                                DBRNet.do_out: FLAGS.dropout_out, \
                                                DBRNet.r_lambda: FLAGS.p_lambda, DBRNet.p_t: p_treated})
                sess.run(train_step_C_1, feed_dict={DBRNet.x: x_batch, DBRNet.t: t_batch, \
                                            DBRNet.y_: y_batch, DBRNet.do_in: FLAGS.dropout_in,
                                            DBRNet.do_out: FLAGS.dropout_out, \
                                            DBRNet.r_lambda: FLAGS.p_lambda, DBRNet.p_t: p_treated})
                sess.run(train_step_C_2, feed_dict={DBRNet.x: x_batch, DBRNet.t: t_batch, \
                                                DBRNet.y_: y_batch, DBRNet.do_in: FLAGS.dropout_in,
                                                DBRNet.do_out: FLAGS.dropout_out, \
                                                DBRNet.r_lambda: FLAGS.p_lambda, DBRNet.p_t: p_treated})
                sess.run(train_step_B, feed_dict={DBRNet.x: x_batch, DBRNet.t: t_batch, \
                                                DBRNet.y_: y_batch, DBRNet.do_in: FLAGS.dropout_in,
                                                DBRNet.do_out: FLAGS.dropout_out, \
                                                DBRNet.r_lambda: FLAGS.p_lambda, DBRNet.p_t: p_treated})
            ''' Project variable selection weights '''
            if FLAGS.varsel:
                wip = simplex_project(sess.run(DBRNet.weights_in[0]), 1)
                sess.run(DBRNet.projection, feed_dict={DBRNet.w_proj: wip})

        ''' Compute loss every N iterations '''
        if i % FLAGS.output_delay == 0 or i==FLAGS.iterations-1:
            obj_loss, f_error, IPM_A, pred_loss_A, pred_loss_C, pred_loss_C_t, pred_loss_I, MI_I, MI_I0, MI_I1, or_loss, bal_loss= sess.run(
                [DBRNet.obj_loss, DBRNet.weighted_factual_loss, DBRNet.imb_dist_A, DBRNet.pred_loss_A,
                 DBRNet.pred_loss_C, DBRNet.pred_loss_C_t,DBRNet.pred_loss_I, DBRNet.MI_I,DBRNet.MI_I0,DBRNet.MI_I1,
                 DBRNet.or_loss, DBRNet.bal_loss], feed_dict=dict_factual)

            cf_error = np.nan
            if D['HAVE_TRUTH']:
                cf_error = sess.run(DBRNet.factual_loss, feed_dict=dict_cfactual)

            valid_obj = np.nan; valid_f_error = np.nan;
            if FLAGS.val_part > 0:
                valid_obj, valid_f_error, valid_imb_dist_A, valid_loss_A,valid_loss_C_pred, valid_loss_C_t, valid_loss_I, valid_MI_I,\
                    valid_or_loss, valid_bal_loss= \
                    sess.run([DBRNet.obj_loss, DBRNet.weighted_factual_loss, DBRNet.imb_dist_A, DBRNet.pred_loss_A,
                            DBRNet.pred_loss_C, DBRNet.pred_loss_C_t, DBRNet.pred_loss_I, DBRNet.MI_I,
                            DBRNet.or_loss, DBRNet.bal_loss], \
                            feed_dict=dict_valid)

            losses.append([obj_loss, f_error, cf_error, valid_f_error, valid_obj])
            loss_str = str(i) + \
                       '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tIPM_A: %.3f,\tloss_A: %.3f, \tloss_C: %.3f,\tloss_C_t: %.3f,\tloss_I: %.3f,\tMI_I: %.3f, \tMI_I0: %.3f,\tMI_I1: %.3f,' \
                       '\tor_loss: %.3f, \tbal_loss: %.3f,\tVal: %.3f,\tVal_F: %.3f,\tVal_ipmA: %.3f, \tVal_A: %.3f, ' \
                       '\tVal_I: %.3f,  \tVal_MI_I: %.3f,  \tVal_or_loss: %.3f, \tVal_bal_loss: %.3f '\
                       % (obj_loss, f_error, cf_error, IPM_A, pred_loss_A,pred_loss_C, pred_loss_C_t, pred_loss_I, MI_I, MI_I0, MI_I1, or_loss, bal_loss,
                           valid_obj, valid_f_error, valid_imb_dist_A, valid_loss_A, valid_loss_I, valid_MI_I,
                           valid_or_loss, valid_bal_loss)

            if FLAGS.loss == 'log':
                y_pred = sess.run(DBRNet.output, feed_dict={DBRNet.x: x_batch, \
                    DBRNet.t: t_batch, DBRNet.do_in: 1.0, DBRNet.do_out: 1.0})

                fpr, tpr, thresholds = metrics.roc_curve(y_batch, y_pred)
                auc = metrics.auc(fpr, tpr)

                loss_str += ',\tAuc_batch: %.2f' % auc


            if np.isnan(obj_loss):
                log(logfile,'Experiment %d: Objective is NaN. Skipping.' % i_exp)
                objnan = True

        ''' Compute predictions every M iterations '''
        if (FLAGS.pred_output_delay > 0 and i % FLAGS.pred_output_delay == 0) or i==FLAGS.iterations-1:

            y_pred_f = sess.run(DBRNet.output,
                                feed_dict={DBRNet.x: D['x'], DBRNet.t: D['t'], DBRNet.do_in: 1.0, DBRNet.do_out: 1.0})
            y_pred_cf = sess.run(DBRNet.output,
                                 feed_dict={DBRNet.x: D['x'], DBRNet.t: 1 - D['t'], DBRNet.do_in: 1.0,
                                            DBRNet.do_out: 1.0})
            preds_train.append(np.concatenate((y_pred_f, y_pred_cf), axis=1))

            t_pre_train = sess.run(DBRNet.t_pre,
                                   feed_dict={DBRNet.x: D['x'], DBRNet.t: D['t'], DBRNet.do_in: 1.0,
                                              DBRNet.do_out: 1.0})
            tpre_train.append(t_pre_train)
            try:
                mu0_train = D['mu0']
                mu1_train = D['mu1']
            except:
                mu0_train = D['yf'] * (1 - D['t']) + D['ycf'] * D['t']
                mu1_train = D['ycf'] * (1 - D['t']) + D['yf'] * D['t']

            mu0_pre = y_pred_f * (1 - D['t']) + y_pred_cf * D['t']
            mu1_pre = y_pred_cf * (1 - D['t']) + y_pred_f * D['t']

            ITE_train = mu1_train - mu0_train
            ITE_pehe_value = np.sqrt(np.mean(np.square(ITE_train - (mu1_pre - mu0_pre))))
            DR_ITE_value = mu1_pre + D['t'] / t_pre_train * (D['yf'] - mu1_pre) - mu0_pre - (1 - D['t']) / (
                    1 - t_pre_train) * (D['yf'] - mu0_pre)
            DR_ITE_train.append(DR_ITE_value)
            ITE_pehe_train.append(ITE_pehe_value)
            loss_str += ',\tITE_pehe_train: %.3f' % ITE_pehe_value

            if FLAGS.loss == 'log' and D['HAVE_TRUTH']:
                fpr, tpr, thresholds = metrics.roc_curve(np.concatenate((D['yf'], D['ycf']),axis=0), \
                    np.concatenate((y_pred_f, y_pred_cf),axis=0))
                auc = metrics.auc(fpr, tpr)
                loss_str += ',\tAuc_train: %.2f' % auc

            if D_test is not None:
                y_pred_f_test = sess.run(DBRNet.output, feed_dict={DBRNet.x: D_test['x'], DBRNet.t: D_test['t'], DBRNet.do_in: 1.0, DBRNet.do_out: 1.0})
                y_pred_cf_test = sess.run(DBRNet.output, feed_dict={DBRNet.x: D_test['x'], DBRNet.t: 1 - D_test['t'], DBRNet.do_in: 1.0, DBRNet.do_out: 1.0})
                preds_test.append(np.concatenate((y_pred_f_test, y_pred_cf_test), axis=1))

                t_pre_test = sess.run(DBRNet.t_pre, feed_dict={DBRNet.x: D_test['x'], DBRNet.t: D_test['t'],  DBRNet.do_in: 1.0, DBRNet.do_out: 1.0})
                tpre_test.append(t_pre_test)

                f_error_test = sess.run(DBRNet.factual_loss, feed_dict={DBRNet.x: D_test['x'], DBRNet.t: D_test['t'], DBRNet.y_: D_test['yf'],  DBRNet.do_in: 1.0, DBRNet.do_out: 1.0})
                bal_loss_test = sess.run(DBRNet.bal_loss, feed_dict={DBRNet.x: D_test['x'], DBRNet.t: D_test['t'],  DBRNet.do_in: 1.0, DBRNet.do_out: 1.0})
                ferror_test.append(f_error_test)
                balloss_test.append(bal_loss_test)

                try:
                    mu0_test = D_test['mu0']
                    mu1_test = D_test['mu1']
                except:
                    mu0_test = D_test['yf'] * (1 - D_test['t']) + D_test['ycf'] * D_test['t']
                    mu1_test = D_test['ycf'] * (1 - D_test['t']) + D_test['yf'] * D_test['t']

                ITE_test = mu1_test - mu0_test
                mu0_pre_test = y_pred_f_test * (1 - D_test['t']) + y_pred_cf_test * D_test['t']
                mu1_pre_test = y_pred_cf_test * (1 - D_test['t']) + y_pred_f_test * D_test['t']
                DR_ITE_test_value = mu1_pre_test + D_test['t'] / t_pre_test * (D_test['yf'] - mu1_pre_test) - mu0_pre_test - (1 - D_test['t']) / (
                            1 - t_pre_test) * (D_test['yf'] - mu0_pre_test)


                DR_ITE_test.append(DR_ITE_test_value)
                if D['HAVE_TRUTH']:
                    if FLAGS.loss == 'log':
                        fpr, tpr, thresholds = metrics.roc_curve(np.concatenate((D_test['yf'], D_test['ycf']),axis=0), \
                            np.concatenate((y_pred_f_test, y_pred_cf_test),axis=0))
                        auc = metrics.auc(fpr, tpr)
                        loss_str += ',\tAuc_test: %.2f' % auc
                    else:
                        DR_ITE_pehe = np.sqrt(np.mean(np.square(ITE_test-(DR_ITE_test_value))))
                        ITE_pehe = np.sqrt(np.mean(np.square(ITE_test-(mu1_pre_test - mu0_pre_test))))
                        ITE_pehe_test.append(ITE_pehe)
                        loss_str += ',\tITE_pehe_test: %.3f' % ITE_pehe
                        loss_str += ',\tDR_ITE_pehe_test: %.3f' % DR_ITE_pehe

            if FLAGS.save_rep and i_exp <= 1:
                reps_i = sess.run([DBRNet.h_rep_A, DBRNet.h_rep_C, DBRNet.h_rep_I], feed_dict={DBRNet.x: D['x'], \
                    DBRNet.do_in: 1.0, DBRNet.do_out: 0.0})
                reps.append(reps_i)

                if D_test is not None:
                    reps_test_i = sess.run([DBRNet.h_rep_A, DBRNet.h_rep_C, DBRNet.h_rep_I], feed_dict={DBRNet.x: D_test['x'], \
                        DBRNet.do_in: 1.0, DBRNet.do_out: 0.0})
                    reps_test.append(reps_test_i)

        if i % 10==0:
            log(logfile, loss_str)

    return losses, preds_train, preds_test, tpre_train, tpre_test, reps, reps_test, ITE_pehe_train, ITE_pehe_test, DR_ITE_train, DR_ITE_test, ferror_test, balloss_test

def run(outdir):
    """ Runs an experiment and stores result in outdir """

    ''' Set up paths and start log '''
    npzfile = outdir+'result'
    npzfile_test = outdir+'result.test'
    logfile = outdir+'log.txt'
    f = open(logfile, 'w')
    f.close()
    dataform = FLAGS.datadir + FLAGS.dataform

    has_test = False
    if not FLAGS.data_test == '': # if test set supplied
        has_test = True
        dataform_test = FLAGS.datadir + FLAGS.data_test

    ''' Set random seeds '''
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ''' Save parameters '''
    save_config(outdir+'config.txt')
    ''' Load Data '''
    npz_input = False
    if dataform[-3:] == 'npz':
        npz_input = True
    if npz_input:
        datapath = dataform
        if has_test:
            datapath_test = dataform_test
    else:
        datapath = dataform % 1
        if has_test:
            datapath_test = dataform_test % 1

    log(logfile,     'Training data: ' + datapath)
    if has_test:
        log(logfile, 'Test data:     ' + datapath_test)
    D = load_data(datapath)
    D_test = None
    if has_test:
        D_test = load_data(datapath_test)

    log(logfile, 'Loaded data with shape [%d,%d]' % (D['n'], D['dim']))

    ''' Start Session '''
    sess = tf.Session()

    ''' Initialize input placeholders '''
    x = tf.placeholder("float", shape=[None, D['dim']], name='x') # Features
    t = tf.placeholder("float", shape=[None, 1], name='t')   # Treatent
    y_ = tf.placeholder("float", shape=[None, 1], name='y_')  # Outcome

    znorm = tf.placeholder("float", shape=[None, FLAGS.dim_in], name='z_norm')

    ''' Parameter placeholders '''
    r_lambda = tf.placeholder("float", name='r_lambda')
    do_in = tf.placeholder("float", name='dropout_in')
    do_out = tf.placeholder("float", name='dropout_out')
    p = tf.placeholder("float", name='p_treated')
    

    ''' Define model graph '''
    log(logfile, 'Defining graph...\n')
    dims = [D['dim'], FLAGS.dim_in, FLAGS.dim_out, FLAGS.dim_d]
    DBRNet = DBRnet.DBR_net(x, t, y_, p, znorm, FLAGS, r_lambda, do_in, do_out, dims)

    lr_ad = FLAGS.lr_ad
    ''' Set up optimizer '''
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(FLAGS.lrate, global_step, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    counter_enc = tf.Variable(0, trainable=False)
    lr_enc = tf.train.exponential_decay(lr_ad, counter_enc, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    counter_dc = tf.Variable(0, trainable=False)
    lr_dc = tf.train.exponential_decay(lr_ad, counter_dc, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)


    if FLAGS.optimizer == 'Adam':
        opt = tf.train.AdamOptimizer(lr)
        opt_enc = tf.train.AdamOptimizer(
            learning_rate=lr_enc, 
            beta1=0.5, 
            beta2=0.9)
        opt_dc = tf.train.AdamOptimizer(
            learning_rate=lr_dc, 
            beta1=0.5, 
            beta2=0.9)

    # var_scope_get
    var_enc_A = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_A')
    var_enc_C = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_C')
    var_enc_I = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_I')
    var_enc_B = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_B')
    var_pred_A_ = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred_A_')
    var_pred_I = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred_I')
    var_pred_C = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred_C')
    var_pred_AB = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred_AB')
    var_pred_AC = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred_AC')
    var_mi_IY = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='mi_net_IY')
    var_mi_AC = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='mi_net_AC')
    var_mi_IC = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='mi_net_IC')
    var_mi_AI = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='mi_net_AI')
    # var_recons = tf.get_collection(
    #     tf.GraphKeys.TRAINABLE_VARIABLES, scope='reconstruction')


    # print("var_enc_A:", [v.name for v in var_enc_A])
    # print()
    # print("var_enc_I:", [v.name for v in var_enc_I])
    # print()
    # print("var_enc_C:", [v.name for v in var_enc_C])
    # print()
    # print("var_pred_A_:", [v.name for v in var_pred_A_])
    # print()
    # print("var_pred_I:", [v.name for v in var_pred_I])
    # print()
    # print("var_pred_AC:", [v.name for v in var_pred_AC])
    # print()
    # print("var_recons:", [v.name for v in var_recons])
    # print()
    # print("var_mi_IY:", [v.name for v in var_mi_IY])
    # print()
    # print("var_mi_AC:", [v.name for v in var_mi_AC])
    # print()
    # print("var_mi_IC:", [v.name for v in var_mi_IC])
    # print()
    # print("var_recons:", [v.name for v in var_recons])
    # print()

    train_step_A = opt_enc.minimize(DBRNet.loss_A, global_step=counter_enc, var_list=var_enc_A + var_pred_A_)
    train_step_I_1 = opt_dc.minimize(DBRNet.mi_estimator, global_step=counter_dc, var_list=var_mi_IY+var_mi_AI)
    train_step_I_2 = opt_enc.minimize(DBRNet.loss_I, global_step=counter_enc, var_list=var_enc_I + var_pred_I)
    train_step_C_1 = opt_dc.minimize(DBRNet.mi_estimator, global_step=counter_dc, var_list=var_mi_AC + var_mi_IC)
    train_step_C_2 = opt_enc.minimize(DBRNet.loss_C, global_step=counter_enc, var_list=var_enc_C+var_pred_C)
    train_step_B = opt.minimize(DBRNet.loss_B, global_step=global_step, var_list=var_enc_B + var_pred_AB)
    # train_step_B = opt.minimize(DBRNet.loss_B, global_step=global_step, var_list=var_enc_C + var_pred_AC)


    ''' Set up for saving variables '''
    all_losses = []
    all_preds_train = []
    all_preds_test = []
    all_tpre_train = []
    all_tpre_test = []
    all_x = []
    all_t = []
    all_yf = []
    all_ycf = []
    all_valid = []
    all_rep=[]
    all_rep_test=[]
    all_pehe_train=[]
    all_pehe_test=[]
    all_ferror_test=[]
    all_balloss_test=[]
    ''' Run for all repeated experiments '''
    for i_exp in range(FLAGS.experiments_start, FLAGS.experiments_end+1):

        if FLAGS.repetitions>1:
            log(logfile, 'Training on repeated initialization %d/%d...' % (i_exp, FLAGS.repetitions))
        else:
            log(logfile, 'Training on experiment %d/%d...' % (i_exp, FLAGS.experiments_end))

        ''' Load Data (if multiple repetitions, reuse first set)'''

        if i_exp==1 or FLAGS.experiments_end>1:
            D_exp_test = None
            if npz_input:
                D_exp = {}
                D_exp['x']  = D['x'][:,:,i_exp-1]
                D_exp['t']  = D['t'][:,i_exp-1:i_exp]
                D_exp['yf'] = D['yf'][:,i_exp-1:i_exp]
                if D['HAVE_TRUTH']:
                    D_exp['ycf'] = D['ycf'][:,i_exp-1:i_exp]
                else:
                    D_exp['ycf'] = None

                if has_test:
                    D_exp_test = {}
                    D_exp_test['x']  = D_test['x'][:,:,i_exp-1]
                    D_exp_test['t']  = D_test['t'][:,i_exp-1:i_exp]
                    D_exp_test['yf'] = D_test['yf'][:,i_exp-1:i_exp]
                    if D_test['HAVE_TRUTH']:
                        try:
                            D_exp_test['mu0'] = D_test['mu0'][:,i_exp-1:i_exp]
                            D_exp_test['mu1'] = D_test['mu1'][:, i_exp - 1:i_exp]
                            D_exp_test['ycf'] = D_test['ycf'][:, i_exp - 1:i_exp]
                        except:
                            D_exp_test['ycf'] = D_test['ycf'][:,i_exp-1:i_exp]
                    else:
                        D_exp_test['ycf'] = None
            else:
                datapath = dataform % i_exp
                D_exp = load_data(datapath)
                if has_test:
                    datapath_test = dataform_test % i_exp
                    D_exp_test = load_data(datapath_test)

            D_exp['HAVE_TRUTH'] = D['HAVE_TRUTH']
            if has_test:
                D_exp_test['HAVE_TRUTH'] = D_test['HAVE_TRUTH']

        ''' Split into training and validation sets '''
        I_train, I_valid = validation_split(D_exp, FLAGS.val_part)

        ''' Run training loop '''
        losses, preds_train, preds_test, tpre_train, tpre_test, reps, reps_test, ITE_pehe_train, ITE_pehe, DR_ITE, DR_ITE_test, ferror_test, balloss_test= \
            train(DBRNet, sess, train_step_A, train_step_I_1, train_step_I_2, train_step_C_1, train_step_C_2, train_step_B, D_exp, I_valid, \
                D_exp_test, logfile, i_exp)

        ''' Collect all reps '''
        all_preds_train.append(preds_train)
        all_preds_test.append(preds_test)
        all_tpre_train.append(tpre_train)
        all_tpre_test.append(tpre_test)
        all_x.append(D_exp['x'])
        all_t.append(D_exp['t'])
        all_yf.append(D_exp['yf'])
        all_ycf.append(D_exp['ycf'])
        all_losses.append(losses)
        all_rep.append(reps)
        all_rep_test.append(reps_test)
        all_pehe_test.append(ITE_pehe)
        all_pehe_train.append(ITE_pehe_train)
        all_ferror_test.append(ferror_test)
        all_balloss_test.append(balloss_test)


        ''' Fix shape for output (n_units, dim, n_reps, n_outputs) '''
        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train, 1, 3), 0, 2)
        out_tpre_train = np.swapaxes(np.swapaxes(all_tpre_train, 1, 3), 0, 2)

        if has_test:
            out_preds_test = np.swapaxes(np.swapaxes(all_preds_test, 1, 3), 0, 2)
            out_tpre_test = np.swapaxes(np.swapaxes(all_tpre_test, 1, 3), 0, 2)
            # out_ferror_test = np.swapaxes(np.swapaxes(all_ferror_test, 1, 3), 0, 2)
            # out_balloss_test = np.swapaxes(np.swapaxes(all_balloss_test, 1, 3), 0, 2)


        out_losses = np.swapaxes(np.swapaxes(all_losses, 0, 2), 0, 1)

        ''' Store predictions '''
        log(logfile, 'Saving result to %s...\n' % outdir)
        # if FLAGS.output_csv:
        #     np.savetxt('%s_%d.csv' % (outform, i_exp), preds_train[-1], delimiter=',')
        #     np.savetxt('%s_%d.csv' % (outform_test, i_exp), preds_test[-1], delimiter=',')
        #     np.savetxt('%s_%d.csv' % (lossform, i_exp), losses, delimiter=',')
        #
        # ''' Compute weights if doing variable selection '''
        # if FLAGS.varsel:
        #     if i_exp == 1:
        #         all_weights = sess.run(PALNet.weights_in[0])
        #         all_beta = sess.run(PALNet.weights_pred)
        #     else:
        #         all_weights = np.dstack((all_weights, sess.run(PALNet.weights_in[0])))
        #         all_beta = np.dstack((all_beta, sess.run(PALNet.weights_pred)))

        ''' Save results and predictions '''
        all_valid.append(I_valid)
        if FLAGS.varsel:
            np.savez(npzfile, pred=out_preds_train, tpre=out_tpre_train, x=all_x, t=all_t, yf=all_yf, ycf = all_ycf, rep =all_rep, rep_test =all_rep_test, w=all_weights, beta=all_beta, val=np.array(all_valid))
        else:
            np.savez(npzfile, pred=out_preds_train, tpre=out_tpre_train, x=all_x, t=all_t, yf=all_yf, ycf = all_ycf, rep=all_rep, loss=out_losses, val=np.array(all_valid),pehe_train=all_pehe_train)

        if has_test:
            np.savez(npzfile_test, pred=out_preds_test, tpre=out_tpre_test, rep_test=all_rep_test, pehe_test=all_pehe_test, ferror=all_ferror_test, bal_test=all_balloss_test)
def main(argv=None):  # pylint: disable=unused-argument
    """ Main entry point """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = FLAGS.outdir+'/results_'+timestamp+'/'
    # outdir = FLAGS.outdir + '/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    try:
        run(outdir)
    except Exception as e:
        with open(outdir+'error.txt','w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise


if __name__ == '__main__':
    tf.app.run()
