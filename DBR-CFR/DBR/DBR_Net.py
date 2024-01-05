import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
from DBR.util import *
from DBR.distance import wasserstein
class DBR_net(object):

    def __init__(self, x, t, y_ , p_t, z_norm, FLAGS, r_lambda, do_in, do_out, dims):
        self.variables = {}
        self.wd_loss = 0

        if FLAGS.nonlin.lower() == 'elu':
            self.nonlin = tf.nn.elu
        else:
            self.nonlin = tf.nn.relu

        self.initializer = tf.contrib.layers.xavier_initializer()
        self._build_graph(x, t, y_ , p_t, z_norm, FLAGS, r_lambda, do_in, do_out, dims)


    def _add_variable(self, var, name):
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i)
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        var = self._create_variable(initializer, name)
        self.wd_loss += wd*tf.nn.l2_loss(var)
        return var

    def _feature_encoder(self, x, do_in, n_in, dim_in, dim_input):
        ''' Construct input/representation layers '''
        weights_in = []
        biases_in = []

        if FLAGS.n_in == 0 or (FLAGS.n_in == 1 and FLAGS.varsel):
            dim_in = dim_input
        if FLAGS.n_out == 0:
            if FLAGS.split_output == False:
                dim_out = dim_in + 1
            else:
                dim_out = dim_in

        if FLAGS.batch_norm:
            bn_biases = []
            bn_scales = []

        h_in = [x]

        for i in range(0, n_in):
            if i == 0:
                ''' If using variable selection, first layer is just rescaling'''
                if FLAGS.varsel:
                    weights_in.append(tf.Variable(1.0 / dim_input * tf.ones([dim_input])))
                else:
                    weights_in.append(tf.Variable(tf.random_normal([dim_input, dim_in], \
                                                                   stddev=FLAGS.weight_init / np.sqrt(dim_input))))
            else:
                weights_in.append(tf.Variable(tf.random_normal([dim_in, dim_in], \
                                                               stddev=FLAGS.weight_init / np.sqrt(dim_in))))

            ''' If using variable selection, first layer is just rescaling'''
            if FLAGS.varsel and i == 0:
                biases_in.append([])
                h_in.append(tf.mul(h_in[i], weights_in[i]))
            else:
                biases_in.append(tf.Variable(tf.zeros([1, dim_in])))
                z = tf.matmul(h_in[i], weights_in[i]) + biases_in[i]

                if FLAGS.batch_norm:
                    batch_mean, batch_var = tf.nn.moments(z, [0])

                    if FLAGS.normalization == 'bn_fixed':
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, 0, 1, 1e-3)
                    else:
                        bn_biases.append(tf.Variable(tf.zeros([dim_in])))
                        bn_scales.append(tf.Variable(tf.ones([dim_in])))
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, bn_biases[-1], bn_scales[-1], 1e-3)

                h_in.append(self.nonlin(z))
                h_in[i + 1] = tf.nn.dropout(h_in[i + 1], do_in)

        h_rep = h_in[len(h_in) - 1]

        if FLAGS.normalization == 'divide':
            h_rep_norm = h_rep / safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keep_dims=True))
        else:
            h_rep_norm = 1.0 * h_rep
        return h_rep, h_rep_norm, weights_in
    def _build_graph(self, x, t, y_ , p_t, z_norm, FLAGS, r_lambda, do_in, do_out, dims):
        self.x = x
        self.t = t
        self.y_ = y_
        self.p_t = p_t
        self.do_in = do_in
        self.do_out = do_out
        self.z_norm = z_norm
        self.r_lambda=r_lambda

        ''' input dimension'''
        dim_input = dims[0]
        ''' representation dimension'''
        dim_in = dims[1]
        ''' output dimension'''
        dim_out = dims[2]
        ''' discriminator dimension'''
        dim_d = dims[3]
        n_in=FLAGS.n_in
        n_IS = n_in//2

        ''' Construct representation layers '''
        with tf.variable_scope('encoder_A') as scope:
            h_rep_A, h_rep_norm_A, weights_in_A = self._feature_encoder(x, do_in, n_in, dim_in, dim_input)
        with tf.variable_scope('encoder_C') as scope:
            h_rep_C, h_rep_norm_C, weights_in_C = self._feature_encoder(x, do_in, n_in, dim_in, dim_input)
        with tf.variable_scope('encoder_I') as scope:
            h_rep_I, h_rep_norm_I, weights_in_I = self._feature_encoder(x, do_in, n_in, dim_in, dim_input)
        with tf.variable_scope('encoder_B') as scope:
            h_rep_B, h_rep_norm_B, weights_in_B = self._feature_encoder(h_rep_norm_C, do_in, n_in, dim_in, dim_in)

        ''' Construct prediction layers '''
        ''' predict y using A '''
        y0_f_A, y1_f_A, y_A, _, weights_out_A, weights_pred_A, _, _, _ = self._build_output_graph(
            h_rep_norm_A, t, dim_in, dim_out, do_out, FLAGS, prefix='pred_A_')
        ''' predict y using C '''
        y0_f_C, y1_f_C, y_C, _, weights_out_C, weights_pred_C, _, _, _ = self._build_output_graph(
            h_rep_norm_C, t, dim_in, dim_out, do_out, FLAGS, prefix='pred_C')

        ''' predict y using A,B '''

        h_rep_norm_AB = tf.concat([h_rep_norm_A, h_rep_norm_B], axis=1)
        y0_f_AB, y1_f_AB, y, _, _, _, _, _, _ = self._build_output_graph(
                                                h_rep_norm_AB, t, dim_in*2, dim_out, do_out, FLAGS, prefix='pred_AB')
        # h_rep_norm_AC = tf.concat([h_rep_norm_A, h_rep_norm_C], axis=1)
        # y0_f_AC, y1_f_AC, y, _, _, _, _, _, _ = self._build_output_graph(
        #                                         h_rep_norm_AC, t, dim_in*2, dim_out, do_out, FLAGS, prefix='pred_AC')
        ''' predict t using I (scope: pred_I) '''
        tpre, weights_dis, weights_discore = self._build_discriminator(h_rep_norm_I, dim_in, dim_d, do_out, FLAGS,suffix='pred_I')

        ''' predict t using C (scope: pred_I) '''
        tpre_C, weights_dis_C, weights_discore_C = self._build_discriminator(h_rep_norm_C, dim_in, dim_d, do_out, FLAGS,suffix='pred_Ct')

        ''' IPM Calculation '''
        if FLAGS.use_p_correction:
            p_ipm = self.p_t
        else:
            p_ipm = 0.5
        ''' IPM A '''
        self.imb_dist_A, self.imb_mat_A = wasserstein(h_rep_norm_A, t, p_ipm, lam=FLAGS.wass_lambda, its=FLAGS.wass_iterations,
                                        sq=False, backpropT=FLAGS.wass_bpt)
        ''' IPM C '''
        imb_dist_C, imb_mat_C = wasserstein(h_rep_norm_C, t, p_ipm, lam=FLAGS.wass_lambda, its=FLAGS.wass_iterations,
                                        sq=False, backpropT=FLAGS.wass_bpt)
        ''' IPM B'''
        imb_dist_B, imb_mat_B = wasserstein(h_rep_norm_B, t, p_ipm, lam=FLAGS.wass_lambda, its=FLAGS.wass_iterations,
                                        sq=False, backpropT=FLAGS.wass_bpt)
        ''' MI Calculation '''
        i0 = tf.to_int32(tf.where(t < 1)[:, 0])
        i1 = tf.to_int32(tf.where(t > 0)[:, 0])

        h_rep_norm_A0 = tf.gather(h_rep_norm_A, i0)
        h_rep_norm_A1 = tf.gather(h_rep_norm_A, i1)
        h_rep_norm_C0 = tf.gather(h_rep_norm_C, i0)
        h_rep_norm_C1 = tf.gather(h_rep_norm_C, i1)
        h_rep_norm_I0 = tf.gather(h_rep_norm_I, i0)
        h_rep_norm_I1 = tf.gather(h_rep_norm_I, i1)
        y0 = tf.gather(y_, i0)
        y1 = tf.gather(y_, i1)
        self.MI_IY1_lld, self.MI_IY1_bound = self.club(y1, h_rep_norm_I1, reuse=False, suffix='IY1')
        self.MI_IY0_lld, self.MI_IY0_bound = self.club(y0, h_rep_norm_I0, reuse=False, suffix='IY0')
        self.MI_AC_lld, self.MI_AC_bound = self.club(h_rep_norm_A, h_rep_norm_C, reuse=False, suffix='AC')
        self.MI_IC_lld, self.MI_IC_bound = self.club(h_rep_norm_I, h_rep_norm_C, reuse=False, suffix='IC')
        self.MI_AI_lld, self.MI_AI_bound = self.club(h_rep_norm_A, h_rep_norm_I, reuse=False, suffix='AI')


        ''' Loss Calculation'''
        """ 1. disentanglement """
        ''' A_related loss '''
        self.pred_loss_A = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_A)))
        self.pred_loss_C = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_C)))


        ''' I_related loss '''
        if FLAGS.t_pre_smooth==1:
            tpre = (tpre + 0.01) / 1.02
        if FLAGS.safelog_t == 1:
            self.pred_loss_I = -tf.reduce_mean(t * safe_log(tpre) + (1.0 - t) * safe_log(1.0 - tpre))
            self.pred_loss_C_t = -tf.reduce_mean(t * safe_log(tpre_C) + (1.0 - t) * safe_log(1.0 - tpre_C))
        else:
            self.pred_loss_I = -tf.reduce_mean(t * tf.log(tpre) + (1.0 - t) * tf.log(1.0 - tpre))
            self.pred_loss_C_t = -tf.reduce_mean(t * safe_log(tpre_C) + (1.0 - t) * safe_log(1.0 - tpre_C))


        ''' C_related loss '''
        ''' 1.orthogonal loss '''
        self.or_loss = self.MI_AC_bound + self.MI_IC_bound+self.MI_AI_bound

        """ 2. balancing loss """
        # self.bal_loss = imb_dist_C
        self.bal_loss = imb_dist_B

        ''' Construct factual loss function'''
        ''' 1. Compute sample reweighting '''
        if FLAGS.reweight_sample==1:
            w_t = t / (2 * p_t)
            w_c = (1 - t) / (2 * (1 - p_t))
            sample_weight = w_t + w_c
        else:
            sample_weight = 1.0
        self.sample_weight = sample_weight
        '''2. Construct factual loss function '''
        if FLAGS.loss == 'l1':
            self.weighted_factual_loss = tf.reduce_mean(sample_weight*tf.abs(y_-y))
            self.factual_loss = -tf.reduce_mean(tf.abs(y_-y))

        elif FLAGS.loss == 'log':
            if FLAGS.y_pre_smooth == 1:
                y = (y + 0.01) / 1.02
            if FLAGS.safelog_y == 1:
                res = y_ * safe_log(y) + (1.0 - y_) * safe_log(1.0 - y)
            else:
                res = y_*tf.log(y) + (1.0-y_)*tf.log(1.0-y)
            self.weighted_factual_loss = -tf.reduce_mean(sample_weight*res)
            self.factual_loss = -tf.reduce_mean(res)

        elif FLAGS.loss == 'mse':
            self.weighted_factual_loss = tf.reduce_mean(sample_weight*tf.square(y_ - y))
            self.factual_loss = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))

        elif FLAGS.loss == 'rmse':
            self.weighted_factual_loss = tf.sqrt(tf.reduce_mean(sample_weight * tf.square(y_ - y)))
            self.factual_loss = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))

        ''' Reps Weight Regularization '''
        if FLAGS.p_lambda > 0 and FLAGS.rep_weight_decay:
            for i in range(0, FLAGS.n_in):
                if not (FLAGS.varsel and i == 0):
                    self.wd_loss += tf.nn.l2_loss(weights_in_A[i])
                    self.wd_loss += tf.nn.l2_loss(weights_in_C[i])
                    self.wd_loss += tf.nn.l2_loss(weights_in_I[i])
                    self.wd_loss += tf.nn.l2_loss(weights_in_B[i])

        ''' Total error '''
        self.loss_A = self.pred_loss_A + FLAGS.p_alpha2 * self.imb_dist_A + FLAGS.p_lambda*self.wd_loss+FLAGS.p_alpha3 *(self.or_loss)
        self.loss_I = self.pred_loss_I + FLAGS.p_alpha1 *(self.MI_IY1_bound + self.MI_IY0_bound)+ FLAGS.p_lambda*self.wd_loss
        self.MI_I = self.MI_IY1_bound + self.MI_IY0_bound
        self.MI_I1 = self.MI_IY1_bound
        self.MI_I0 = self.MI_IY0_bound

        self.loss_C = self.pred_loss_C + self.pred_loss_C_t+FLAGS.p_alpha3 *(self.or_loss) +FLAGS.p_lambda*self.wd_loss
        # self.loss_C = FLAGS.p_alpha3 * self.recons_loss
        self.loss_B = self.weighted_factual_loss + FLAGS.p_alpha2* self.bal_loss+FLAGS.p_lambda*self.wd_loss
        # self.loss_B = self.weighted_factual_loss + FLAGS.p_alpha2 * self.bal_loss+ FLAGS.p_alpha1 *(self.or_loss) +FLAGS.p_lambda*self.wd_loss
        self.mi_estimator = -(self.MI_IY1_lld + self.MI_IY0_lld + self.MI_AI_lld + self.MI_AC_lld + self.MI_IC_lld)
        # self.mi_estimator_1 = -(self.MI_IY1_lld + self.MI_IY0_lld + self.MI_AI_lld)
        # self.mi_estimator_2 = -(self.MI_AC_lld + self.MI_IC_lld)
        self.obj_loss = self.weighted_factual_loss + self.bal_loss

        if FLAGS.varsel:
            self.w_proj = tf.placeholder("float", shape=[dim_input], name='w_proj')
            self.projection = weights_in_C[0].assign(self.w_proj)

        self.output = y
        self.h_rep_A=h_rep_A
        self.h_rep_C=h_rep_C
        self.h_rep_I=h_rep_I
        self.weights_dis = weights_dis
        self.weights_discore = weights_discore
        self.t_pre = tpre

    def _build_output(self, h_input, dim_in, dim_out, do_out, FLAGS, suffix='pred'):
        h_out = [h_input]
        dims = [dim_in] + ([dim_out]*FLAGS.n_out)
        with tf.variable_scope(suffix) as scope:
            weights_out = []; biases_out = []

            for i in range(0, FLAGS.n_out):
                wo = self._create_variable_with_weight_decay(
                        tf.random_normal([dims[i], dims[i+1]],
                            stddev=FLAGS.weight_init/np.sqrt(dims[i])),
                        'out_w_%d' % i, 1.0)
                weights_out.append(wo)

                biases_out.append(tf.Variable(tf.zeros([1,dim_out])))
                z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]

                h_out.append(self.nonlin(z))
                h_out[i+1] = tf.nn.dropout(h_out[i+1], do_out)

            weights_pred = self._create_variable(tf.random_normal([dim_out,1],
                stddev=FLAGS.weight_init/np.sqrt(dim_out)), 'w_pred')
            bias_pred = self._create_variable(tf.zeros([1]), 'b_pred')

            if FLAGS.varsel or FLAGS.n_out == 0:
                self.wd_loss += tf.nn.l2_loss(tf.slice(weights_pred,[0,0],[dim_out-1,1])) #don't penalize treatment coefficient
            else:
                self.wd_loss += tf.nn.l2_loss(weights_pred)

            ''' Construct linear classifier '''
            h_pred = h_out[-1]
            y = tf.matmul(h_pred, weights_pred)+bias_pred

        return y, h_out, weights_out, weights_pred

    def _build_output_graph(self, rep, t, dim_in, dim_out, do_out, FLAGS, prefix='predict'):
        ''' Construct output/regression layers '''
        # w_x = tf.Variable(tf.random_normal([dim_input,dim_in], stddev=FLAGS.weight_init/np.sqrt(dim_in)))
        # rep = tf.concat([rep, rep2], axis=1)
        # if FLAGS.split_output:

        i0 = tf.to_int32(tf.where(t < 1)[:,0])
        i1 = tf.to_int32(tf.where(t > 0)[:,0])

        rep0 = tf.gather(rep, i0)
        rep1 = tf.gather(rep, i1)

        y0, f0_out, weights_out0, weights_pred0 = self._build_output(rep0, dim_in, dim_out, do_out, FLAGS, suffix=prefix+'0')
        y1, f1_out, weights_out1, weights_pred1 = self._build_output(rep1, dim_in, dim_out, do_out, FLAGS, suffix=prefix+'1')

        y = tf.dynamic_stitch([i0, i1], [y0, y1])

        weights_out = weights_out0 + weights_out1
        weights_pred = weights_pred0 + weights_pred1
        # else:
        h_input = tf.concat([rep, t], 1)
        y_Slearner, f_Slearner_out, _, _ = self._build_output(h_input, dim_in+1, dim_out, do_out, FLAGS)

        return y0, y1, y, y_Slearner, weights_out, weights_pred, f0_out[1:], f1_out[1:], f_Slearner_out[1:]

    def _build_discriminator(self, hrep, dim_in, dim_d, do_out, FLAGS, reuse=False, suffix = 'pred_I'):
        ''' Construct adversarial discriminator layers '''
        h_dis = [hrep]
        with tf.variable_scope(suffix) as scope:
            if reuse:
                scope.reuse_variables()
            weights_dis = []
            biases_dis = []
            for i in range(0, FLAGS.n_dc):

                if i==0:
                    weights_dis.append(tf.Variable(tf.random_normal([dim_in,dim_d], \
                    stddev=FLAGS.weight_init/np.sqrt(dim_in))))
                else:
                    weights_dis.append(tf.Variable(tf.random_normal([dim_d,dim_d], \
                    stddev=FLAGS.weight_init/np.sqrt(dim_d))))
                biases_dis.append(tf.Variable(tf.zeros([1,dim_d])))
                z = tf.matmul(h_dis[i], weights_dis[i])+biases_dis[i]
                h_dis.append(self.nonlin(z))
                # if i != FLAGS.n_dc - 1:
                #     h_dis.append(self.nonlin(z))
                # else:
                #     h_dis.append(tf.tanh(z))
                h_dis[i + 1] = tf.nn.dropout(h_dis[i + 1], do_out)

            weights_discore = self._create_variable(tf.random_normal([dim_d,1],
                stddev=FLAGS.weight_init/np.sqrt(dim_d)), 'dc_p')
            bias_dc = self._create_variable(tf.zeros([1]), 'dc_b_p')

            h_score = h_dis[-1]
            dis_score = tf.nn.sigmoid(tf.matmul(h_score, weights_discore) + bias_dc)
            # dis_score = 0.995 / (1.0 + tf.exp(-dis_score)) + 0.0025

        return dis_score, weights_dis, weights_discore

    def club(self, input_1, input_2, reuse=False, suffix='A'):
        with tf.variable_scope('mi_net_'+suffix, reuse=reuse):
            p_0 = layers.fully_connected(inputs=input_1, num_outputs=128,
                                         activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            p_0 = layers.fully_connected(inputs=p_0, num_outputs=64,
                                         activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            prediction = layers.fully_connected(inputs=p_0, num_outputs=int(input_2.shape[1]),
                                                activation_fn=None, weights_initializer=self.initializer)

            p_1 = layers.fully_connected(inputs=input_1, num_outputs=128,
                                         activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            p_1 = layers.fully_connected(inputs=p_1, num_outputs=64,
                                         activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            prediction_1 = layers.fully_connected(inputs=p_1, num_outputs=int(input_2.shape[1]),
                                                  activation_fn=tf.nn.tanh, weights_initializer=self.initializer)

        mu = prediction
        logvar = prediction_1

        prediction_tile = tf.tile(tf.expand_dims(prediction, dim=1), [1, tf.shape(input_1)[0], 1])
        features_d_tile = tf.tile(tf.expand_dims(input_2, dim=0), [tf.shape(input_1)[0], 1, 1])

        positive = -(mu - input_2) ** 2 / 2. / tf.exp(logvar)
        negative = -tf.reduce_mean((features_d_tile - prediction_tile) ** 2, 1) / 2. / tf.exp(logvar)

        # positive_1 = -(mu - input_2) ** 2 / tf.exp(logvar)-logvar
        # lld = tf.reduce_mean(tf.reduce_sum(positive_1, -1))
        lld = tf.reduce_mean(tf.reduce_sum(positive, -1))
        bound = tf.reduce_mean(tf.reduce_sum(positive, -1) - tf.reduce_sum(negative, -1))

        return lld, bound

    def mi_net(self, input_sample, reuse = False, suffix='A'):
        with tf.variable_scope('mi_net_'+suffix, reuse=reuse):
            fc_1 = layers.fully_connected(inputs=input_sample, num_outputs=64, activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            fc_2 = layers.fully_connected(inputs=fc_1, num_outputs=128, activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            fc_3 = layers.fully_connected(inputs=fc_2, num_outputs=64, activation_fn=tf.nn.relu, weights_initializer=self.initializer)
            fc_4 = layers.fully_connected(inputs=fc_3, num_outputs=1, activation_fn=None, weights_initializer=self.initializer)
        return fc_4

    def mine(self, input_0, input_1, suffix='A'):
        tmp = tf.random_shuffle(tf.range(tf.shape(input_1)[0]))
        shuffle_d = tf.gather(input_1, tmp)
        input_1 = tf.concat([input_0, input_1], axis = -1)
        input_0 = tf.concat([input_0, shuffle_d], axis = -1)
        T_1 = self.mi_net(input_0, suffix=suffix)
        T_0 = self.mi_net(input_1, reuse=True, suffix=suffix)

        # E_pos = math.log(2.) - tf.nn.softplus(-T_0)
        E_pos = tf.reduce_mean(T_0)
        shape = tf.cast(tf.shape(input_1)[0], tf.float32)
        # E_neg = tf.nn.softplus(-T_1) + T_1 - math.log(2.)
        E_neg = tf.reduce_logsumexp(T_1)-tf.log(shape)

        bound = E_pos - E_neg
        return bound
def rep2_CT(t, rep):
    i0 = tf.to_int32(tf.where(t < 1)[:, 0])
    i1 = tf.to_int32(tf.where(t > 0)[:, 0])

    rep0 = tf.gather(rep, i0)
    rep1 = tf.gather(rep, i1)

    return tf.concat([rep0, rep1], axis=0)