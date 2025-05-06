# coding=UTF-8
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from code.utils import layers
from code.gat_model.base_gattn import BaseGAttN


class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                          out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                              out_sz=hid_units[i], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                        out_sz=nb_classes, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]

        return logits


class HeteGAT_multi(BaseGAttN):
    def __init__(self, params):
        self.relation_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = 1
        self.hidden_size = 50
        self.batch_size = params['batch_size'] * params['num_rollouts']
        with tf.variable_scope("gat_step"):
            cells = []
            for _ in range(self.LSTM_Layers):
                cells.append(tf.nn.rnn_cell.LSTMCell(6 * self.hidden_size, use_peepholes=True, state_is_tuple=True,
                                                       name='gat_lstm_cell'))
            self.rnn_step = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

    def inference1(self, input, entity_table, relation_table, time_in_x, bias_mat, mask_mat, nb_nodes, nb_input):
        rk = tf.nn.embedding_lookup(relation_table, bias_mat)
        ei = tf.expand_dims(input, axis=1)
        ei = tf.tile(ei, [1, nb_nodes, 1])
        ej = tf.expand_dims(entity_table, axis=0)
        ej = tf.tile(ej, [nb_input, 1, 1])
        t = tf.expand_dims(time_in_x, axis=0)
        t = tf.expand_dims(t, axis=0)
        t = tf.tile(t, [nb_input, nb_nodes, 1])
        cijk = tf.concat([ei, ej, rk, t], axis=2)
        mcijk = tf.layers.dense(cijk, 100)
        bijk = tf.layers.dense(mcijk, 1)
        bijk = tf.squeeze(bijk)
        bijk = tf.nn.leaky_relu(bijk)
        aijk = tf.nn.softmax(bijk + mask_mat * (1e9))
        eih = tf.matmul(tf.expand_dims(aijk, axis=1), mcijk)

        return tf.squeeze(eih) + input

    def inference2(self, entity_table, relation_table, time_in_x, query, trace, batch_size, nb_nodes, prev_state, range_h):
        query_embedding = tf.nn.embedding_lookup(relation_table, query)
        t = tf.expand_dims(time_in_x, axis=0)
        t = tf.tile(t, [batch_size, 1])
        es = trace[:, 0]
        eo = trace[:, -1]
        es_embedding = tf.nn.embedding_lookup(entity_table, es)
        rnn_input = tf.concat([query_embedding, es_embedding, t], axis=1)
        output_f, state_f = self.rnn_step(rnn_input, prev_state)
        score = tf.layers.dense(output_f, nb_nodes)
        score_softmax = tf.nn.softmax(score)
        score_eo = tf.gather_nd(score_softmax, tf.transpose(tf.stack([range_h, eo])))

        return score, state_f, score_eo


class HeteGAT_no_coef(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat_list, hid_units, n_heads, activation=tf.nn.elu, residual=False,
                  mp_att_size=128):
        embed_list = []
        # coef_list = []
        for bias_mat in bias_mat_list:
            attns = []
            head_coef_list = []
            for _ in range(n_heads[0]):
                attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                              out_sz=hid_units[0], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                                              return_coef=return_coef))
            h_1 = tf.concat(attns, axis=-1)
            for i in range(1, len(hid_units)):
                h_old = h_1
                attns = []
                for _ in range(n_heads[i]):
                    attns.append(layers.attn_head(h_1,
                                                  bias_mat=bias_mat,
                                                  out_sz=hid_units[i],
                                                  activation=activation,
                                                  in_drop=ffd_drop,
                                                  coef_drop=attn_drop,
                                                  residual=residual))
                h_1 = tf.concat(attns, axis=-1)
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))
        # att for metapath
        # prepare shape for SimpleAttLayer
        # print('att for mp')
        multi_embed = tf.concat(embed_list, axis=1)
        final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)
        # print(att_val)
        # last layer for clf
        out = []
        for i in range(n_heads[-1]):
            out.append(tf.layers.dense(final_embed, nb_classes, activation=None))
        #     out.append(layers.attn_head(h_1, bias_mat=bias_mat,
        #                                 out_sz=nb_classes, activation=lambda x: x,
        #                                 in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]
        # logits_list.append(logits)
        print('de')
        logits = tf.expand_dims(logits, axis=0)
        # if return_coef:
        #     return logits, final_embed, att_val, coef_list
        # else:
        return logits, final_embed, att_val


class HeteGAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat_list, hid_units, n_heads, activation=tf.nn.elu, residual=False,
                  mp_att_size=128,
                  return_coef=False):
        embed_list = []
        coef_list = []
        for bias_mat in bias_mat_list:
            attns = []
            head_coef_list = []
            for _ in range(n_heads[0]):
                if return_coef:
                    a1, a2 = layers.attn_head(inputs, bias_mat=bias_mat,
                                              out_sz=hid_units[0], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                                              return_coef=return_coef)
                    attns.append(a1)
                    head_coef_list.append(a2)
                    # attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                    #                               out_sz=hid_units[0], activation=activation,
                    #                               in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                    #                               return_coef=return_coef)[0])
                    #
                    # head_coef_list.append(layers.attn_head(inputs, bias_mat=bias_mat,
                    #                                        out_sz=hid_units[0], activation=activation,
                    #                                        in_drop=ffd_drop, coef_drop=attn_drop,
                    #                                        residual=False,
                    #                                        return_coef=return_coef)[1])
                else:
                    attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                                  out_sz=hid_units[0], activation=activation,
                                                  in_drop=ffd_drop, coef_drop=attn_drop, residual=False,
                                                  return_coef=return_coef))
            head_coef = tf.concat(head_coef_list, axis=0)
            head_coef = tf.reduce_mean(head_coef, axis=0)
            coef_list.append(head_coef)
            h_1 = tf.concat(attns, axis=-1)
            for i in range(1, len(hid_units)):
                h_old = h_1
                attns = []
                for _ in range(n_heads[i]):
                    attns.append(layers.attn_head(h_1,
                                                  bias_mat=bias_mat,
                                                  out_sz=hid_units[i],
                                                  activation=activation,
                                                  in_drop=ffd_drop,
                                                  coef_drop=attn_drop,
                                                  residual=residual))
                h_1 = tf.concat(attns, axis=-1)
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))
        # att for metapath
        # prepare shape for SimpleAttLayer
        # print('att for mp')
        multi_embed = tf.concat(embed_list, axis=1)
        final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)
        # print(att_val)
        # last layer for clf
        out = []
        for i in range(n_heads[-1]):
            out.append(tf.layers.dense(final_embed, nb_classes, activation=None))
        #     out.append(layers.attn_head(h_1, bias_mat=bias_mat,
        #                                 out_sz=nb_classes, activation=lambda x: x,
        #                                 in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]
        # logits_list.append(logits)
        logits = tf.expand_dims(logits, axis=0)
        if return_coef:
            return logits, final_embed, att_val, coef_list
        else:
            return logits, final_embed, att_val


