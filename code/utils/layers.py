# coding=UTF-8
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

conv1d = tf.layers.conv1d


def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False, return_coef=False):
    """[summary]
    [description]
    Arguments:
        seq {[type]} -- shape=(batch_size, nb_nodes, fea_size))
    """
    if in_drop != 0.0:
        seq = tf.nn.dropout(seq, 1.0 - in_drop)
    seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False, name='conv1')

    f_1 = tf.layers.conv1d(seq_fts, 1, 1, name='conv2')
    f_2 = tf.layers.conv1d(seq_fts, 1, 1, name='conv3')

    logits = f_1 + tf.transpose(f_2, [0, 2, 1])
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

    if coef_drop != 0.0:
        coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
    if in_drop != 0.0:
        seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

    vals = tf.matmul(coefs, seq_fts)
    ret = tf.layers.batch_normalization(vals)

    # residual connection
    if residual:
        if seq.shape[-1] != ret.shape[-1]:
            ret = ret + conv1d(seq, ret.shape[-1], 1, name='conv4')
        else:
            seq_fts = ret + seq
    if return_coef:
        return activation(ret), coefs
    else:
        return activation(ret)

def attn_head1(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False, return_coef=False):
    """[summary]
    [description]
    Arguments:
        seq {[type]} -- shape=(batch_size, nb_nodes, fea_size))
    """
    if in_drop != 0.0:
        seq = tf.nn.dropout(seq, 1.0 - in_drop)
    seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False, name='conv1')

    f_1 = tf.layers.conv1d(seq_fts, 1, 1, name='conv2')
    f_2 = tf.layers.conv1d(seq_fts, 1, 1, name='conv3')
    logits = f_1 + tf.transpose(f_2, [0, 2, 1])
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

    if coef_drop != 0.0:
        coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
    if in_drop != 0.0:
        seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

    vals = tf.matmul(coefs, seq_fts)
    ret = tf.layers.batch_normalization(vals)

    # residual connection
    if residual:
        if seq.shape[-1] != ret.shape[-1]:
            ret = ret + tf.layers.conv1d(seq, ret.shape[-1], 1, name='conv4')
        else:
            ret = ret + seq
    if return_coef:
        return activation(ret), coefs
    else:
        return activation(ret)


def attn_head_const_1(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    """[summary]

    [description]


    """
    adj_mat = 1.0 - bias_mat / -1e9
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        

        logits = adj_mat 
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        #ret = tf.contrib.layers.bias_add(vals)
        ret = tf.layers.batch_normalization(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = tf.sparse_add(adj_mat * f_1, adj_mat *
                               tf.transpose(f_2, [0, 2, 1]))
        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values),
                                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=tf.nn.dropout(
                                        coefs.values, 1.0 - coef_drop),
                                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        #ret = tf.contrib.layers.bias_add(vals)
        ret = tf.layers.batch_normalization(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

def SimpleAttLayer(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2]  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))  # 论文里的q

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)  # (3025, 2, 128)
        #print(f'shape of v : {v.shape}')

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape (3025, 2)
    #print(f'shape of vu : {vu.shape}')  #
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape (3025, 2)
    #print(f'shape of alphas : {alphas.shape}')

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    #output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)  # (3025, 2, 64) x (3025, 2, 1) ->sum (3025, 64)
    myalphas = tf.nn.softmax(tf.reduce_mean(vu, axis=0))  # (2)
    myalphas = tf.expand_dims(tf.expand_dims(myalphas, axis=-1), axis=0)  # (1, 2, 1)
    myalphas = tf.repeat(myalphas, inputs.shape[0], axis=0)  # (3025, 2, 1)
    #print(f'shape of myalphas : {myalphas.shape}')
    output = tf.reduce_sum(inputs * myalphas, 1)  # (3025, 64)
    myalphas = tf.squeeze(myalphas, axis=-1)

    #print(f'shape of output : {output.shape}')

    if not return_alphas:
        return output
    else:
        return output, myalphas  # (3025, 64) (3025, 2)