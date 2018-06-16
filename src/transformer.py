from util import Record
import numpy as np
import tensorflow as tf


def placeholder(dtype, shape, x= None):
    """returns a placeholder with `dtype` and `shape`.

    if tensor `x` is given, converts and uses it as default.

    """

    if x is None:
        return tf.placeholder(dtype, shape)
    else:
        return tf.placeholder_with_default(tf.cast(x, dtype), shape)


def normalize(x, axis= -1, eps= 1e-8, name= "normalize"):
    """returns a tensor from `x` scaled and centered across `axis`."""
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(x, axis, keep_dims=True)
        return (x - mean) * tf.rsqrt(var + eps * eps)


def sinusoid(time, dim, freq= 1e-4, name= 'sinusoid', array= False):
    """returns a rank-2 tensor of shape `time, dim`, where each row
    corresponds to a time step and each column a sinusoid, with
    frequencies in a geometric progression from 1 to `freq`.

    """
    if array:
        a = (freq ** ((2 / dim) * np.arange(dim // 2))).reshape(-1, 1) @ np.arange(time).reshape(1, -1)
        return np.concatenate((np.sin(a), np.cos(a)), -1).reshape(dim, time).T
    with tf.variable_scope(name):
        a = tf.reshape(
            freq ** ((2 / dim) * tf.range(dim // 2, dtype= tf.float32))
            , (-1, 1)) @ tf.reshape(
                tf.range(tf.cast(time, tf.float32), dtype= tf.float32)
                , (1, -1))
        return tf.transpose(tf.reshape(tf.concat((tf.sin(a), tf.cos(a)), -1), (dim, time)))


def multihead_attention(value, query, dim= 64, num_head= 8, bias= None, name= 'attention'):
    """computes multi-head attention from `value` and `query` tensors.

    with batch size `b`, time steps `s, t`, dimensions `k, q`

    - value : b,s,k
    - query : b,t,q

    the returned tensor has shape `b, t, dim * num_head`, and `bias`
    when supplied must have shape compatible to `num_head, b, t, s`.

    """
    dense = lambda x, d, name: tf.layers.dense(x, d, use_bias= False, name= name)
    split = lambda x: tf.split(x, num_head, -1)
    with tf.variable_scope(name):
        v = tf.stack(split(dense(value, dim * num_head, 'v'))) # h,b,s,d
        k = tf.stack(split(dense(value, dim * num_head, 'k'))) # h,b,s,d
        q = tf.stack(split(dense(query, dim * num_head, 'q'))) # h,b,t,d
        q = (dim ** -0.5) * tf.matmul(q, k, transpose_b= True) # h,b,t,s
        if bias is not None: q += bias
        # todo try square and normalize instead of softmax
        return tf.concat(tf.unstack(tf.matmul(tf.nn.softmax(q), v)), -1)


def model(len_cap= None
          , src= None, dim_src= 256
          , tgt= None, dim_tgt= 256
          , dim= 256,  dim_mid= 512
          , num_head= 4, num_layer= 2
          , activation= tf.nn.relu
          , training= True
          , dropout= 0.25
          , warmup= 4e3
          , end= 1):
    # src : ?, s
    # tgt : ?, t
    # src should be padded at the end (with `end`)
    #
    # as an autoregressive model, this is a function : w, x -> y
    # encoded src, dense w : ?, s, dim
    # tgt history, index x : ?, t
    # current tgt, logit y : ?, dim_tgt
    assert not dim % 2 and not dim % num_head
    self = Record(end= end)
    # trim `src` to the maximum valid index among the batch
    count = lambda x: tf.reduce_sum(tf.to_int32(x), 0)
    with tf.variable_scope('src'):
        src = self.src = placeholder(tf.int32, (None, None), src)
        len_src = 1 + count(tf.not_equal(count(tf.equal(src, end)), tf.shape(src)[0]))
        src = src[:,:len_src]
    # same for `tgt`, todo: keep track of valid length for secondary prediction
    with tf.variable_scope('tgt'):
        tgt = self.tgt = placeholder(tf.float32, (None, None, dim_tgt), tgt)
        len_tgt = 1 + count(tf.not_equal(count(tf.is_nan(tf.reduce_sum(tgt, -1))), tf.shape(tgt)[0]))
        tgt = tgt[:,:len_tgt]
        tgt = tf.where(tf.is_nan(tgt), tf.zeros_like(tgt), tgt)
        tgt, gold = tgt[:,:-1], tgt[:,1:]
    # building blocks
    if training:
        with tf.variable_scope('train/'):
            self.dropout = tf.placeholder_with_default(dropout, (), 'dropout')
            def dropout(x, keep = 1.0 - self.dropout):
                with tf.variable_scope('dropout'):
                    return tf.nn.dropout(x, keep, (tf.shape(x)[0], 1, dim))
    else:
        dropout = lambda x: x
    attention = lambda v, q, **args: multihead_attention(
        value= v, query= q, dim= dim // num_head, num_head= num_head, **args)
    if len_cap: emb_pos = tf.constant(sinusoid(len_cap, dim, array= True), tf.float32, name= 'sinusoid')
    # construction
    with tf.variable_scope('encode'):
        with tf.variable_scope('embed'):
            pos = emb_pos[:len_src] if len_cap else sinusoid(len_src, dim)
            emb = tf.get_variable('emb', (dim_src, dim), tf.float32)
            w = dropout(pos + tf.gather(normalize(emb), src))
        for i in range(num_layer):
            with tf.variable_scope("layer{}".format(i)):
                w = normalize(w + dropout(attention(w, w)))
                h = tf.layers.dense(w, dim_mid, activation, name= 'relu')
                h = tf.layers.dense(h, dim, name= 'linear')
                w = normalize(w + dropout(h))
    self.w, self.x = w, tgt
    with tf.variable_scope('decode'):
        with tf.variable_scope('embed'):
            x = normalize(dropout(tf.layers.dense(tgt, dim)))
        with tf.variable_scope('mask'):
            len_tgt = tf.shape(tgt)[1] # in case tgt is fed by user
            mask = tf.log(tf.linalg.LinearOperatorLowerTriangular(tf.ones((len_tgt, len_tgt))).to_dense())
        for i in range(num_layer):
            with tf.variable_scope("layer{}".format(i)):
                x = normalize(x + dropout(attention(x, x, bias= mask, name= 'masked_attention')))
                x = normalize(x + dropout(attention(w, x)))
                h = tf.layers.dense(x, dim_mid, activation, name= 'relu')
                h = tf.layers.dense(h, dim, name= 'linear')
                x = normalize(x + dropout(h))
        frame = self.frame = tf.layers.dense(x, dim_tgt, name= 'frame')
    self.y = frame[:,-1]
    # done
    with tf.variable_scope('loss'):
        diff = gold - frame
        self.err1 = tf.reduce_mean(tf.reduce_sum(tf.abs(diff), -1))
        self.err2 = tf.reduce_mean(tf.reduce_sum(tf.square(diff), -1))
    if training:
        with tf.variable_scope('train/'):
            self.step = tf.train.get_or_create_global_step()
            step = tf.to_float(self.step + 1)
            self.lr = tf.placeholder_with_default(
                (dim ** -0.5) * tf.minimum(step ** -0.5, step * (warmup ** -1.5))
                , (), 'lr')
            self.up = tf.train.AdamOptimizer(self.lr, 0.9, 0.98, 1e-9).minimize(self.err1, self.step)
    return self
