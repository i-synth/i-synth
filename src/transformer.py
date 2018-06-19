from util import Record
from util_tf import tf, placeholder, normalize
import numpy as np


def sinusoid(time, dim, freq= 1e-4, name= 'sinusoid', scale= True, array= False):
    """returns a rank-2 tensor of shape `time, dim`, where each row
    corresponds to a time step and each column a sinusoid, with
    frequencies in a geometric progression from 1 to `freq`.

    """
    if array:
        a = (freq ** ((2 / dim) * np.arange(dim // 2))).reshape(-1, 1) @ np.arange(time).reshape(1, -1)
        s = np.concatenate((np.sin(a), np.cos(a)), -1).reshape(dim, time)
        if scale: s *= dim ** -0.5
        return s.T
    with tf.variable_scope(name):
        a = tf.reshape(
            freq ** ((2 / dim) * tf.range(dim // 2, dtype= tf.float32))
            , (-1, 1)) @ tf.reshape(
                tf.range(tf.cast(time, tf.float32), dtype= tf.float32)
                , (1, -1))
        s = tf.reshape(tf.concat((tf.sin(a), tf.cos(a)), -1), (dim, time))
        if scale: s *= dim ** -0.5
        return tf.transpose(s)


def multihead_attention(value, query, dim= 64, num_head= 8, softmax= True, mask= None, name= 'attention'):
    """computes multi-head attention from `value` and `query` tensors.

    with batch size `b`, time steps `s, t`, dimensions `k, q`

    - value : b,s,k
    - query : b,t,q

    the returned tensor has shape `b, t, dim * num_head`, and `mask`
    when supplied must have shape compatible to `num_head, b, t, s`.

    """
    dense = lambda x, d, name: tf.layers.dense(x, d, use_bias= False, name= name)
    split = lambda x: tf.split(x, num_head, -1)
    # v : h,b,s,d
    # k : h,b,s,d
    # q : h,b,t,d
    # a : h,b,t,s
    with tf.variable_scope(name):
        q = tf.stack(split(dense(query, dim * num_head, 'q')))
        if softmax:
            v = tf.stack(split(dense(value, dim * num_head, 'v')))
            k = tf.stack(split(dense(value, dim * num_head, 'k')))
            a = tf.matmul(q, k, transpose_b= True)
            a *= (dim ** -0.5)
            if mask is not None: a += mask
            a = tf.nn.softmax(a)
        else:
            v = k = tf.stack(split(value))
            a = tf.matmul(q, k, transpose_b= True)
            if mask is not None: a *= mask
            a = tf.square(a)
            a /= tf.reduce_sum(a, -1, True) + 1e-8
        return tf.concat(tf.unstack(a @ v), -1)


def model(len_cap= None
          , src= None, dim_src= 256
          , tgt= None, dim_tgt= 256
          , dim= 256,  dim_mid= 512
          , num_head= 4, num_layer= 2
          , softmax= True
          , activation= tf.nn.relu
          , training= True
          , dropout= 0.1
          , warmup= 4e3
          , end= 1):
    """-> Record, with the following fields of tensors

    dropout : f32 ()              dropout rate, has no effect if not `training`
        end : i32 ()              end padding for `src`
        src : i32 (b, s)          source feed, in range `[0, dim_src)`
        tgt : f32 (b, t, dim_tgt) target feed, padded at the end with `nan`
      frame : f32 (b, t, dim_tgt) frame prediction
      close : f32 (b, t)          close prediction, aka end of frames
       err0 : f32 ()              close prediction loss
       err1 : f32 ()              frame prediction l1 loss
       err2 : f32 ()              frame prediction l2 loss

    and as an autoregressive model : w, x -> y, z

    w : f32  (b, s, dim)     encoded `src`
    x : f32  (b, ?, dim_tgt) target feed for the current prediction
    y : f32  (b, dim_tgt)    current frame prediction
    z : bool (b,)            current close prediction

    and if `training`

    step : i64 () global update step
      lr : f32 () learning rate for the current step
      up :        update operation

    setting `len_cap` makes it more efficient for training.  you won't
    be able to feed it longer sequences, but it doesn't affect any
    model parameters, so you can build another without `len_cap`
    reusing all variables.

    """
    assert not dim % 2 and not dim % num_head
    self = Record()
    with tf.variable_scope('dropout'):
        self.dropout = placeholder(tf.float32, (), dropout)
        keep = 1.0 - self.dropout
    def dropout(x, keep= keep):
        with tf.variable_scope('dropout'):
            return tf.nn.dropout(x, keep, (tf.shape(x)[0], 1, dim))
    if not training: dropout = lambda x: x
    attention = lambda v, q, mask= None: multihead_attention(
        v, q, dim // num_head, num_head, softmax, mask)
    forward = lambda x, dim_mid= dim_mid, dim= dim: tf.layers.dense(
        tf.layers.dense(
            x, dim_mid, activation, name= 'relu')
        , dim, name= 'linear')
    nrd = lambda x, y: normalize(x + dropout(y))
    # trim `src` to the maximum valid index among the batch, plus one for padding
    count_not_all = lambda x: tf.reduce_sum(tf.to_int32(~ tf.reduce_all(x, 0)))
    with tf.variable_scope('src'):
        end = self.end = tf.constant(end, tf.int32, (), 'end')
        src = self.src = placeholder(tf.int32, (None, None), src)
        len_src = count_not_all(tf.equal(src, end)) + 1
        src = src[:,:len_src]
    # same for `tgt`
    with tf.variable_scope('tgt'):
        tgt = self.tgt = placeholder(tf.float32, (None, None, dim_tgt), tgt)
        ended = tf.is_nan(tgt[:,:,0])
        len_tgt = count_not_all(ended) + 1
        tgt = tgt[:,:len_tgt]
        tgt = tf.where(tf.is_nan(tgt), tf.zeros_like(tgt), tgt)
        tgt, gold, ended = tgt[:,:-1], tgt[:,1:], ended[:,1:len_tgt]
    # embedding
    if len_cap: emb_pos = tf.constant(sinusoid(len_cap, dim, array= True), tf.float32, name= 'sinusoid')
    with tf.variable_scope('emb_src'):
        pos = emb_pos[:len_src] if len_cap else sinusoid(len_src, dim)
        emb = tf.get_variable('emb', (dim_src, dim), tf.float32)
        w = dropout(pos + tf.gather(emb, src))
        # w = normalize(x) todo test if necessary
    self.x = tgt
    with tf.variable_scope('emb_tgt'):
        x = dropout(forward(tgt))
        # todo add position encoding
        # w = normalize(x) todo test if necessary
    # transformer
    with tf.variable_scope('encode'):
        for i in range(num_layer):
            with tf.variable_scope("layer{}".format(i + 1)):
                with tf.variable_scope("attention"):
                    w = nrd(w, attention(w, w))
                with tf.variable_scope("forward"):
                    w = nrd(w, forward(w))
    self.w = w
    with tf.variable_scope('decode'):
        with tf.variable_scope('mask'):
            t = tf.shape(x)[1]
            mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones((t, t))).to_dense()
            if softmax: mask = tf.log(mask)
        for i in range(num_layer):
            with tf.variable_scope("layer{}".format(i + 1)):
                with tf.variable_scope("causal_attention"):
                    x = nrd(x, attention(x, x, mask))
                with tf.variable_scope("attention"):
                    x = nrd(x, attention(w, x))
                with tf.variable_scope("forward"):
                    x = nrd(x, forward(x))
    # output
    with tf.variable_scope('close'):
        close = self.close = tf.squeeze(forward(x, dim, 1), -1)
        self.z = 0.0 < close[:,-1]
    with tf.variable_scope('frame'):
        frame = self.frame = tf.layers.dense(x, dim_tgt)
        self.y = frame[:,-1]
    # done
    with tf.variable_scope('loss'):
        # todo smoothing
        self.err0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits= close, labels= tf.to_float(ended)))
        diff = gold - frame
        self.err1 = tf.reduce_mean(tf.reduce_sum(tf.abs(diff), -1))
        self.err2 = tf.reduce_mean(tf.reduce_sum(tf.square(diff), -1))
        loss = self.err0 + self.err2 * 1e2
    if training:
        with tf.variable_scope('lr'):
            self.step = tf.train.get_or_create_global_step()
            step = tf.to_float(self.step + 1)
            self.lr = tf.placeholder_with_default(
                (dim ** -0.5) * tf.minimum(step ** -0.5, step * (warmup ** -1.5))
                , (), 'lr')
        self.up = tf.train.AdamOptimizer(self.lr, 0.9, 0.98, 1e-9).minimize(loss, self.step)
    return self


def synth_batch(sess, m, src, len_cap= 256):
    w = sess.run(m.w, {m.src: src, m.dropout: 0})
    x = np.zeros((len(src), len_cap, int(m.frame.shape[-1])), np.float32)
    acc, idx = [], []
    for i in range(1, len_cap):
        y, z = sess.run((m.y, m.z), {m.w: w, m.x: x[:,:i], m.dropout: 0})
        x[:,i] = y
        if z.any():
            for k, j in enumerate(np.flatnonzero(z)):
                acc.append(x[j,1:i])
                idx.append(j - k)
            w = w[~z]
            x = x[~z]
            if z.all(): break
    res = list(x[:,1:])
    for i, x in zip(idx[::-1], acc[::-1]):
        res.insert(i, x)
    return res


def synth(sess, m, src, len_cap= 256, batch_size= 32):
    res, rng = [], range(0, len(src) + batch_size, batch_size)
    for i, j in zip(rng, rng[1:]):
        res.extend(synth_batch(sess, m, src[i:j], len_cap))
    return res
