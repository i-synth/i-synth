from util import Record, identity
from util_tf import tf, placeholder, Normalize, Dense, Forward, Attention, Dropout, Smooth
import numpy as np


def sinusoid(time, dim, freq= 1e-4, name= 'sinusoid', scale= True, array= False):
    """returns a rank-2 tensor of shape `time, dim`, where each row
    corresponds to a time step and each column a sinusoid, with
    frequencies in a geometric progression from 1 to `freq`.

    """
    assert not dim % 2
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


class Sinusoid(Record):

    def __init__(self, dim, len_cap= None, name= 'sinusoid'):
        self.dim, self.name = dim, name
        self.pos = tf.constant(
            sinusoid(len_cap, dim, array= True), tf.float32, name= name
        ) if len_cap else None

    def __call__(self, time, name= None):
        with tf.variable_scope(name or self.name):
            return sinusoid(time, self.dim) if self.pos is None else self.pos[:time]


class ForwardBlock(Record):

    def __init__(self, dim, dim_mid, name= 'forward'):
        with tf.variable_scope(name):
            self.name = name
            self.forward = Forward(dim, dim_mid)
            self.normalize = Normalize(dim)

    def __call__(self, x, act, dropout, name= None):
        with tf.variable_scope(name or self.name):
            return self.normalize(x + dropout(self.forward(x, act)))


class AttentionBlock(Record):

    def __init__(self, dim, softmax, name= 'attention'):
        with tf.variable_scope(name):
            self.name = name
            self.attention = Attention(dim, softmax= softmax)
            self.normalize = Normalize(dim)

    def __call__(self, x, value, dropout, num_head, mask= None, name= None):
        with tf.variable_scope(name or self.name):
            return self.normalize(x + dropout(self.attention(x, value, num_head, mask)))


class EncodeBlock(Record):

    def __init__(self, dim, dim_mid, num_head, softmax, name):
        self.num_head, self.softmax = num_head, softmax
        with tf.variable_scope(name):
            self.name = name
            self.attention = AttentionBlock(dim, softmax)
            self.forward = ForwardBlock(dim, dim_mid)

    def __call__(self, x, act, dropout, name= None):
        with tf.variable_scope(name or self.name):
            x = self.attention(x, x, dropout, self.num_head)
            x = self.forward(x, act, dropout)
        return x


class DecodeBlock(Record):

    def __init__(self, dim, dim_mid, num_head, softmax, name):
        self.num_head, self.softmax = num_head, softmax
        with tf.variable_scope(name):
            self.name = name
            self.causal = AttentionBlock(dim, softmax, 'causal')
            self.attention = AttentionBlock(dim, softmax)
            self.forward = ForwardBlock(dim, dim_mid)

    def __call__(self, x, v, w, act, dropout, mask= None, name= None):
        with tf.variable_scope(name or self.name):
            x = self.causal(x, v, dropout, self.num_head, mask)
            x = self.attention(x, w, dropout, self.num_head)
            x = self.forward(x, act, dropout)
        return x


class Transformer(Record):
    """-> Record

    model = Transformer.new()
    model_train = model.data(src_train, tgt_train, len_cap)
    model_valid = model.data(src_valid, tgt_valid)

    forcing_train = model_train.forcing().train()
    forcing_valid = model_valid.forcing()

    autoreg_train = model_train.autoreg(trainable= True).train()
    autoreg_valid = model_valid.autoreg(trainable= False)

    """

    @staticmethod
    def new(end= 1
            , dim_src= 256, dim= 512
            , dim_tgt= 512, dim_mid= 1024
            , num_layer= 2, num_head= 4
            , softmax= True
            , smooth= 0.4
            , dropout= 0.1):
        """-> Transformer with fields

            end : i32 ()
        emb_src : f32 (dim_src, dim)
        emb_tgt : Forward
         encode : tuple EncodeBlock
         decode : tuple DecodeBlock
          frame : Forward
          close : Forward
         smooth : Smooth
        dropout : Dropout

        `end` is treated as the padding for source.

        """
        assert not dim % 2 and not dim % num_head
        emb_src = tf.get_variable('emb_src', (dim_src, dim), tf.float32)
        emb_tgt = Forward(dim_tgt, dim_mid, dim, name= 'emb_tgt')
        with tf.variable_scope('encode'):
            encode = tuple(EncodeBlock(
                dim, dim_mid, num_head, softmax, "layer{}".format(i + 1))
                             for i in range(num_layer))
        with tf.variable_scope('decode'):
            decode = tuple(DecodeBlock(
                dim, dim_mid, num_head, softmax, "layer{}".format(i + 1))
                        for i in range(num_layer))
        frame = Forward(dim, dim_mid, dim_tgt, name= 'frame')
        close = Forward(dim, dim_mid, 1, name= 'close')
        return Transformer(
            dim= dim, dim_tgt= dim_tgt
            , end= tf.constant(end, tf.int32, (), 'end')
            , emb_src= emb_src, encode= encode
            , emb_tgt= emb_tgt, decode= decode
            , frame= frame, close= close
            , smooth= Smooth(smooth)
            , dropout= Dropout(dropout, (None, 1, dim)))

    def data(self, src= None, tgt= None, len_cap= None):
        """-> Transformer with new fields

            src_ : i32  (b, ?)          source feed, in range `[0, dim_src)`
            tgt_ : f32  (b, ?, dim_tgt) target feed
             src : i32  (b, s)          source with `end` trimmed among the batch
             tgt : i32  (b, t, dim_tgt) target with `nan` trimmed among the batch
            gold : i32  (b, t)          target one step ahead
           ended : bool (b, t)          has ended for gold
        position : Sinusoid

        setting `len_cap` makes it more efficient for training.  you
        won't be able to feed it longer sequences, but it doesn't
        affect any model parameters.

        """
        end, dim, dim_tgt = self.end, self.dim, self.dim_tgt
        count_not_all = lambda x: tf.reduce_sum(tf.to_int32(~ tf.reduce_all(x, 0)))
        with tf.variable_scope('src'):
            src = src_ = placeholder(tf.int32, (None, None), src)
            len_src = count_not_all(tf.equal(src, end)) + 1
            src = src[:,:len_src]
        with tf.variable_scope('tgt'):
            tgt = tgt_ = placeholder(tf.float32, (None, None, dim_tgt), tgt)
            ended = tf.is_nan(tgt[:,:,0])
            # len_tgt = count_not_all(ended) + 1
            # tgt, ended = tgt[:,:len_tgt], ended[:,:len_tgt]
            tgt = tf.where(tf.is_nan(tgt), tf.zeros_like(tgt), tgt)
            tgt, gold, ended = tgt[:,:-1], tgt[:,1:], ended[:,1:]
        return Transformer(
            position= Sinusoid(dim, len_cap)
            , src_= src_, src= src
            , tgt_= tgt_, tgt= tgt
            , gold= gold, ended= ended
            , **self)

    def autoreg(self, act= tf.nn.relu, trainable= True):
        """-> Transformer with new fields, autoregressive

        len_tgt : i32 ()              steps to unfold aka t
         output : f32 (b, t, dim_tgt) frame prediction
         closed : f32 (b, t)          close prediction, aka end of frames, on logit scale
            acc : f32 ()              close prediction accuracy
           err0 : f32 ()              close prediction loss
           err1 : f32 ()              frame prediction l1 loss
           err2 : f32 ()              frame prediction l2 loss
           loss : f32 ()              err0 + err2

        must be called after `data`.

        """
        frame, close = self.frame, self.close
        position, dropout = self.position, self.dropout if trainable else identity
        src, emb_src, encode = self.src, self.emb_src, self.encode
        tgt, emb_tgt, decode = self.tgt, self.emb_tgt, self.decode
        dim, dim_tgt = self.dim, self.dim_tgt
        with tf.variable_scope('emb_src_autoreg'):
            w = tf.gather(emb_src, src)
            w = dropout(w + position(tf.shape(w)[1]))
        with tf.variable_scope('encode_autoreg'):
            for enc in encode: w = enc(w, act, dropout)
        with tf.variable_scope('decode_autoreg'):
            with tf.variable_scope('init'):
                len_tgt = tf.shape(tgt)[1]
                pos = position(len_tgt)
                x = tgt[:,:1]
                y = x[:,1:]
                z = tf.reshape(y, (tf.shape(y)[0], 0, 1))
                v = tf.reshape(y, (tf.shape(y)[0], 0, dim))
            def autoreg(i, x, vs, y, z):
                # i : ()              time step from 0 to t=len_tgt
                # x : (b, 1, dim_tgt) frame at step i
                # v : (b, t, dim)     embeded x
                # y : (b, t, dim_tgt) x one step ahead
                # z : (b, t, 1)       close prediction
                with tf.variable_scope('emb_tgt'): x = dropout(emb_tgt(tgt, act) + pos[i])
                us = []
                for dec, v in zip(decode, vs):
                    with tf.variable_scope('cache_v'):
                        v = tf.concat((v, x), 1)
                        us.append(v)
                    x = dec(x, v, w, act, dropout)
                x, c = frame(x, act), close(x, act)
                with tf.variable_scope('cache_y'): y = tf.concat((y, x), 1)
                with tf.variable_scope('cache_z'): z = tf.concat((z, c), 1)
                return i + 1, x, tuple(us), y, z
            _, _, _, y, z = tf.while_loop(
                lambda i, *_: i < len_tgt # todo stop when end is reached if not trainable
                , autoreg
                , (0, x, (v,)*len(decode), y, z)
                , (tf.TensorShape(())
                   , x.shape
                   , (tf.TensorShape((None, None, dim)),)*len(decode)
                   , y.shape
                   , tf.TensorShape((None, None, 1)))
                , back_prop= trainable
                , swap_memory= True
                , name= 'autoreg')
            z = tf.squeeze(z, -1)
        return Transformer(len_tgt= len_tgt, output= y, closed= z, **self)._pred()

    def forcing(self, act= tf.nn.relu):
        """-> Transformer with new fields, teacher forcing

        output : f32 (b, t, dim_tgt) frame prediction
        closed : f32 (b, t)          close prediction, aka end of frames, on logit scale
           acc : f32 ()              close prediction accuracy
          err0 : f32 ()              close prediction loss
          err1 : f32 ()              frame prediction l1 loss
          err2 : f32 ()              frame prediction l2 loss
          loss : f32 ()              err0 + err2

        must be called after `data`.

        """
        frame, close = self.frame, self.close
        position, dropout = self.position, self.dropout
        src, emb_src, encode = self.src, self.emb_src, self.encode
        tgt, emb_tgt, decode = self.tgt, self.emb_tgt, self.decode
        with tf.variable_scope('emb_src_forcing'):
            w = tf.gather(emb_src, src)
            w = dropout(w + position(tf.shape(w)[1]))
        with tf.variable_scope('emb_tgt_forcing'):
            x = emb_tgt(tgt, act)
            x = dropout(x + position(tf.shape(x)[1]))
        with tf.variable_scope('encode_forcing'):
            for enc in encode: w = enc(w, act, dropout)
        with tf.variable_scope('decode_forcing'):
            with tf.variable_scope('mask'):
                mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones((tf.shape(x)[1],)*2)).to_dense()
                if self.decode[0].softmax: mask = tf.log(mask)
            for dec in decode: x = dec(x, x, w, act, dropout, mask)
        with tf.variable_scope('frame_forcing'):
            y = frame(x, act)
        with tf.variable_scope('close_forcing'):
            z = tf.squeeze(close(x, act), -1)
        return Transformer(output= y, closed= z, **self)._pred()

    def _pred(self):
        gold, output, smooth = self.gold, self.output, self.smooth
        ended, closed = self.ended, self.closed
        with tf.variable_scope('acc'):
            acc = tf.reduce_mean(tf.to_float(tf.equal(ended, 0.0 < closed)))
        with tf.variable_scope('loss'):
            err0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits= closed, labels= smooth(tf.to_float(ended))))
            diff = gold - output
            err1 = tf.reduce_mean(tf.reduce_sum(tf.abs(diff), -1))
            err2 = tf.reduce_mean(tf.reduce_sum(tf.square(diff), -1))
            loss = err0 + err2
        return Transformer(acc= acc, err0= err0, err1= err1, err2= err2, loss= loss, **self)

    def train(self, warmup= 4e3, beta1= 0.9, beta2= 0.98, epsilon= 1e-9):
        """-> Transformer with new fields

        step : i64 () global update step
          lr : f32 () learning rate for the current step
          up :        update operation

        """
        dim, loss = self.dim, self.loss
        with tf.variable_scope('lr'):
            s = tf.train.get_or_create_global_step()
            t = tf.to_float(s + 1)
            lr = placeholder(tf.float32, (), (dim ** -0.5) * tf.minimum(t ** -0.5, t * (warmup ** -1.5)), 'lr')
        up = tf.train.AdamOptimizer(lr, beta1, beta2, epsilon).minimize(loss, s)
        return Transformer(step= s, lr= lr, up= up, **self)
