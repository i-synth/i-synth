from util import Record, identity
from util_tf import QueryAttention as Attention
from util_tf import tf, placeholder, Normalize, Smooth, Dropout, Linear, Affine, Forward
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


class EncodeBlock(Record):

    def __init__(self, dim, dim_mid, act, name):
        with tf.variable_scope(name):
            self.name = name
            with tf.variable_scope('att'):
                self.att = Attention(dim, layer= Forward, mid= dim_mid, act= act)
                self.norm_att = Normalize(dim)
            with tf.variable_scope('fwd'):
                self.fwd = Forward(dim, dim, dim_mid, act)
                self.norm_fwd = Normalize(dim)

    def __call__(self, x, mask, dropout, name= None):
        with tf.variable_scope(name or self.name):
            with tf.variable_scope('att'): x = self.norm_att(x + dropout(self.att(x, x, mask)))
            with tf.variable_scope('fwd'): x = self.norm_fwd(x + dropout(self.fwd(x)))
            return x


class DecodeBlock(Record):

    def __init__(self, dim, dim_mid, act, name):
        with tf.variable_scope(name):
            self.name = name
            with tf.variable_scope('csl'):
                self.csl = Attention(dim, layer= Forward, mid= dim_mid, act= act)
                self.norm_csl = Normalize(dim)
            with tf.variable_scope('att'):
                self.att = Attention(dim, layer= Forward, mid= dim_mid, act= act)
                self.norm_att = Normalize(dim)
            with tf.variable_scope('fwd'):
                self.fwd = Forward(dim, dim, dim_mid, act)
                self.norm_fwd = Normalize(dim)

    def __call__(self, x, v, w, m, dropout, mask= None, name= None):
        with tf.variable_scope(name or self.name):
            with tf.variable_scope('csl'): x = self.norm_csl(x + dropout(self.csl(x, v, mask)))
            with tf.variable_scope('att'): x = self.norm_att(x + dropout(self.att(x, w, m)))
            with tf.variable_scope('fwd'): x = self.norm_fwd(x + dropout(self.fwd(x)))
            return x


# # original transformer
# from util_tf import TransformerAttention as Attention
# class EncodeBlock(Record):
#     def __init__(self, dim, dim_mid, act, name):
#         with tf.variable_scope(name):
#             self.name = name
#             with tf.variable_scope('att'):
#                 self.att = Attention(dim)
#                 self.norm_att = Normalize(dim)
#             with tf.variable_scope('fwd'):
#                 self.fwd = Forward(dim, dim, dim_mid, act)
#                 self.norm_fwd = Normalize(dim)
#     def __call__(self, x, mask, dropout, name= None):
#         with tf.variable_scope(name or self.name):
#             with tf.variable_scope('att'): x = self.norm_att(x + dropout(self.att(x, x, mask)))
#             with tf.variable_scope('fwd'): x = self.norm_fwd(x + dropout(self.fwd(x)))
#             return x
# class DecodeBlock(Record):
#     def __init__(self, dim, dim_mid, act, name):
#         with tf.variable_scope(name):
#             self.name = name
#             with tf.variable_scope('csl'):
#                 self.csl = Attention(dim)
#                 self.norm_csl = Normalize(dim)
#             with tf.variable_scope('att'):
#                 self.att = Attention(dim)
#                 self.norm_att = Normalize(dim)
#             with tf.variable_scope('fwd'):
#                 self.fwd = Forward(dim, dim, dim_mid, act)
#                 self.norm_fwd = Normalize(dim)
#     def __call__(self, x, v, w, m, dropout, mask= None, name= None):
#         with tf.variable_scope(name or self.name):
#             with tf.variable_scope('csl'): x = self.norm_csl(x + dropout(self.csl(x, v, mask)))
#             with tf.variable_scope('att'): x = self.norm_att(x + dropout(self.att(x, w, m)))
#             with tf.variable_scope('fwd'): x = self.norm_fwd(x + dropout(self.fwd(x)))
#             return x


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
            , dim_tgt= 512, dim_mid= 1024, num_layer= 2
            , act= tf.nn.relu
            , smooth= 0.4
            , dropout= 0.1):
        """-> Transformer with fields

            end : i32 ()
        emb_src : Linear
        emb_tgt : Forward
         encode : tuple EncodeBlock
         decode : tuple DecodeBlock
          frame : Forward
          close : Forward
         smooth : Smooth
        dropout : Dropout

        `end` is treated as the padding for source.

        """
        assert not dim % 2
        emb_src = Linear(dim, dim_src, 'emb_src')
        emb_tgt = Forward(dim, dim_tgt, dim_mid, act, 'emb_tgt')
        with tf.variable_scope('encode'):
            encode = tuple(EncodeBlock(dim, dim_mid, act, "layer{}".format(1+i)) for i in range(num_layer))
        with tf.variable_scope('decode'):
            decode = tuple(DecodeBlock(dim, dim_mid, act, "layer{}".format(1+i)) for i in range(num_layer))
        return Transformer(
            dim= dim, dim_tgt= dim_tgt
            , end= tf.constant(end, tf.int32, (), 'end')
            , emb_src= emb_src, encode= encode
            , emb_tgt= emb_tgt, decode= decode
            , frame= Forward(dim_tgt, dim, dim_mid, act, 'frame')
            , close= Forward(      1, dim, dim_mid, act, 'close')
            , smooth= Smooth(smooth)
            , dropout= Dropout(dropout, (None, None, dim)))

    def data(self, src= None, tgt= None, len_cap= None):
        """-> Transformer with new fields

            src_ : i32  (b, ?)          source feed, in range `[0, dim_src)`
            tgt_ : f32  (b, ?, dim_tgt) target feed
             src : i32  (b, s)          source with `end` trimmed among the batch
             tgt : i32  (b, t, dim_tgt) target with `nan` trimmed among the batch
            mask : f32  (b, s)          source mask
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
            len_src = count_not_all(tf.equal(src, end))
            src = src[:,:len_src]
        with tf.variable_scope('tgt'):
            tgt = tgt_ = placeholder(tf.float32, (None, None, dim_tgt), tgt)
            ended = tf.is_nan(tgt[:,:,0])
            # this optimization is already performed during data feeding
            # len_tgt = count_not_all(ended) + 1
            # tgt, ended = tgt[:,:len_tgt], ended[:,:len_tgt]
            tgt = tf.where(tf.is_nan(tgt), tf.zeros_like(tgt), tgt)
            tgt, gold, ended = tgt[:,:-1], tgt[:,1:], ended[:,1:]
        return Transformer(
            position= Sinusoid(dim, len_cap)
            , src_= src_, src= src, mask= tf.to_float(tf.expand_dims(tf.not_equal(src, end), 1))
            , tgt_= tgt_, tgt= tgt
            , gold= gold, ended= ended
            , **self)

    def autoreg(self, trainable= True, minimal= False):
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
        assert not trainable or not minimal
        frame, close = self.frame, self.close
        dropout = self.dropout if trainable else identity
        mask, position = self.mask, self.position
        src, emb_src, encode = self.src, self.emb_src, self.encode
        tgt, emb_tgt, decode = self.tgt, self.emb_tgt, self.decode
        with tf.variable_scope('emb_src_autoreg'): w = position(tf.shape(src)[1]) + dropout(emb_src.embed(src))
        with tf.variable_scope('encode_autoreg'):
            for enc in encode: w = enc(w, mask, dropout)
        with tf.variable_scope('decode_autoreg'):
            with tf.variable_scope('init'):
                len_tgt = tf.shape(tgt)[1]
                pos = position(len_tgt)
                i = tf.constant(0)
                x = tgt[:,:1]
                v = w[:,:0]
                y = x[:,1:]
                z = tf.reshape(y, (tf.shape(y)[0], 0, 1))
            def autoreg(i, x, vs, y, z):
                # i : ()              time step from 0 to t=len_tgt
                # x : (b, 1, dim_tgt) frame at step i
                # v : (b, t, dim)     embeded x
                # y : (b, t, dim_tgt) x one step ahead
                # z : (b, t, 1)       close prediction
                with tf.variable_scope('emb_tgt'): x = pos[i] + dropout(emb_tgt(x))
                us = []
                for dec, v in zip(decode, vs):
                    with tf.variable_scope('cache_v'):
                        v = tf.concat((v, x), 1)
                        us.append(v)
                    x = dec(x, v, w, mask, dropout)
                x, c = frame(x), close(x)
                with tf.variable_scope('cache_y'): y = tf.concat((y, x), 1)
                with tf.variable_scope('cache_z'): z = tf.concat((z, c), 1)
                return i + 1, x, tuple(us), y, z
            _, _, _, y, z = tf.while_loop(
                lambda i, *_: i < len_tgt # todo stop when minimal
                , autoreg
                , (i, x, (v,)*len(decode), y, z)
                , (i.shape, x.shape, (v.shape,)*len(decode), y.shape, tf.TensorShape((None, None, 1)))
                , back_prop= trainable
                , swap_memory= True
                , name= 'autoreg')
            z = tf.squeeze(z, -1)
        return Transformer(len_tgt= len_tgt, output= y, closed= z, **self)._eval()

    def forcing(self, trainable= True):
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
        dropout = self.dropout if trainable else identity
        mask, position = self.mask, self.position
        src, emb_src, encode = self.src, self.emb_src, self.encode
        tgt, emb_tgt, decode = self.tgt, self.emb_tgt, self.decode
        with tf.variable_scope('emb_src_forcing'): w = position(tf.shape(src)[1]) + dropout(emb_src.embed(src))
        with tf.variable_scope('emb_tgt_forcing'): x = position(tf.shape(tgt)[1]) + dropout(emb_tgt(tgt))
        with tf.variable_scope('encode_forcing'):
            for enc in encode: w = enc(w, mask, dropout)
        with tf.variable_scope('decode_forcing'):
            with tf.variable_scope('mask'):
                causal_mask = tf.linalg.LinearOperatorLowerTriangular(tf.ones((tf.shape(x)[1],)*2)).to_dense()
            for dec in decode: x = dec(x, x, w, mask, dropout, causal_mask)
        with tf.variable_scope('frame_forcing'): y = frame(x)
        with tf.variable_scope('close_forcing'): z = tf.squeeze(close(x), -1)
        return Transformer(output= y, closed= z, **self)._eval()

    def _eval(self):
        gold, output, ended, closed, smooth = self.gold, self.output, self.ended, self.closed, self.smooth
        with tf.variable_scope('acc'):
            acc = tf.reduce_mean(tf.to_float(tf.equal(ended, 0.0 < closed)))
        with tf.variable_scope('loss'):
            err0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits= closed, labels= smooth(tf.to_float(ended))))
            diff = gold - output
            err1 = tf.reduce_mean(tf.reduce_sum(tf.abs(diff), -1))
            err2 = tf.reduce_mean(tf.reduce_sum(tf.square(diff), -1))
            loss = err1 + err2
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
            lr = (dim ** -0.5) * tf.minimum(t ** -0.5, t * (warmup ** -1.5))
        up = tf.train.AdamOptimizer(lr, beta1, beta2, epsilon).minimize(loss, s)
        return Transformer(step= s, lr= lr, up= up, **self)
