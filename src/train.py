#!/usr/bin/env python3


trial      = '01'
len_cap    = 318
dim_tgt    = 512
batch_size = 2**4
ckpt       = None


from model import Transformer
from os.path import expanduser, join
from tqdm import tqdm
from util import comp
from util_io import path, load, save
from util_np import np, vpack, c2r
from util_tf import tf, batch
tf.set_random_seed(0)

def load_batch(names):
    x = vpack(map(comp(load, path), names), complex('(nan+nanj)'), 1, 1)
    x[:,0] = 0j
    x = c2r(x)
    _, t, d = x.shape
    assert t <= len_cap
    assert d == dim_tgt
    return x
batch_fn = lambda src, names: (src, tf.py_func(load_batch, (names,), tf.float32))

###############
# preparation #
###############

logdir = expanduser("~/cache/tensorboard-logdir/i-synth")
index = np.load("trial/data/index.npy").item()
texts = np.load("trial/data/texts.npy")
names = np.load("trial/data/names.npy")
epoch, split = divmod(len(texts), batch_size)
print("{} batches of {} training instances, {} validation".format(epoch, batch_size, split))

####################
# validation model #
####################

model = Transformer.new(dim_src= len(index), dim_tgt= dim_tgt)
model_valid = model.data(texts[:split], load_batch(names[:split]), len_cap)
autoreg_valid = model_valid.autoreg(trainable= False)
forcing_valid = model_valid.forcing()

# # for profiling
# from util_tf import profile
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     with tf.summary.FileWriter(join(logdir, "graph"), sess.graph) as wtr:
#         profile(sess, wtr, forcing_valid.loss, tag= 'forcing')
#         profile(sess, wtr, autoreg_valid.loss, tag= 'autoreg')

# ' according to their categories or crimes.\n'
src, tgt = texts[978:979,:42], load_batch(names[978:979])[:,:-1]
synth_forcing = {forcing_valid.src: src, forcing_valid.tgt: tgt}
synth_autoreg = {autoreg_valid.src: src, autoreg_valid.tgt: tgt[:,:1], autoreg_valid.len_tgt: tgt.shape[1]}

##################
# training model #
##################

model_train = model.data(*batch((texts[split:], names[split:]), batch_size, fn= batch_fn), len_cap)
forcing_train = model_train.forcing().train(warmup= epoch)
autoreg_train = model_train.autoreg().train(warmup= epoch)

############
# training #
############

saver = tf.train.Saver()
sess = tf.InteractiveSession()
wtr = tf.summary.FileWriter(join(logdir, "trial{}".format(trial)))

if ckpt:
    saver.restore(sess, ckpt)
else:
    tf.global_variables_initializer().run()

step_eval = epoch // 32
summ = tf.summary.merge((
    tf.summary.scalar('step_acc',    forcing_valid.acc)
    , tf.summary.scalar('step_err0', forcing_valid.err0)
    , tf.summary.scalar('step_err1', forcing_valid.err1)
    , tf.summary.scalar('step_err2', forcing_valid.err2)))

# warmup only with teacher forcing
for _ in tqdm(range(epoch), ncols= 70):
    sess.run(forcing_train.up)
    step = sess.run(forcing_train.step)
    if not (step % step_eval):
        wtr.add_summary(sess.run(summ), step)

# mixed teacher forcing and backprop through time
for r in 5, 4, 3, 2, 1:
    for _ in tqdm(range(epoch), ncols= 70):
        if not step % r:
            sess.run(autoreg_train.up)
        else:
            sess.run(forcing_train.up)
        step = sess.run(forcing_train.step)
        if not step % step_eval:
            wtr.add_summary(sess.run(summ), step)
    saver.save(sess, "trial/model/{}_{}".format(trial, step), write_meta_graph= False)
    save("trial/pred/{}_{}_forcing".format(step, trial), forcing_valid.output.eval(synth_forcing))
    save("trial/pred/{}_{}_autoreg".format(step, trial), autoreg_valid.output.eval(synth_autoreg))
