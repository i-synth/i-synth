#!/usr/bin/env python3


trial      = '01'
batch_size = 2**4
step_eval  = 2**7
step_save  = 2**12
ckpt       = None


from os.path import expanduser, join
from tqdm import tqdm
from transformer import model
from util_io import save
from util_np import np, permute, c2r
from util_tf import tf, batch
tf.set_random_seed(0)

path = expanduser("~/cache/tensorboard-logdir/i-synth")
idx = np.load("trial/data/index.npy").item()
src = np.load("trial/data/texts.npy")
tgt = np.load("trial/data/grams.npy")
tgt = c2r(tgt)
s,t = src[5:6], tgt[5:6]

i = permute(len(src))
src = src[i]
tgt = tgt[i]
del i

# # for testing
# m = model(dim_src= len(idx), training= False)

# # for profiling
# from util_tf import profile
# m = model(dim_src= len(idx))
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     profile(join(path, "graph"), sess, m.up, {m.src: src[:batch_size], m.tgt: tgt[:batch_size]})

# for training
src, tgt = batch((src, tgt), batch_size= batch_size, shuffle= len(src))
m = model(dim_src= len(idx), len_cap= int(tgt.shape[1]), src= src, tgt= tgt)

############
# training #
############

saver = tf.train.Saver()
sess = tf.InteractiveSession()
wtr = tf.summary.FileWriter(join(path, "trial{}".format(trial)))

if ckpt:
    saver.restore(sess, ckpt)
else:
    tf.global_variables_initializer().run()

summ = tf.summary.merge((
    tf.summary.scalar('step_acc', m.acc)
    , tf.summary.scalar('step_err0', m.err0)
    , tf.summary.scalar('step_err1', m.err1)
    , tf.summary.scalar('step_err2', m.err2)))
feed_eval = {m.dropout: 0}
feed_pred = {m.dropout: 0, m.src: s, m.tgt: t}

for _ in range(5):
    for _ in tqdm(range(step_save), ncols= 70):
        sess.run(m.up)
        step = sess.run(m.step)
        if not (step % step_eval):
            wtr.add_summary(sess.run(summ, feed_eval), step)
    save("trial/pred/{}_{}.wav".format(step, trial), m.frame.eval(feed_pred)[0])
saver.save(sess, "trial/model/m{}".format(trial), step, write_meta_graph= False)
