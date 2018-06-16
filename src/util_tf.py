import tensorflow as tf


def batch(data, batch_size, shuffle= 1e4, repeat= True, name= "batch"):
    """returns a tensorflow dataset iterator from `data`."""
    with tf.variable_scope(name):
        ds = tf.data.Dataset.from_tensor_slices(data)
        if shuffle: ds = ds.shuffle(int(shuffle))
        if repeat:  ds = ds.repeat()
        return ds.batch(batch_size) \
                 .make_one_shot_iterator() \
                 .get_next()


def profile(path, sess, run, feed_dict= None, prerun= 3, tag= "step"):
    for _ in range(prerun): sess.run(run, feed_dict)
    meta = tf.RunMetadata()
    sess.run(run, feed_dict, tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE), meta)
    with tf.summary.FileWriter(path, sess.graph) as wtr:
        wtr.add_run_metadata(meta, tag)
