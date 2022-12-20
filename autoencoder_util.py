
import tensorflow as tf


def _parse_function(serialized):
    features = {
        "L": tf.io.FixedLenFeature([1], dtype=tf.int64),
        'ht50_res_embed': tf.io.VarLenFeature(dtype=tf.float32),
        }
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)

    # Get all data
    L = parsed_example['L'][0]
    res_embed_shape = tf.stack([L, 1024])
    res_embed = parsed_example['ht50_res_embed']
    res_embed = tf.cast(res_embed, tf.float32)
    res_embed = tf.sparse.to_dense(res_embed)
    res_embed = tf.reshape(res_embed, res_embed_shape)
    return res_embed,res_embed

def get_batched_dataset(filenames, batch_size=64):
    # settings to read from all the shards in parallel
    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    # list all files
    filenames = tf.io.gfile.glob(filenames)
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)

    # Parse the serialized data in the TFRecords files.
    dataset = dataset.map(lambda x: _parse_function(x))

    # Randomizes input using a window of 2000 elements (read into memory)
    dataset = dataset.shuffle(buffer_size=2000 + 3 * batch_size)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, 1024], [None, 1024]), drop_remainder=True)
    dataset = dataset.repeat()

    return dataset