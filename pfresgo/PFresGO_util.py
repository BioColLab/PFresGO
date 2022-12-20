import csv
import numpy as np
import gzip
import tensorflow as tf

from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser



def load_FASTA(filename):
    # Loads fasta file and returns a list of the Bio SeqIO records
    infile = open(filename, 'rU')
    entries = []
    proteins = []
    for entry in SeqIO.parse(infile, 'fasta'):
        entries.append(str(entry.seq))
        proteins.append(str(entry.id))
    return proteins, entries


def load_GO_annot(filename):
    # Load GO annotations
    onts = ['mf', 'bp', 'cc']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}

    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        goterms[onts[0]] = next(reader)

        next(reader, None)  # skip the headers
        gonames[onts[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        goterms[onts[1]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        goterms[onts[2]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        next(reader, None)  # skip the headers
        counts = {ont: np.zeros(len(goterms[ont]), dtype=float) for ont in onts}
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                goterm_indices = [goterms[onts[i]].index(goterm) for goterm in prot_goterms[i].split(',') if
                                  (goterm != '')]
                prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]))
                prot2annot[prot][onts[i]][goterm_indices] = 1.0
                counts[onts[i]][goterm_indices] += 1.0
    return prot2annot, goterms, gonames, counts


def seq2onehot(seq):
    """Create 26-dim embedding"""
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))

    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    embed_x = [vocab_embed[v] for v in seq]
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

    return seqs_x


def _parse_function_PFresGO(serialized, n_goterms, channels=26,ont='cc'):
    features = {
        "seq_1hot": tf.io.VarLenFeature(dtype=tf.float32),
        ont + "_labels": tf.io.FixedLenFeature([n_goterms], dtype=tf.int64),
        "L": tf.io.FixedLenFeature([1], dtype=tf.int64),
        'ht50_res_embed': tf.io.VarLenFeature(dtype=tf.float32),
    }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)

    # Get all data
    L = parsed_example['L'][0]
    non_padding = tf.ones(L, tf.float32)
    S_shape = tf.stack([L, channels])
    S = parsed_example['seq_1hot']
    S = tf.cast(S, tf.float32)
    S = tf.sparse.to_dense(S)
    S = tf.reshape(S, S_shape)

    res_embed_shape = tf.stack([L, 1024])
    res_embed = parsed_example['ht50_res_embed']
    res_embed = tf.cast(res_embed, tf.float32)
    res_embed = tf.sparse.to_dense(res_embed)
    res_embed = tf.reshape(res_embed, res_embed_shape)

    labels = parsed_example[ont + '_labels']
    labels = tf.cast(labels, tf.float32)

    y = tf.reshape(labels, shape=[n_goterms])
    return {'seq': S, 'padding': non_padding, 'res_embed': res_embed}, y



def get_batched_dataset(filenames, batch_size=64, pad_len=1000, n_goterms=347, channels=26, ont='cc'):
    # settings to read from all the shards in parallel
    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    # list all files
    filenames = tf.io.gfile.glob(filenames)
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)

    # Parse the serialized data in the TFRecords files.
    dataset = dataset.map(lambda x: _parse_function_PFresGO(x, n_goterms=n_goterms, channels=channels,ont=ont))

    # Randomizes input using a window of 2000 elements (read into memory)
    dataset = dataset.shuffle(buffer_size=2000 + 3 * batch_size)
    dataset = dataset.padded_batch(batch_size, padded_shapes=({'seq': [pad_len, channels], 'padding': [pad_len], 'res_embed': [pad_len, 1024]}, [None]), drop_remainder=True)
    dataset = dataset.repeat()

    return dataset



def go_embeddings_to_dict(go_embed_pth):
    embeddings_dict = {}
    embeddings = open(go_embed_pth).read().splitlines()[1:]
    embeddings = [x.split(" ") for x in embeddings]

    for i in range(0, len(embeddings)):
        # set the GO id as the key
        key = int(embeddings[i][0])
        # add all the dimension of the embedings as a list of floats
        embeddings_dict[key] = [float(x) for x in embeddings[i][1:]]

    return embeddings_dict

def read_fasta(fn_fasta):
    prot2seq = {} #prot:seq, not one-hot
    if fn_fasta.endswith('gz'):
        handle = gzip.open(fn_fasta, "rt")
    else:
        handle = open(fn_fasta, "rt")

    for record in SeqIO.parse(handle, "fasta"):
        seq = str(record.seq)
        prot = record.id
        prot2seq[prot] = seq
    return prot2seq


class GroupWiseLinear(tf.keras.layers.Layer):

    def __init__(self, num_class, hidden_dim, bias=True, **kwargs):
        super(GroupWiseLinear, self).__init__(**kwargs)
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.output_layer = tf.keras.layers.Dense(num_class, activation='sigmoid', name='dense_out')
        W_init = tf.random_normal_initializer(mean=0.0, stddev=0.2)
        self.W = tf.Variable(initial_value=W_init(shape=(1, num_class, hidden_dim),dtype='float32'), trainable=True, name="W")

        if bias:
            b_init = tf.random_normal_initializer(mean=0.0, stddev=0.2)
            self.b = tf.Variable(initial_value=b_init(shape=(1, num_class), dtype='float32'),trainable=True, name="b")

    def call(self, x):
        # x: B,K,d
        x = (self.W * x)
        x = tf.reduce_sum(x, axis=-1)
        if self.bias:
            x = x + self.b
        out = self.output_layer(x)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_class': self.num_class,
            "hidden_dim":self.hidden_dim,
            "bias":self.bias
            })
        return config