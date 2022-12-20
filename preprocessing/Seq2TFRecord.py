import csv
from Bio import SeqIO
import os.path
import gzip
import argparse
import h5py

import numpy as np
import tensorflow as tf
import multiprocessing


def read_fasta(fn_fasta):
    aa = set(['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E'])
    prot2seq = {} #prot:seq, not one-hot
    if fn_fasta.endswith('gz'):
        handle = gzip.open(fn_fasta, "rt")
    else:
        handle = open(fn_fasta, "rt")

    for record in SeqIO.parse(handle, "fasta"):
        seq = str(record.seq)
        prot = record.id
        pdb, chain = prot.split('_') if '_' in prot else prot.split('-')
        prot = pdb.upper() + '-' + chain
        if len(seq) >= 60 and len(seq) <= 1000:
            if len((set(seq).difference(aa))) == 0:
                prot2seq[prot] = seq

    return prot2seq

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

def prot_id2res_embed(prot_id):
    index = str(prot_id) + ' ' + 'nrPDB'
    with h5py.File('../Datasets/per_residue_embeddings.h5', "r") as f:
        res = f[index][:]
    return res

def prot_id2prot_embed(prot_id):
    index = str(prot_id) + ' ' + 'nrPDB'
    with h5py.File('../Datasets/per_protein_embeddings.h5', "r") as f:
        res = f[index][:]
    return res


def load_list(fname):
    #Load PDB chains
    pdb_chain_list = []
    fRead = open(fname, 'r')
    for line in fRead:
        pdb_chain_list.append(line.strip())
    fRead.close()
    return pdb_chain_list


def load_GO_annot(filename):
    """ Load GO annotations """
    onts = ['molecular_function', 'biological_process', 'cellular_component']
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
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                goterm_indices = [goterms[onts[i]].index(goterm) for goterm in prot_goterms[i].split(',') if (goterm != '')]
                prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]), dtype=np.int64)
                prot2annot[prot][onts[i]][goterm_indices] = 1.0
    return prot2annot, goterms, gonames


class GenerateTFRecord(object):
    def __init__(self, prot_list, prot2annot, tfrecord_fn, prot2seq,num_shards=30):
        self.prot_list = prot_list
        self.prot2annot = prot2annot
        self.tfrecord_fn = tfrecord_fn
        self.num_shards = num_shards
        self.prot2seq = prot2seq

        shard_size = len(prot_list)//num_shards
        indices = [(i*(shard_size), (i+1)*(shard_size)) for i in range(0, num_shards)]
        indices[-1] = (indices[-1][0], len(prot_list))
        self.indices = indices

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _dtype_feature(self):
        return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))

    def _int_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value])) #tf.train.Int64List的value值需要是int数据的列表

    def _serialize_example(self, prot_id, sequence, ca_dist_matrix=None, cb_dist_matrix=None):
        labels = self._dtype_feature()

        d_feature = {}
        # load appropriate tf.train.Featur class depending on dtype
        d_feature['prot_id'] = self._bytes_feature(prot_id.encode())
        d_feature['seq_1hot'] = self._float_feature(seq2onehot(sequence).reshape(-1))
        d_feature['L'] = self._int_feature(len(sequence))

        d_feature['ht50_res_embed'] = self._float_feature(prot_id2res_embed(prot_id).reshape(-1))
        d_feature['ht50_prot_embed'] = self._float_feature(prot_id2prot_embed(prot_id).reshape(-1))


        d_feature['mf_labels'] = labels(self.prot2annot[prot_id]['molecular_function'])
        d_feature['bp_labels'] = labels(self.prot2annot[prot_id]['biological_process'])
        d_feature['cc_labels'] = labels(self.prot2annot[prot_id]['cellular_component'])


        example = tf.train.Example(features=tf.train.Features(feature=d_feature))
        return example.SerializeToString()

    def _convert_numpy_folder(self, idx):
        tfrecord_fn = self.tfrecord_fn + '_%0.2d-of-%0.2d.tfrecords' % (idx, self.num_shards)
        # writer = tf.python_io.TFRecordWriter(tfrecord_fn)
        writer = tf.io.TFRecordWriter(tfrecord_fn)
        print("### Serializing %d examples into %s" % (len(self.prot_list), tfrecord_fn))

        tmp_prot_list = self.prot_list[self.indices[idx][0]:self.indices[idx][1]]

        for i, prot in enumerate(tmp_prot_list):
            if i % 500 == 0:
                print("### Iter = %d/%d" % (i, len(tmp_prot_list)))
            sequence = str(prot2seq[prot])
            example = self._serialize_example(prot, sequence)
            writer.write(example)
        print("Writing {} done!".format(tfrecord_fn))

    def run(self, num_threads):
        pool = multiprocessing.Pool(processes=num_threads)
        shards = [idx for idx in range(0, self.num_shards)]
        pool.map(self._convert_numpy_folder, shards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-annot', type=str, default='../Datasets/nrPDB-GO_2019.06.18_annot.tsv', help="Input file (*.tsv) with preprocessed annotations.")
    parser.add_argument('-prot_list', type=str, default='../Datasets/nrPDB-GO_2019.06.18_valid.txt', help="Input file (*.txt) with a set of protein PDB IDs.")
    parser.add_argument('-seqres', type=str, default='../Datasets/nrPDB-GO_2019.06.18_sequences.fasta', help="PDB chain seqres fasta.")
    parser.add_argument('-num_threads', type=int, default=3, help="Number of threads (CPUs) to use in the computation.")
    parser.add_argument('-num_shards', type=int, default=3, help="Number of tfrecord files per protein set.")
    parser.add_argument('-tfr_prefix', type=str, default='../Datasets/TFRecords_sequences/PDB_GO_valid', help="Directory with tfrecord files for model training.")
    args = parser.parse_args()

    prot_list = load_list(args.prot_list)
    prot2seq = read_fasta(args.seqres)
    prot2annot, _, _ = load_GO_annot(args.annot)

    tfr = GenerateTFRecord(prot_list, prot2annot, args.tfr_prefix,prot2seq, num_shards=args.num_shards)
    tfr.run(num_threads=args.num_threads)
