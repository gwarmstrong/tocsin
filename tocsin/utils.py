import gzip
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import io as tfio

dna_keys_tensor = tf.constant([65, 67, 71, 84])
dna_vals_tensor = tf.constant([0, 1, 2, 3])
dna_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(dna_keys_tensor, dna_vals_tensor), -1)


def load_fastq_tf(file_to_load, max_read_length=100, zipped=True):
    """
    Loads a fastq file into a tensor

    Parameters
    ----------
    file_to_load : str
        path to file to load
    max_read_length : int
        truncate the reads in the file to max_read_length
    zipped : bool
        indicates if the file is gzipped or not

    Returns
    -------
    tf.Tensor
        tensor of one-hot encodings of sequences in fastq

    """
    start = time.time()
    lines = []

    if zipped:
        loader = gzip.open
        args = ()
    else:
        loader = open
        args = ('rb', )
    with loader(file_to_load, *args) as fp:
        all_lines = fp.readlines()
    for i, line in enumerate(tqdm(all_lines)):
        if i % 4 == 1:
            bytes_ = line.rstrip()[:max_read_length]
            lines.append([byte_ for byte_ in bytes_])

    demo_lines = encode_dna(lines)
    end = time.time()
    print("Loaded FQ in: {:.4}s".format(end - start))
    return demo_lines


def encode_dna(ord_list):
    """
    Performs a one-hot encoding of a given list of ord's of characters

    Parameters
    ----------
    ord_list : list of int
        each entry of list corresponds to ord of a character

    Returns
    -------
    tf.Tensor of shape (len(ord_list), l, 4)
        one-hot encoding or elements of ord_list. l is the length of
        lists in ord_list

    """
    ord_list = dna_table.lookup(tf.constant(ord_list))
    encoding = tf.one_hot(ord_list, 4)
    return encoding


def find_matching_sequences(reads, filter_seq, min_motif_score):
    """
    Finds all subsequences of the tensors in reads that match a filter
    called filter_seq. Rejects any subsequence with fewer than
    min_motif_score matches with filter_seq.

    Parameters
    ----------
    reads : tf.Tensor
        tensor of one-hot encodings to scan for matches
    filter_seq : str
        DNA string to match. Can use 'N' for wildcard.
        TODO match motifs with any symbol from:
         https://en.wikipedia.org/wiki/Nucleic_acid_sequence
    min_motif_score : int
        number of (non-N) matches required between a read and filter_seq

    Returns
    -------
    tf.Tensor
        tensor of one-hot encodings of matching subsequences

    """
    # prepare convolutions to search for motif
    filter_ = tf.expand_dims(encode_dna_str(filter_seq), 2)
    activations = tf.nn.convolution(reads, filter_)
    activated_ids = tf.nn.convolution(reads, identity_filter(filter_seq))

    # subset the original sequences to things that matched all the
    # specified bases
    matches_motif = tf.squeeze(
        tf.math.greater_equal(activations, min_motif_score), 2)
    match_sequences = tf.boolean_mask(activated_ids, matches_motif)
    matches_ohe = tf.one_hot(tf.cast(match_sequences, tf.uint8), 4)
    return matches_ohe


def encode_dna_str(sequence_str):
    """
    Encodes a DNA string with one-hot encoding

    Parameters
    ----------
    sequence_str : str
        string to be encoded

    Returns
    -------

    """
    return encode_dna([ord(char) for char in sequence_str])


def dna_encode_bit_manipulation(seq, name='dna_encode'):
    # TODO deprecate
    bytes_ = tfio.decode_raw(seq, tf.uint8)
    bytes_ = tf.bitwise.bitwise_and(bytes_, ~((1 << 6) | (1 << 4)))
    bytes_ = tf.bitwise.right_shift(bytes_, 1)
    mask = tf.bitwise.bitwise_and(bytes_, 2)
    mask = tf.bitwise.right_shift(mask, 1)
    bytes_ = tf.bitwise.bitwise_xor(bytes_, mask)
    return bytes_


def identity_filter(filter_seq):
    motif_length = len(filter_seq)
    identity_filter_ = np.zeros((motif_length, 4, motif_length))
    for i in range(len(filter_seq)):
        identity_filter_[i, :, i] = tf.range(4)
    identity_filter_ = tf.cast(tf.constant(identity_filter_), tf.float32)
    return identity_filter_
