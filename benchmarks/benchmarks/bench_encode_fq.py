import os
import gzip
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import pkg_resources
from tocsin.utils import dna_table
from tocsin.reader import fq_reader
import tensorflow as tf

logging.getLogger('tensorflow').setLevel(logging.ERROR)


class TimeReadingSuite:
    def setup(self):
        path = 'H10.None.4C.8weeks.II.R1.trimmed.filtered.fastq.gz'
        self.abs_path = pkg_resources.resource_filename('tocsin.tests',
                                                        'data/%s' % path)

    def time_read_fq_with_gzip(self):
        with gzip.open(self.abs_path) as fp:
            fp.readlines()

    def time_read_fq_reader_cpp(self):
        fq_reader(bytes(self.abs_path, 'utf-8'))


# class TimeEncodingSuite:
#     """
#     An example benchmark that times the performance of various kinds
#     of iterating over dictionaries in Python.
#     """
#     def setup(self):
#         path = 'H10.None.4C.8weeks.II.R1.trimmed.filtered.fastq.gz'
#         self.abs_path = pkg_resources.resource_filename('tocsin.tests',
#                                                    'data/%s' % path)
#         with gzip.open(self.abs_path) as fp:
#             self.lines = fp.readlines()
#
#         self.max_read_length = 100
#
#
#         self.d = {}
#         for x in range(500):
#             self.d[x] = None
#
#     def time_indexed_list_python_tf(self):
#         lines = []
#         for i, line in enumerate(self.lines):
#             if i % 4 == 1:
#                 bytes_ = line.rstrip()[:self.max_read_length]
#                 lines.append([byte_ for byte_ in bytes_])
#
#         ord_list = dna_table.lookup(tf.constant(lines))
#
#     def time_indexed_list_plus_ohe_python_tf(self):
#         lines = []
#         for i, line in enumerate(self.lines):
#             if i % 4 == 1:
#                 bytes_ = line.rstrip()[:self.max_read_length]
#                 lines.append([byte_ for byte_ in bytes_])
#         ord_list = dna_table.lookup(tf.constant(lines))
#         encoding = tf.one_hot(ord_list, 4)
#
