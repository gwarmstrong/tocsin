from tocsin.tests.testing import TocsinTestCase
from tocsin.utils import (
    load_fastq_tf, encode_dna, dna_encode_bit_manipulation,
    find_matching_sequences, encode_dna_str
)
import numpy as np
from numpy.testing import assert_array_equal


class TestLoadFastqTF(TocsinTestCase):

    package = 'tocsin.tests'

    def test_load_fastq_tf_zipped_false(self):
        demo_file = self.get_data_path('small_test_fastq_01.fq')
        obs_lines = load_fastq_tf(demo_file,
                                  zipped=False,
                                  max_read_length=5,
                                  )
        assert_array_equal(obs_lines.shape, [4, 4, 4])
        exp_0 = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ])
        exp_1 = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ])
        exp_2 = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        assert_array_equal(obs_lines[0], exp_0)
        assert_array_equal(obs_lines[1], exp_1)
        assert_array_equal(obs_lines[2], exp_2)

    def test_load_fastq_tf_zipped_true(self):
        demo_file = self.get_data_path('small_test_fastq_01.fq.gz')
        obs_lines = load_fastq_tf(demo_file,
                                  zipped=True,
                                  max_read_length = 4,
                                  )
        exp_0 = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ])
        exp_1 = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ])
        exp_2 = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        assert_array_equal(obs_lines[0], exp_0)
        assert_array_equal(obs_lines[1], exp_1)
        assert_array_equal(obs_lines[2], exp_2)


class TestEncoding(TocsinTestCase):

    def test_encode_dna(self):
        str_ = [ord("A"), ord("G"), ord("T"), ord("C"), ord("G")]
        exp = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ])
        obs = encode_dna(str_)
        assert_array_equal(exp, obs)

    # TODO remove (when deprecating encode_bit_manipulation)
    def test_dna_encode_bit_manipulation(self):
        str_ = 'AGTCG'
        exp = np.array([0, 2, 3, 1, 2])
        obs = dna_encode_bit_manipulation(str_)
        assert_array_equal(exp, obs)


class TestFindMatchingSequences(TocsinTestCase):

    package = 'tocsin.tests'

    def test_find_matching_sequences_sample_01(self):
        demo_file = self.get_data_path('small_test_fastq_01.fq')
        filter_seq = 'CGNN'
        reads = load_fastq_tf(demo_file,
                              zipped=False,
                              max_read_length=5,
                              )
        matches = find_matching_sequences(reads, filter_seq, 2)

        # assertion depends on fact that all lines match filter_seq fully
        assert_array_equal(reads, matches)

    def test_find_matching_sequences_sample_02(self):
        demo_file = self.get_data_path('small_test_fastq_02.fq')
        filter_seq = 'CGN'
        reads = load_fastq_tf(demo_file,
                              zipped=False,
                              max_read_length=5,
                              )
        matches = find_matching_sequences(reads, filter_seq, 2)
        # expected by parsing file for pattern manually
        exp_01 = encode_dna_str('CGT')
        exp_02 = encode_dna_str('CGG')
        exp_03 = encode_dna_str('CGC')
        assert_array_equal(exp_01, matches[0])
        assert_array_equal(exp_02, matches[1])
        assert_array_equal(exp_03, matches[2])

        filter_seq = 'ANC'
        matches = find_matching_sequences(reads, filter_seq, 2)
        # expected by parsing file for pattern manually
        exp_04 = encode_dna_str('AAC')
        exp_05 = encode_dna_str('AGC')
        assert_array_equal(exp_04, matches[0])
        assert_array_equal(exp_05, matches[1])
