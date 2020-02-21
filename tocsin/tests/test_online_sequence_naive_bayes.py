from tocsin.tests.testing import TocsinTestCase
from tocsin.online_sequence_naive_bayes import OnlineSequenceNB
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import tensorflow as tf


class TestOnlineSequenceNaiveBayes(TocsinTestCase):

    # helps find testing data, needs to be changed if tests change directory
    package = 'tocsin.tests'

    def test_fit_first_pass(self):
        clf = OnlineSequenceNB(filter_sequence='CGNNN', mask=True)
        paths = [
            self.get_data_path('small_test_fastq_03.fq'),
            self.get_data_path('small_test_fastq_04.fq'),
        ]
        clf.fit(paths, ['cls1', 'cls2'], zipped=False,
                max_read_length=5,
                )
        exp_counts_01 = np.array([
            [0, 4, 0, 0],
            [0, 0, 4, 0],
            [4, 0, 0, 0],
            [0, 2, 1, 1],
            [0, 1, 0, 3],
        ])
        exp_counts_02 = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
        ])

        assert_array_equal(exp_counts_01, clf.counts['cls1'])
        assert_array_equal(exp_counts_02, clf.counts['cls2'])

        # relies on alpha = 1.0
        exp_prob_01 = np.array([
            [1 / 8, 5 / 8, 1 / 8, 1 / 8],
            [1 / 8, 1 / 8, 5 / 8, 1 / 8],
            [5 / 8, 1 / 8, 1 / 8, 1 / 8],
            [1 / 8, 3 / 8, 2 / 8, 2 / 8],
            [1 / 8, 2 / 8, 1 / 8, 4 / 8]
        ])
        exp_prob_02 = np.array([
            [1 / 5, 2 / 5, 1 / 5, 1 / 5],
            [1 / 5, 1 / 5, 2 / 5, 1 / 5],
            [1 / 5, 1 / 5, 2 / 5, 1 / 5],
            [1 / 5, 1 / 5, 2 / 5, 1 / 5],
            [2 / 5, 1 / 5, 1 / 5, 1 / 5],
        ])
        assert_array_almost_equal(exp_prob_01, clf.probs['cls1'])
        assert_array_almost_equal(exp_prob_02, clf.probs['cls2'])

        assert_array_almost_equal(np.log(exp_prob_01), clf.log_probs['cls1'])
        assert_array_almost_equal(np.log(exp_prob_02), clf.log_probs['cls2'])

        exp_mask_lp_01 = np.log(np.array([
            [5 / 8, 1 / 8, 1 / 8, 1 / 8],
            [1 / 8, 3 / 8, 2 / 8, 2 / 8],
            [1 / 8, 2 / 8, 1 / 8, 4 / 8]
        ]))
        exp_mask_lp_02 = np.log(np.array([
            [1 / 5, 1 / 5, 2 / 5, 1 / 5],
            [1 / 5, 1 / 5, 2 / 5, 1 / 5],
            [2 / 5, 1 / 5, 1 / 5, 1 / 5],
        ]))
        assert_array_almost_equal(exp_mask_lp_01, clf.masked_log_probs['cls1'])
        assert_array_almost_equal(exp_mask_lp_02, clf.masked_log_probs['cls2'])

    def test_joint_log_likelihood(self):
        clf = OnlineSequenceNB(filter_sequence='CGNNN', mask=True)
        paths = [
            self.get_data_path('small_test_fastq_03.fq'),
            self.get_data_path('small_test_fastq_04.fq'),
        ]
        clf.fit(paths, ['cls1', 'cls2'], zipped=False,
                max_read_length=5,
                )

        jll = clf._joint_log_likelihood(tf.constant([
            [
                [0., 1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
             ],
            [
                [0., 1, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ],
            [
                [0., 1, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ],
        ]))

        exp_cls_01_jll = np.array(
            [
                np.sum(np.log([5 / 8, 3 / 8, 4 / 8])),
                np.sum(np.log([1 / 8, 3 / 8, 2 / 8])),
                np.sum(np.log([1 / 8, 3 / 8, 2 / 8])),
            ]
        ) + np.log(1 / 2)

        exp_cls_02_jll = np.array(
            [
                np.sum(np.log([1 / 5, 1 / 5, 1 / 5])),
                np.sum(np.log([1 / 5, 1 / 5, 1 / 5])),
                np.sum(np.log([1 / 5, 1 / 5, 1 / 5])),
            ]
        ) + np.log(1 / 2)

        assert_array_almost_equal(jll['cls1'], exp_cls_01_jll)
        assert_array_almost_equal(jll['cls2'], exp_cls_02_jll)

    def test_predict_log_proba(self):
        clf = OnlineSequenceNB(filter_sequence='CGNNN', mask=True)
        paths = [
            self.get_data_path('small_test_fastq_03.fq'),
            self.get_data_path('small_test_fastq_04.fq'),
            ]
        clf.fit(paths, ['cls1', 'cls2'], zipped=False,
                max_read_length=5,
                )
        clf.predict_log_proba([paths[0]],
                              zipped=False,
                              )
        # TODO assert values

    def test_predict_log_proba_file(self):
        clf = OnlineSequenceNB(filter_sequence='CGNNN', mask=True)
        paths = [
            self.get_data_path('small_test_fastq_03.fq'),
            self.get_data_path('small_test_fastq_04.fq'),
        ]
        clf.fit(paths, ['cls1', 'cls2'], zipped=False,
                max_read_length=5,
                )
        clf.predict_log_proba_file([paths[0]],
                                   zipped=False,
                                   )
        # TODO check values and more cases

    def test_predict_proba_file(self):
        clf = OnlineSequenceNB(filter_sequence='CGNNN', mask=True)
        paths = [
            self.get_data_path('small_test_fastq_03.fq'),
            self.get_data_path('small_test_fastq_04.fq'),
        ]
        clf.fit(paths, ['cls1', 'cls2'], zipped=False,
                max_read_length=5,
                )
        clf.predict_proba_file([paths[0]],
                               zipped=False,
                               )

    def test_predict_file(self):
        clf = OnlineSequenceNB(filter_sequence='CGNNN', mask=True)
        paths = [
            self.get_data_path('small_test_fastq_03.fq'),
            self.get_data_path('small_test_fastq_04.fq'),
        ]
        clf.fit(paths, ['cls1', 'cls2'], zipped=False,
                max_read_length=5,
                )
        clf.predict_file([paths[0]],
                         zipped=False,
                         )

    def test_predict_proba(self):
        clf = OnlineSequenceNB(filter_sequence='CGNNN', mask=True)
        paths = [
            self.get_data_path('small_test_fastq_03.fq'),
            self.get_data_path('small_test_fastq_04.fq'),
        ]
        clf.fit(paths, ['cls1', 'cls2'], zipped=False,
                max_read_length=5,
                )
        clf.predict_proba([paths[0]],
                          zipped=False,
                          )

    def test_predict(self):
        clf = OnlineSequenceNB(filter_sequence='CGNNN', mask=True)
        paths = [
            self.get_data_path('small_test_fastq_03.fq'),
            self.get_data_path('small_test_fastq_04.fq'),
        ]
        clf.fit(paths, ['cls1', 'cls2'], zipped=False,
                max_read_length=5,
                )
        clf.predict([paths[0]],
                    zipped=False,
                    )
