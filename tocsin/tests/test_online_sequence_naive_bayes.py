from tocsin.tests.testing import TocsinTestCase
from tocsin.online_sequence_naive_bayes import OnlineSequenceNB
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


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
        pass


