import time

import numpy as np
import tensorflow as tf
from sklearn.base import ClassifierMixin, BaseEstimator
from tocsin.utils import load_fastq_tf, find_matching_sequences


class OnlineSequenceNB(ClassifierMixin, BaseEstimator):

    # TODO refactor so fit and predict are on filenames, labels, and the there is something like
    #  predict_sequence for single sequence (I think this is done)
    # TODO filter_sequence should not be set by default
    # TODO filter_sequence should take a list
    def __init__(self, filter_sequence, alpha=1.0,
                 motif_tolerance=0, mask=False, min_motif_score=None):
        self.alpha = 1.0
        self.counts = dict()
        self.filter_sequence = filter_sequence
        self.motif_tolerance = motif_tolerance
        self.mask = mask
        if self.mask:
            self.mask_sequence = [l is 'N' for l in self.filter_sequence]
        else:
            self.mask_sequence = None

    @staticmethod
    def _prepare_sequences(file_, max_motif_score, filter_sequence,
                           max_read_length=100,
                           zipped=True,
                           ):
        file_sequences = load_fastq_tf(file_,
                                       max_read_length=max_read_length,
                                       zipped=zipped,
                                       )
        matching_sequences = find_matching_sequences(file_sequences,
                                                     filter_sequence,
                                                     max_motif_score,
                                                     )
        return matching_sequences

    def fit(self, X_files, file_labels, sample_weights=None,
            zipped=True,
            max_read_length=100,
            ):
        """
        Fits a Naive Bayes model to all files in X_files. All sequences
        in a file must have the same label.
        TODO currently max_read_length must be no greater than the minimum
         length of reads in each file.
        TODO need better strategy for handling different read sizes

        Parameters
        ----------
        X_files : iterable of str
            each item contains a path to a file to use for training data
        file_labels : iterable
            Same length as X_files. Contains the label of the class for all
            sequences in the corresonding entry of X_files
        sample_weights
            can be used to weight samples differently
        zipped : bool
            indicates whether the files in X_files are zipped
            TODO allow iterable so that zipped and non-zipped can be mixed
        max_read_length : int
            the length to trunate the reads to
            TODO if max_read_length is longer than the smallest read,
            then we will get an error

        Returns
        -------
        self
            The fit classifier.

        """
        # WARNING set up to persist counts through multiple fits
        # assumes that each file contains a single class

        # TODO make this check better later (create validation methods)
        assert len(X_files) == len(file_labels)
        if sample_weights is None:
            sample_weights = np.ones(len(X_files))
        else:
            assert len(sample_weights) == len(X_files)

        self.min_motif_score = self._get_min_motif_score()

        total_start = time.time()
        for i, (file_, label, weight) in enumerate(
                zip(X_files, file_labels, sample_weights)):
            print('----------------------------')
            print('Starting training iteration {}...'.format(i))
            start = time.time()
            # update coutns for label with counts form X_files
            matching_sequences = self._prepare_sequences(
                file_,
                self.min_motif_score,
                self.filter_sequence,
                zipped=zipped,
                max_read_length=max_read_length,
                )
            counts = tf.reduce_sum(matching_sequences, axis=0)
            if weight != 1:
                counts = tf.scalar_mul(weight, counts)
            # potential for not liking it when adding two tensors...
            if label not in self.counts:
                self.counts[label] = counts
            else:
                self.counts[label] = tf.math.add(self.counts[label], counts)
            end = time.time()
            print('Done. {:.2f}s\tTotal: {:.2f}s'.format(end - start,
                                                         end - total_start))

        self._update_log_prob()
        self.classes_ = np.array(list(self.counts.keys()))
        print('----------------------------')
        return self

    def _get_min_motif_score(self):
        max_motif_score = sum(letter is not 'N' for letter in
                              self.filter_sequence)
        min_motif_score = max_motif_score - self.motif_tolerance
        return min_motif_score

    @property
    def class_log_prior_(self):
        """
        For now this is just uniform prior...
        """
        return {label: tf.math.log(1 / len(self.counts)).numpy() for label in
                self.counts}

    def _update_log_prob(self):
        smoothed_counts = {label: tf.add(self.alpha, self.counts[label]) for
                           label in self.counts}
        cat_counts_by_class = {label: tf.reduce_sum(sc, axis=1, keepdims=True)
                               for label, sc in smoothed_counts.items()}
        self.probs = {label: tf.divide(smoothed_counts[label],
                                       cat_counts_by_class[label])
                      for label in smoothed_counts}
        self.log_probs = {label: tf.math.log(prob) for label, prob in
                          self.probs.items()}
        if self.mask:
            self.masked_log_probs = {
                label: self._index_on_mask_sequence(log_prob) for
                label, log_prob
                in self.log_probs.items()}
            return self.probs, self.log_probs, self.masked_log_probs

        return self.probs, self.log_probs

    def _index_on_mask_sequence(self, X, axis=None):
        return tf.boolean_mask(X, self.mask_sequence, axis=axis)

    def _joint_log_likelihood(self, X):
        """
        Assumes X is a tensor with dims (n_samples, len_seq (only N's if
        self.mask), n_bases)
        """
        if self.mask:
            # index X on mask
            log_probs = self.masked_log_probs
            X = self._index_on_mask_sequence(X, axis=1)
        else:
            log_probs = self.log_probs

        jlls = {label: tf.multiply(X, log_probs[label]) for label in log_probs}
        log_prior = self.class_log_prior_
        jlls = {
            label: tf.reduce_sum(jlls[label], axis=[1, 2]) + log_prior[label]
            for label in jlls}
        return jlls

    def predict_log_proba(self, X_files):
        """
        Depends on having been fit
        """
        log_probs = []
        for i, file_ in enumerate(X_files):
            print('----------------------------')
            print('Starting prediction iteration {}...'.format(i))
            start = time.time()
            # update counts for label with counts form X_files
            matching_sequences = self._prepare_sequences(file_,
                                                         self.min_motif_score,
                                                         self.filter_sequence,
                                                         )
            jlls = self._joint_log_likelihood(matching_sequences)
            jlls = tf.stack([jlls[class_] for class_ in self.classes_])
            # if len(jlls.shape) == 1:
            #   jlls = tf.expand_dims(jlls, 0)
            # return jlls
            norm_factor = tf.reduce_logsumexp(jlls, 0)
            # return norm_factor
            log_probs.append(jlls - norm_factor)
        # return jlls
        # TODO may want to transpose eventually
        return log_probs

    def predict_log_proba_file(self, X_files):
        """
        Returns a single probability for the whole file
        """
        all_log_probs = self.predict_log_proba(X_files)
        return [tf.reduce_sum(log_proba, axis=1) for log_proba in
                all_log_probs]

    def predict_file(self, X_files):
        """
        Returns a single class for the whole file
        """
        all_log_probs_file = self.predict_log_proba_file(X_files)
        arg_maxes = [tf.argmax(log_prob) for log_prob in all_log_probs_file]
        return [self.classes_[idx.numpy()] for idx in arg_maxes]

    def predict_probab_file(self, X_files):
        # TODO kind of breaks the convention of the other methods
        all_log_probs = self.predict_log_proba_file(X_files)
        tf_test_log_probs = tf.stack(all_log_probs)
        norm_factors = tf.reduce_logsumexp(tf_test_log_probs, axis=1,
                                           keepdims=True)
        probs = tf.exp(tf_test_log_probs - norm_factors)
        return probs

    def predict_proba(self, X_files):
        all_log_probs = self.predict_log_proba(X_files)
        return [tf.exp(log_prob) for log_prob in all_log_probs]

    def predict(self, X_files):
        all_log_probs = self.predict_log_proba(X_files)
        max_indexes = [tf.argmax(log_prob, axis=0) for log_prob in
                       all_log_probs]
        return [self.classes_[idx.numpy()] for idx in max_indexes]