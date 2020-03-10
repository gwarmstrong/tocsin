import time

import numpy as np
import tensorflow as tf
from sklearn.base import ClassifierMixin, BaseEstimator
from tocsin.utils import load_fastq_tf, find_matching_sequences, SliceTensor


class OnlineSequenceNB(ClassifierMixin, BaseEstimator):

    # TODO refactor so fit and predict are on filenames, labels, and the there is something like
    #  predict_sequence for single sequence (I think this is done)
    # TODO filter_sequence should not be set by default
    # TODO filter_sequence should take a list
    def __init__(self, filter_sequence, alpha=1.0,
                 motif_tolerance=0, mask=False, min_motif_score=None):
        self.alpha = 1.0
        self.counts = SliceTensor()
        self.filter_sequence = filter_sequence
        self.motif_tolerance = motif_tolerance
        self.mask = mask
        if self.mask:
            self.mask_sequence = [l is 'N' for l in self.filter_sequence]
        else:
            self.mask_sequence = None

        self.min_motif_score = self._get_min_motif_score()

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

        sample_weights = self._validate_sample_weights(X_files, file_labels,
                                                       sample_weights)

        total_start = time.time()
        for i, (file_, label, weight) in enumerate(
                zip(X_files, file_labels, sample_weights)):
            print('----------------------------')
            print('Starting training iteration {}...'.format(i))
            start = time.time()
            # update counts for label with counts form X_files
            matching_sequences = self._prepare_sequences(
                file_, self.min_motif_score, self.filter_sequence,
                zipped=zipped, max_read_length=max_read_length,
                )

            self._update_counts(matching_sequences, label, weight)
            end = time.time()
            print('Done. {:.2f}s\tTotal: {:.2f}s'.format(end - start,
                                                         end - total_start))

        self._update_log_prob()
        self.classes_ = np.array(self.counts.index)
        print('----------------------------')
        return self

    def _update_counts(self, matching_sequences, label, weight):
        counts = tf.reduce_sum(matching_sequences, axis=0)
        if weight != 1:
            counts = tf.scalar_mul(weight, counts)
        # potential for not liking it when adding two tensors...
        if label not in self.counts:
            self.counts[label] = counts
        else:
            self.counts[label] += counts

    @staticmethod
    def _validate_sample_weights(X_files, file_labels, sample_weights):
        # TODO make this check better later (create validation methods)
        assert len(X_files) == len(file_labels)
        if sample_weights is None:
            sample_weights = np.ones(len(X_files))
        else:
            assert len(sample_weights) == len(X_files)
        return sample_weights

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
        # set float32 because defaults to double but needs lower prec...
        return {label: np.log(1 / len(self.counts), dtype=np.float32) for
                label in self.counts.index}

    def _update_log_prob(self):
        smoothed_counts = self.counts.slicetensor_op(tf.add, self.alpha)
        cat_counts_by_class = smoothed_counts.tensor_op(tf.reduce_sum, axis=2,
                                                        keepdims=True)

        self.probs = smoothed_counts.slicetensor_op(tf.divide,
                                                    cat_counts_by_class)

        self.log_probs = self.probs.slicetensor_op(tf.math.log)
        if self.mask:
            self.masked_log_probs = self.log_probs.slicetensor_op(
                tf.boolean_mask, self.mask_sequence, axis=1)
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

        jlls = {label: tf.multiply(X, log_probs[label]) for label in
                log_probs.index}

        jlls = SliceTensor.from_dict(jlls)
        # log_prior = self.class_log_prior_
        # jlls2 = {
        #     label: tf.reduce_sum(jlls[label], axis=[1, 2]) + log_prior[label]
        #     for label in jlls.index}
        # jlls2 = {
        #     label: tf.add(tf.reduce_sum(jlls[label], axis=[1, 2]), log_prior[
        #         label])
        #     for label in jlls.index}
        # print("JLL", jlls2)
        # print("prior", log_prior)
        log_prior = SliceTensor.from_dict(self.class_log_prior_)\
            .tensor_op(tf.expand_dims, axis=1)
        jlls = jlls.slicetensor_op(tf.reduce_sum, axis=[2, 3])
        # print("JLL", jlls2)
        # print("prior", log_prior)
        jlls = jlls.slicetensor_op(tf.add, log_prior)

        return jlls

    def predict_log_proba(self, X_files, zipped=True):
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
                                                         zipped=zipped,
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

    def predict_log_proba_file(self, X_files, zipped=True):
        """
        Returns a total probability for each file in X_files

        Parameters
        ----------
        X_files : iterable of file paths
            each file in the list contains reads from
        zipped : bool
            indicates if the files passed in are zipped or not

        Returns
        -------

        """
        all_log_probs = self.predict_log_proba(X_files, zipped=zipped)
        return [tf.reduce_sum(log_proba, axis=1) for log_proba in
                all_log_probs]

    def predict_proba_file(self, X_files, zipped=True):
        # TODO kind of breaks the convention of the other methods
        all_log_probs = self.predict_log_proba_file(X_files, zipped=zipped)
        tf_test_log_probs = tf.stack(all_log_probs)
        norm_factors = tf.reduce_logsumexp(tf_test_log_probs, axis=1,
                                           keepdims=True)
        probs = tf.exp(tf_test_log_probs - norm_factors)
        return probs

    def predict_file(self, X_files, zipped=True):
        """
        Returns a single class for the whole file
        """
        all_log_probs_file = self.predict_log_proba_file(X_files,
                                                         zipped=zipped)
        # TODO fix __iter__ here
        arg_maxes = [tf.argmax(log_prob) for log_prob in all_log_probs_file]
        return [self.classes_[idx.numpy()] for idx in arg_maxes]

    def predict_proba(self, X_files, zipped=True):
        all_log_probs = self.predict_log_proba(X_files, zipped=zipped)
        return [tf.exp(log_prob) for log_prob in all_log_probs]

    def predict(self, X_files, zipped=True):
        all_log_probs = self.predict_log_proba(X_files, zipped=zipped)
        max_indexes = [tf.argmax(log_prob, axis=0) for log_prob in
                       all_log_probs]
        return [self.classes_[idx.numpy()] for idx in max_indexes]
