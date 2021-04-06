import sys
import time

from sklearn.datasets import make_classification
from sklearn.neighbors import NearestCentroid
import numpy as np
import warnings
from numpy import inf
from scipy import sparse as sp
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.utils.sparsefuncs import csc_median_axis_0
from sklearn.utils.multiclass import check_classification_targets


class NCMClassifier(NearestCentroid):
    def __init__(self, metric, sub_features, shrink_treshold=None):
        """

        :param metric: string , distance {euclidean, manhattan , mahalanobis...}
        :param shrink_treshold: float ,     Threshold for shrinking centroids to remove features.
        :variable inv_cov_vectors: storage of the diagonal of the inverse covariance matrix per class
        :variable nk: class effective
        :variable sub_features: selected features
        """
        NearestCentroid.__init__(self, metric=metric)  # init sklearn Nearest Centroid source
        self.inv_cov_vectors = None
        self.nk = None
        self.sub_features = sub_features

    def fit(self, X, y):
        """
        Fit the NearestCentroid model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
            Note that centroid shrinking cannot be used with sparse matrices.
        y : array, shape = [n_samples]
            Target values (integers)
        """
        if self.metric == 'precomputed':
            raise ValueError("Precomputed is not supported.")
        # If X is sparse and the metric is "manhattan", store it in a csc
        # format is easier to calculate the median.
        if self.metric == 'manhattan':
            X, y = check_X_y(X, y, ['csc'])
        else:
            X, y = check_X_y(X, y, ['csr', 'csc'])
        is_X_sparse = sp.issparse(X)
        if is_X_sparse and self.shrink_threshold:
            raise ValueError("threshold shrinking not supported"
                             " for sparse input")
        check_classification_targets(y)

        n_samples, n_features = X.shape
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        self.classes_ = classes = le.classes_
        n_classes = classes.size
        eps = 1e-8  # epsilon to avoid singular matrix error
        # np.seterr(divide='ignore', invalid='ignore')  # Warning ! Some matrixes are nan (division by 0)
        #if n_classes < 2:
        #    raise ValueError('The number of classes has to be greater than'
        #                     ' one; got %d class' % (n_classes))

        # Mask mapping each class to its members.
        self.centroids_ = np.empty((n_classes, n_features), dtype=np.float64)
        # Number of clusters in each class.
        nk = np.zeros(n_classes)
        self.inv_cov_vectors = []

        for cur_class in range(n_classes):
            center_mask = y_ind == cur_class
            nk[cur_class] = np.sum(center_mask)

            if is_X_sparse:
                center_mask = np.where(center_mask)[0]

            # XXX: Update other averaging methods according to the metrics.
            if self.metric == "manhattan":
                # NumPy does not calculate median of sparse matrices.
                if not is_X_sparse:
                    self.centroids_[cur_class] = np.median(X[center_mask], axis=0)
                else:
                    self.centroids_[cur_class] = csc_median_axis_0(X[center_mask])
            else:
                self.centroids_[cur_class] = X[center_mask].mean(axis=0)

                if X[center_mask].shape[0] == 1:
                    inv_cov_vector = np.ones(X.shape[1])
                else:
                    cov_vector = np.var(X[center_mask].T, axis=1)
                    inv_cov_vector = np.round(1/(cov_vector+eps), 3)
                    inv_cov_vector[inv_cov_vector == inf] = 1

                self.inv_cov_vectors.append(inv_cov_vector)
        self.inv_cov_vectors = np.array(self.inv_cov_vectors)
        self.nk = nk
        if self.shrink_threshold:
            dataset_centroid_ = np.mean(X, axis=0)

            # m parameter for determining deviation
            m = np.sqrt((1. / nk) - (1. / n_samples))
            # Calculate deviation using the standard deviation of centroids.
            variance = (X - self.centroids_[y_ind]) ** 2
            variance = variance.sum(axis=0)
            s = np.sqrt(variance / (n_samples - n_classes))
            s += np.median(s)  # To deter outliers from affecting the results.
            mm = m.reshape(len(m), 1)  # Reshape to allow broadcasting.
            ms = mm * s
            deviation = ((self.centroids_ - dataset_centroid_) / ms)
            # Soft thresholding: if the deviation crosses 0 during shrinking,
            # it becomes zero.
            signs = np.sign(deviation)
            deviation = (np.abs(deviation) - self.shrink_threshold)
            np.clip(deviation, 0, None, out=deviation)
            deviation *= signs
            # Now adjust the centroids using the deviation
            msd = ms * deviation
            self.centroids_ = dataset_centroid_[np.newaxis, :] + msd
        return self

    def predict(self, X):
        """Perform classification on an array of test vectors X.
        The predicted class C for each sample in X is returned.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        C : ndarray of shape (n_samples,)
        Notes
        -----
        If the metric constructor parameter is "precomputed", X is assumed to
        be the distance matrix between the data to be predicted and
        ``self.centroids_``.
        """
        # check_is_fitted(self)
        X = check_array(X, accept_sparse='csr')
        X = X[:, self.sub_features]
        if self.metric == "mahalanobis":
            distances = []
            for centroid, vector in zip(self.centroids_[:, self.sub_features], self.inv_cov_vectors[:, self.sub_features]):
                delta = X - centroid
                distance = np.sqrt(np.matmul(delta * delta, vector))
                distances.append(distance)

            distances = np.array(distances)

            return self.classes_[distances.argmin(axis=0)]
        else:
            return self.classes_[pairwise_distances(
                X, self.centroids_[:, self.sub_features], metric=self.metric).argmin(axis=1)]
