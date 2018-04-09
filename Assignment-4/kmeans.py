import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        random_k = np.random.random_integers(0, N, self.n_cluster)
        """ Centres of clusters """
        mu_k = x[random_k]
        """ Current cluster of n-th sample """
        R = np.array( [0]*N )
        """ Initialize J """
        J_old = 0
        for i in range(0, self.max_iter):
            J_new = 0
            """ Find closest cluster for each sample """
            for n in range(0, N):
                distance_xn = np.sum( (mu_k - x[n])**2, axis=1 )
                R[n] = np.argmin( distance_xn )
                J_new += distance_xn[ R[n] ]
            """ Check if significant J change """
            if np.abs(J_new-J_old) < N*self.e:
                return (mu_k, np.array(R), i)
            else:
                J_old = J_new
            """ Update cluster centers """
            def pick_cluster(arr, k):
                matches = np.vectorize(lambda x: 1 if x==k else 0)
                return matches(arr)
            for k in range(0, self.n_cluster):
                upd = np.sum( (x.transpose()*pick_cluster(R,k)).transpose(), axis=0 )
                divide = np.sum(pick_cluster(R,k))
                if not divide == 0:
                    mu_k[k] = upd/divide
        return (mu_k, np.array(R), self.max_iter)
        # DONOT CHANGE CODE BELOW THIS LINE


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        centroids, centroid_x, _ = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e).fit(x)
        centroid_labels = [0]*self.n_cluster
        for i in range(0, self.n_cluster):
            label_counts = dict()
            for n in range(0, N):
                if centroid_x[n]==i:
                    if y[n] in label_counts:
                        label_counts[y[n]] += 1
                    else:
                        label_counts[y[n]] = 1
            centroid_labels[i] = max(label_counts, key=lambda i: label_counts[i])
        centroid = np.array(centroids)
        centroid_labels = np.array(centroid_labels)
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        prediction = [0]*N
        for n in range(0, N):
            closest = np.argmin( np.sum( (self.centroids-x[n])**2, axis=1) )
            prediction[n] = self.centroid_labels[closest]
        return np.array(prediction)
        # DONOT CHANGE CODE BELOW THIS LINE
